# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing_extensions import List, Tuple

import json
import enum
import imagesize
# import jax
# import jax.numpy as np
import numpy as np


def extract_gt_action(example):
    ex_action_type = example['result_action_type']

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = example['result_lift_yx']
        touch_yx = example['result_touch_yx']
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            action_type = 'CLICK'
            w, h = imagesize.get(example['image_full_path'])
            click_y, click_x = round(lift_yx[0] * h), round(lift_yx[1] * w)
            action_detail = (click_x, click_y)
        else:
            action_type = 'SWIPE_FINGER'
            v_change = abs(touch_yx[0] - lift_yx[0])
            h_change = abs(lift_yx[1] - touch_yx[1])
            is_scroll_up = lift_yx[0] < touch_yx[0]     # touch is lower
            is_scroll_left = lift_yx[1] < touch_yx[1]   # touch is bigger
            if v_change >= 0.9*h_change:   # vertical
                action_detail = "up" if is_scroll_up else "down"
            else: action_detail = "left" if is_scroll_left else "right"
    elif ex_action_type in (
        ActionType.PRESS_BACK, 
        ActionType.PRESS_HOME,
        ActionType.PRESS_ENTER,
    ):  
        button = ActionType(ex_action_type).name.split('_')[1].lower()
        action_detail, action_type = button, 'PRESS'
    elif ex_action_type == ActionType.TYPE:
        action_text = example['result_action_text']
        action_type, action_detail = 'SET_TEXT', action_text
    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        action_type, action_detail = 'DONE', ''
    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        action_type, action_detail = 'IMPOSSIBLE', ''
    else: raise NotImplementedError
    
    return (action_type.lower(), action_detail)


'''======================================
                Global Args
   ======================================'''


_TAP_DISTANCE_THRESHOLD = 0.14    # Fraction of the screen
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4

# Interval determining if an action is a tap or a swipe.
_SWIPE_DISTANCE_THRESHOLD = 0.04


def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = np.linalg.norm(
        np.array(normalized_start_yx) - np.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD


def _is_non_dual_point_action(action_type):
    return np.not_equal(action_type, ActionType.DUAL_POINT)


'''======================================
   === AndroidInTheWild action types. ===     
   ======================================'''


class ActionType(enum.IntEnum):
    """Integer values for each supported action type in AndroidInTheWild."""

    # Placeholders for unused enum values
    UNUSED_0 = 0
    UNUSED_1 = 1
    UNUSED_2 = 2
    UNUSED_8 = 8
    UNUSED_9 = 9

    ########### Agent actions ###########

    # A type action that sends text to the emulator. Note that this simply sends
    # text and does not perform any clicks for element focus or enter presses for
    # submitting text.
    TYPE = 3

    # The dual point action used to represent all gestures.
    DUAL_POINT = 4

    # These actions differentiate pressing the home and back button from touches.
    # They represent explicit presses of back and home performed using ADB.
    PRESS_BACK = 5
    PRESS_HOME = 6

    # An action representing that ADB command for hitting enter was performed.
    PRESS_ENTER = 7

    ########### Episode status actions ###########

    # An action used to indicate the desired task has been completed and resets
    # the environment. This action should also be used in the case that the task
    # has already been completed and there is nothing to do.
    # e.g. The task is to turn on the Wi-Fi when it is already on
    STATUS_TASK_COMPLETE = 10

    # An action used to indicate that desired task is impossible to complete and
    # resets the environment. This can be a result of many different things
    # including UI changes, Android version differences, etc.
    STATUS_TASK_IMPOSSIBLE = 11


'''===============================================================================
   ===     Utilites for performing action matching on AndroidInTheWild data.   ===
   ===     Note: this code is implemented using JAX so it can be 'vmap'ed and  ===
   ===     efficiently appied over a batch of data.                            ===
   ==============================================================================='''


def _yx_in_bounding_boxes(
    yx, bounding_boxes
):
    """Check if the (y,x) point is contained in each bounding box.

    Args:
        yx: The (y, x) coordinate in pixels of the point.
        bounding_boxes: A 2D int array of shape (num_bboxes, 4), where each row
            represents a bounding box: (y_top_left, x_top_left, box_height,
            box_width). Note: containment is inclusive of the bounding box edges.

    Returns:
        is_inside: A 1D bool array where each element specifies if the point is
            contained within the respective box.
    """
    y, x = yx

    # `bounding_boxes` has shape (n_elements, 4); we extract each array along the
    # last axis into shape (n_elements, 1), then squeeze unneeded dimension.
    top, left, height, width = [
        np.squeeze(v, axis=-1) for v in np.split(bounding_boxes, 4, axis=-1)
    ]

    # The y-axis is inverted for AndroidEnv, so bottom = top + height.
    bottom, right = top + height, left + width

    return np.logical_and(y >= top, y <= bottom) & np.logical_and(x >= left, x <= right)


def _resize_annotation_bounding_boxes(
    annotation_positions, annotation_width_augment_fraction,
    annotation_height_augment_fraction):
    """Resize the bounding boxes by the given fractions.

    Args:
        annotation_positions: Array of shape (N, 4), where each row represents the
            (y, x, height, width) of the bounding boxes.
        annotation_width_augment_fraction: The fraction to augment the box widths,
            E.g., 1.4 == 240% total increase.
        annotation_height_augment_fraction: Same as described for width, but for box
            height.

    Returns:
        Resized bounding box.

    """
    height_change = (
            annotation_height_augment_fraction * annotation_positions[:, 2])
    width_change = (
            annotation_width_augment_fraction * annotation_positions[:, 3])

    # Limit bounding box positions to the screen.
    resized_annotations = np.stack([
        np.maximum(0, annotation_positions[:, 0] - (height_change / 2)),
        np.maximum(0, annotation_positions[:, 1] - (width_change / 2)),
        np.minimum(1, annotation_positions[:, 2] + height_change),
        np.minimum(1, annotation_positions[:, 3] + width_change),
    ],axis=1)

    return resized_annotations


def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = np.linalg.norm(
        np.array(normalized_start_yx) - np.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD


def _is_non_dual_point_action(action_type):
    return np.not_equal(action_type, ActionType.DUAL_POINT)


def _check_tap_actions_match(
    tap_1_yx,
    tap_2_yx,
    annotation_positions,
    matching_tap_distance_threshold_screen_percentage,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
    """Determines if two tap actions are the same."""
    resized_annotation_positions = _resize_annotation_bounding_boxes(
        annotation_positions,
        annotation_width_augment_fraction,
        annotation_height_augment_fraction,
    )

    # Check if the ground truth tap action falls in an annotation's bounding box.
    tap1_in_box = _yx_in_bounding_boxes(tap_1_yx, resized_annotation_positions)
    tap2_in_box = _yx_in_bounding_boxes(tap_2_yx, resized_annotation_positions)
    both_in_box = np.max(tap1_in_box & tap2_in_box)

    # If the ground-truth tap action falls outside any of the annotation
    # bounding boxes or one of the actions is inside a bounding box and the other
    # is outside bounding box or vice versa, compare the points using Euclidean
    # distance.
    within_threshold = (
        np.linalg.norm(np.array(tap_1_yx) - np.array(tap_2_yx))
        <= matching_tap_distance_threshold_screen_percentage
    )
    return np.logical_or(both_in_box, within_threshold)


def _check_drag_actions_match(
    drag_1_touch_yx,
    drag_1_lift_yx,
    drag_2_touch_yx,
    drag_2_lift_yx,
):
    """Determines if two drag actions are the same."""
    # Store drag deltas (the change in the y and x coordinates from touch to
    # lift), magnitudes, and the index of the main axis, which is the axis with
    # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
    # ending at (0.3, 0.5) has a main axis index of 1).
    drag_1_deltas = drag_1_lift_yx - drag_1_touch_yx
    drag_1_magnitudes = np.abs(drag_1_deltas)
    drag_1_main_axis = np.argmax(drag_1_magnitudes)
    drag_2_deltas = drag_2_lift_yx - drag_2_touch_yx
    drag_2_magnitudes = np.abs(drag_2_deltas)
    drag_2_main_axis = np.argmax(drag_2_magnitudes)

    return np.equal(drag_1_main_axis, drag_2_main_axis)


def check_actions_match(
    action_1_touch_yx,
    action_1_lift_yx,
    action_1_action_type,
    action_2_touch_yx,
    action_2_lift_yx,
    action_2_action_type,
    annotation_positions,
    tap_distance_threshold = _TAP_DISTANCE_THRESHOLD,
    annotation_width_augment_fraction = ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction = ANNOTATION_HEIGHT_AUGMENT_FRACTION,
):
    """Determines if two actions are considered to be the same.

    Two actions being "the same" is defined here as two actions that would result
    in a similar screen state.

    Args:
        action_1_touch_yx: The (y, x) coordinates of the first action's touch.
        action_1_lift_yx: The (y, x) coordinates of the first action's lift.
        action_1_action_type: The action type of the first action.
        action_2_touch_yx: The (y, x) coordinates of the second action's touch.
        action_2_lift_yx: The (y, x) coordinates of the second action's lift.
        action_2_action_type: The action type of the second action.
        annotation_positions: The positions of the UI annotations for the screen. It
            is A 2D int array of shape (num_bboxes, 4), where each row represents a
            bounding box: (y_top_left, x_top_left, box_height, box_width). Note that
            containment is inclusive of the bounding box edges.
        tap_distance_threshold: The threshold that determines if two taps result in
            a matching screen state if they don't fall the same bounding boxes.
        annotation_width_augment_fraction: The fraction to increase the width of the
            bounding box by.
        annotation_height_augment_fraction: The fraction to increase the height of
            of the bounding box by.

    Returns:
        A boolean representing whether the two given actions are the same or not.
    """
    action_1_touch_yx = np.asarray(action_1_touch_yx)
    action_1_lift_yx = np.asarray(action_1_lift_yx)
    action_2_touch_yx = np.asarray(action_2_touch_yx)
    action_2_lift_yx = np.asarray(action_2_lift_yx)

    # Checks if at least one of the actions is global (i.e. not DUAL_POINT),
    # because if that is the case, only the actions' types need to be compared.
    has_non_dual_point_action = np.logical_or(
        _is_non_dual_point_action(action_1_action_type),
        _is_non_dual_point_action(action_2_action_type),
    )

    different_dual_point_types = np.logical_xor(
        is_tap_action(action_1_touch_yx, action_1_lift_yx),
        is_tap_action(action_2_touch_yx, action_2_lift_yx),
    )

    is_tap = np.logical_and(
        is_tap_action(action_1_touch_yx, action_1_lift_yx),
        is_tap_action(action_2_touch_yx, action_2_lift_yx),
    )

    taps_match = _check_tap_actions_match(
        action_1_touch_yx,
        action_2_touch_yx,
        annotation_positions,
        tap_distance_threshold,
        annotation_width_augment_fraction,
        annotation_height_augment_fraction,
    )

    taps_match = np.logical_and(is_tap, taps_match)

    drags_match = _check_drag_actions_match(
        action_1_touch_yx, action_1_lift_yx, action_2_touch_yx, action_2_lift_yx
    )
    drags_match = np.where(is_tap, False, drags_match)

    return np.where(
        has_non_dual_point_action,
        np.equal(action_1_action_type, action_2_action_type),
        np.where(
            different_dual_point_types,
            False,
            np.logical_or(taps_match, drags_match),
        ),
    )



'''==============================================================
   ===         User added Functions for Data Procesing        ===
   =============================================================='''


def match_ui_bbox(
    tap_rel_yx, width, height,
    annotation_abs_positions,
    annotation_width_augment_fraction = ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction = ANNOTATION_HEIGHT_AUGMENT_FRACTION,
    return_augment_bbox=True
) -> Tuple[int, int, int, int]:
    """Finds the bounding box that contains the given tap action.
    
    Args:
        tap_rel_yx: The (y, x) relative coordinates of the tap, in the range 0-1.
        width, height: The height and width of the image.
        annotation_abs_positions: The absolute positions of the UI annotations for the screen. It
            is A 2D int array of shape (num_bboxes, 4), where each row represents a
            bounding box: (y_top_left, x_top_left, box_height, box_width). Note that
            containment is inclusive of the bounding box edges.
        annotation_width_augment_fraction: The fraction to increase the width of the
            bounding box by.
        annotation_height_augment_fraction: The fraction to increase the height of
            of the bounding box by.
    
    Returns:
        matched bounding box in the form of (y_top_left, x_top_left, box_height, box_width)
    """
    annotation_abs_positions = np.asarray(annotation_abs_positions)
    
    # transform the absolute coords into relative coords
    annotation_positions = annotation_abs_positions / np.array([height, width, height, width])
    
    # resized_annotation_positions = _resize_annotation_bounding_boxes(
    #     annotation_positions,
    #     annotation_width_augment_fraction,
    #     annotation_height_augment_fraction,
    # )
    resized_annotation_positions = annotation_positions

    # Check if the ground truth tap action falls in an annotation's bounding box.
    tap_in_box = _yx_in_bounding_boxes(tap_rel_yx, resized_annotation_positions)
    # Return all index of the matched bbox
    bbox_ids = np.where(tap_in_box)[0]

    if return_augment_bbox:
        resized_abs_annotation_positions = resized_annotation_positions * np.array([height, width, height, width])
        matched_bboxes = resized_abs_annotation_positions[bbox_ids]
    else: matched_bboxes = annotation_abs_positions[bbox_ids]

    if len(bbox_ids) > 1:
        # Calculate the distances between the tap_yx with the center of each matched bbox
        tap_yx = np.array([tap_rel_yx[0] * height, tap_rel_yx[1] * width])
        bbox_center = np.stack([
            matched_bboxes[:, 0] + matched_bboxes[:, 2] / 2, 
            matched_bboxes[:, 1] + matched_bboxes[:, 3] / 2], axis=-1)
        # distances has shape (N,)
        distances = np.linalg.norm(tap_yx - bbox_center)
        # Select the minimum distance as the matched bbox
        min_distance_index = np.argmin(distances)
        single_matched_bbox = matched_bboxes[min_distance_index]
    elif len(bbox_ids) == 1: single_matched_bbox = matched_bboxes.squeeze()
    else: single_matched_bbox = np.array([-1, -1, 0, 0])
    
    ui_bbox_tuple = tuple(single_matched_bbox.astype(np.int64).tolist())
    return ui_bbox_tuple
