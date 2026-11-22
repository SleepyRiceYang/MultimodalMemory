"""Shared procedural-only MAGNET evolution prompts; experience images vary only at planning."""
ABSTRACT = '''Distill reusable GUI workflows from the supplied SUCCESSFUL task trajectories.
Each cluster was formed from similar task instructions. Abstract common intent and operation steps;
preserve distinct valid execution paths. Use only recorded actions and visible observations.
Replace instance-specific names, numbers, dates, paths and values with <parameter_name> slots.
Do not generalize on to off, create to delete, or draft to send. Do not invent missing actions.
The postcondition must be visible in the representative AFTER image, not an expected future result.
Never copy source click coordinates into instructions. Return at most TWO workflows.
Return JSON {"workflows":[{"title":"task category", "app":"application", "precondition":"...",
"goal":"...", "plan":["high-level executable instruction"], "postcondition":"...",
"parameters":[{"name":"slot_name","value":"literal observed in source"}],
"source_task_ids":["supplied ID"], "representative_task_id":"one supplied supporting ID"}]}.
Use 1-12 plan steps. An empty workflows list is valid when evidence is inadequate.'''

VERIFY = '''Verify a proposed abstract GUI workflow against the supplied successful source actions
and their actual screenshots. Reject invented operations, wrong operation direction, copied coordinates,
or a postcondition not visibly achieved in the representative AFTER screenshot. A selected item or open
dialog is not proof of save/open/creation completion. Source overall success alone is insufficient
proof of an unrelated subtask. The plan must apply to every declared supporting task.
Return JSON {"supported":true/false,"before_evidence":"...","after_evidence":"...","reason":"..."}.
Reject when uncertain. Do not solve the current evaluation task or consult hidden answers.'''

PLANNER = '''You are the planner of a procedural-memory GUI agent.
Use the CURRENT observation, current user goal and current action history to choose useful subtasks.
Relevant Workflows are abstract experience from OTHER executions, not current facts or completed work.
Check preconditions, bind slots to the CURRENT goal, and adapt steps to the CURRENT visible state.
If a workflow is unrelated or has the opposite operation direction, ignore it. Do not copy example
values or screen coordinates. Historical experience images, if present, ONLY illustrate that workflow;
they do not show the current GUI. Only the image labelled CURRENT OBSERVATION determines current state.
Return JSON {"subtasks":["short task-specific step"],"next_subtask":"next step or verify completion",
"reason":"brief grounding in current goal and observation"}. Use 1-8 subtasks. Preserve all goal constraints.
Do not output executable actions; the actor grounds your next subtask using current screenshots.'''

ACTOR_NOTICE = '''MAGNET planner guidance follows. Execute the next subtask using CURRENT observations
and the original user goal. The plan is guidance, not evidence of task completion. Verify actual results;
ignore a plan that conflicts with the current goal or screen. Use the native action schema exactly.'''
