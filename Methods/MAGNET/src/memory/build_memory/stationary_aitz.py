from memory.core.paths import dataset_path
from memory.datasets.aitz import AITZDataset
from tqdm import tqdm

dataset = AITZDataset(
    split="train_omni",
    data_dir=str(dataset_path("android_in_the_zoo")),
    ratio=1.0,
    chunks=1,
    chunk_idx=0,
    click_only=False,
    swipe_only=False
)

json_data = dataset.data
cropped_num = 0
for item in tqdm(json_data):
    image_path = item['image_path']
    image_full_path = dataset_path("android_in_the_zoo", "train", image_path)
    matched_ui_bbox = item['matched_ui_bbox']
    if matched_ui_bbox is not None:
        # break  # 只取第一个样本
        cropped_num += 1
print(cropped_num)
# from PIL import Image, ImageDraw
# import imagesize
# # 打开图像
# img = Image.open(image_full_path).convert("RGB")
# width, height = imagesize.get(image_full_path)
# # 绘图对象
# drawer = ImageDraw.Draw(img)
# print(matched_ui_bbox)

# def cropped_image(image, bbox):
#     """
#     裁剪图像到指定的边界框
#     :param image: PIL图像对象
#     :param bbox: 边界框 [x1, y1, x2, y2]
#     :return: 裁剪后的图像
#     """
#     return image.crop(bbox)

# # 假设 matched_ui_bbox 是 [x1, y1, x2, y2]
# if matched_ui_bbox and len(matched_ui_bbox) == 4:
#     y_top_left, x_top_left, box_height, box_width = matched_ui_bbox
#     # drawer.rectangle([x1, y1, x2, y2], outline="red", width=3)
#     drawer.rectangle([x_top_left, y_top_left, x_top_left + box_width, y_top_left + box_height], outline="red", width=3)
#     xmin, ymin, xmax, ymax = x_top_left, y_top_left, x_top_left + box_width, y_top_left + box_height
#     # 裁剪图像
#     cropped_img = cropped_image(img, (xmin, ymin, xmax, ymax))


# # 显示图像
# img.show()

# # 可选：保存图像
# img.save("/root/GUI-Memory/source/build_memory/output_bbox.jpg")
