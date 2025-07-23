import json
import os

# Đường dẫn tới file JSON COCO và nơi lưu YOLO labels
coco_json_path = r'F:\INTERN-MOBI\PROJECT-INTERN\Find-Parking-Slot-and-Car\annotations\valid\_annotations.coco.json'
output_dir = 'F:\INTERN-MOBI\PROJECT-INTERN/Find-Parking-Slot-and-Car/datasets/valid/'  # nơi lưu file .txt
os.makedirs(output_dir, exist_ok=True)

# Đọc file JSON
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# Tạo từ điển ánh xạ ID ảnh -> (file_name, width, height)
image_id_map = {
    image['id']: {
        'file_name': image['file_name'],
        'width': image['width'],
        'height': image['height']
    }
    for image in coco_data['images']
}

# Ánh xạ category_id gốc -> chỉ số liên tục từ 0
category_ids = sorted({cat['id'] for cat in coco_data['categories']})
category_id_map = {orig_id: idx for idx, orig_id in enumerate(category_ids)}

# Duyệt annotation và ghi file YOLO
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    category_id = category_id_map[ann['category_id']]
    x, y, w, h = ann['bbox']

    # Thông tin ảnh
    img_info = image_id_map[image_id]
    img_w, img_h = img_info['width'], img_info['height']

    # Chuyển bbox từ (x_min, y_min, w, h) -> (x_center, y_center, w, h) (normalized)
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h

    # Tên file txt tương ứng
    txt_filename = os.path.splitext(img_info['file_name'])[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)

    # Ghi file (append nếu có nhiều annotation trong 1 ảnh)
    with open(txt_path, "a") as f:
        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
