import json
import os
out_dir = "datasets/vehicle/labels/training"
# os.makedirs(out_dir, exist_ok=True)
# Load JSON file
with open('datasets/vehicle-edge-671-4ctg/ei-vehicle-detection-bd-v5-image-y_training.json') as f:
    data = json.load(f)

# Assuming you know the image dimensions
image_width = 96  # replace with actual image width
image_height = 96  # replace with actual image height

# Convert JSON to YOLO format

for sample in data:
    yolo_data = []
    sample_id = sample['sampleId']
    for box in sample['boundingBoxes']:
        obj_class = box['label']
        x = box['x']
        y = box['y']
        w = box['w']
        h = box['h']

        # Convert to YOLO format (normalize values)
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        width = w / image_width
        height = h / image_height

        # Prepare YOLO format string
        yolo_format = f"{obj_class} {x_center} {y_center} {width} {height}"
        yolo_data.append(yolo_format)

    output_file = os.path.join(out_dir, f"{sample_id}.txt")
    with open(output_file, 'w') as f:
        for line in yolo_data:
            f.write(line + '\n')

print("Conversion complete!")
