import json
import os

# Sample JSON content from 'info.json'
with open('datasets/vehicle/info.json') as f:
    data = json.load(f)

# Directory where you want to save the YOLO txt files
output_dir = "datasets/labels2/training"
os.makedirs(output_dir, exist_ok=True)

# Assuming you know the image dimensions (replace with actual image dimensions)
image_width = 96  # Example image width
image_height = 96  # Example image height


# Function to convert the bounding boxes to YOLO format
def convert_to_yolo_format(bbox, image_width, image_height):
    x = bbox['x']
    y = bbox['y']
    w = bbox['width']
    h = bbox['height']

    # Normalize the bounding box coordinates
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    width = w / image_width
    height = h / image_height

    return x_center, y_center, width, height


# Process each image entry in the JSON
for image_info in data:
    image_name = image_info['name']
    bounding_boxes = image_info['boundingBoxes']

    # YOLO file path
    yolo_file_path = os.path.join(output_dir, f"{image_name}.txt")

    # Prepare data for YOLO file
    with open(yolo_file_path, 'w') as f:
        for bbox in bounding_boxes:
            obj_class = bbox[
                'label']  # This should be converted to an integer class label (e.g., mapping "bus" to a number)

            # Convert the bounding box to YOLO format
            x_center, y_center, width, height = convert_to_yolo_format(bbox, image_width, image_height)

            # Write YOLO formatted bounding box (assuming label is a string, adjust as needed)
            yolo_line = f"{obj_class} {x_center} {y_center} {width} {height}\n"
            f.write(yolo_line)

    print(f"YOLO file created for {image_name}: {yolo_file_path}")

print("Conversion complete!")
