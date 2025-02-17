import json
import os

# === Configuration ===
# Path to your input JSON file (COCO format)
json_file = "/Users/jens-jakobskotingerslev/Desktop/4 semester/harborfront/Test.json"
# Directory where YOLO-format txt files will be saved
labels_dir = "/labels"

# Create the output directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# === Load JSON Data ===
with open(json_file, "r") as f:
    data = json.load(f)

# === Build Category Mapping ===
# This dictionary will map original COCO category IDs to new 0-indexed YOLO class IDs.
# Here we ignore the "background" category if its name is "background".
cat_mapping = {}
for cat in data["categories"]:
    # Ignore background if desired (adjust this condition if needed)
    if cat["name"].lower() == "background":
        continue
    # Assign a new index based on the order (0-indexed)
    cat_mapping[cat["id"]] = len(cat_mapping)

# === Build Image and Annotation Dictionaries ===
# Create a dictionary to look up image info by image_id
images_dict = {img["id"]: img for img in data["images"]}

# Group annotations by image_id for quick lookup
annotations_by_image = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    annotations_by_image.setdefault(image_id, []).append(ann)

# === Process Each Image ===
for image_id, img_info in images_dict.items():
    file_name = img_info["file_name"]
    img_width = img_info["width"]
    img_height = img_info["height"]

    # Derive the output txt file name based on the image file name
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    out_file = os.path.join(labels_dir, base_name + ".txt")

    # Get all annotations for this image (if there are any)
    anns = annotations_by_image.get(image_id, [])

    with open(out_file, "w") as f_out:
        for ann in anns:
            # Get the original category id and map it to our YOLO class index.
            orig_cat_id = ann["category_id"]
            # Skip annotations that are not in our mapping (for example, background)
            if orig_cat_id not in cat_mapping:
                continue
            yolo_class = cat_mapping[orig_cat_id]

            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann["bbox"]

            # Convert COCO bbox (top-left x,y and width, height) to YOLO bbox (center x, center y, width, height)
            x_center = x + w / 2.0
            y_center = y + h / 2.0

            # Normalize coordinates by image dimensions
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # Write out the annotation in YOLO format
            f_out.write(f"{yolo_class} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("Conversion complete!")
