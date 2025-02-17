import json
import os

# === Configuration ===
# Path to your input JSON file (COCO format)
json_file = "/Users/jens-jakobskotingerslev/Desktop/4 semester/harborfront/Valid.json"
# Directory where YOLO-format txt files will be saved
labels_dir = "labels/"

# Create the base output directory if it doesn't exist
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

# === Counters for debugging ===
image_count_json = len(images_dict)
annotation_count_json = len(annotations_by_image)
files_created_count = 0
images_without_annotations = 0
images_with_annotations = 0

# === Process Each Image ===
for image_id, img_info in images_dict.items():
    file_name = img_info["file_name"]
    img_width = img_info["width"]
    img_height = img_info["height"]

    # === Extract directory structure from file_name ===
    dirs = file_name.split(os.sep) # Split by OS path separator to handle different systems
    if len(dirs) >= 3: # Assuming year/clip_id/image_name.jpg structure
        year_dir = dirs[0]
        clip_dir = dirs[1]
        image_name_with_ext = dirs[2]
    else:
        print(f"Warning: Unexpected file_name format: {file_name}.  Using image_id as filename.")
        year_dir = "flat_structure" # Or some default name
        clip_dir = "flat_structure"
        image_name_with_ext = file_name


    image_base_name = os.path.splitext(image_name_with_ext)[0]

    # === Create output directories mirroring input structure ===
    output_clip_dir = os.path.join(labels_dir, year_dir, clip_dir)
    os.makedirs(output_clip_dir, exist_ok=True) # Create year and clip dirs if needed

    # === Construct the output txt file path ===
    out_file = os.path.join(output_clip_dir, image_base_name + ".txt")


    # Get all annotations for this image (if there are any)
    anns = annotations_by_image.get(image_id, [])

    if not anns:
        images_without_annotations += 1
    else:
        images_with_annotations += 1

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
        files_created_count += 1 # Increment only if file is opened

print("=== Debugging Information ===")
print(f"Total images in JSON: {image_count_json}")
print(f"Total images with annotations in JSON: {annotation_count_json} (This count represents unique image_ids that have annotations)")
print(f"Text files created: {files_created_count}")
print(f"Images without annotations: {images_without_annotations}")
print(f"Images with annotations: {images_with_annotations}")
print("Conversion complete!")