import os
import glob
from utils.dataset import Dataset

def main():
    # Replace this path with your own images directory
    image_directory = '/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset/images/test'

    # Collect all image filenames
    filenames = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]:
        filenames.extend(glob.glob(os.path.join(image_directory, ext)))

    # Instantiate the Dataset
    input_size = 640
    params = {}  # pass any additional parameters you need
    augment = False  # set to True if you want to apply augmentation
    ds = Dataset(filenames, input_size=input_size, params=params, augment=augment)

    # Print the number of images
    print(f"Number of images: {len(ds)}")

    # Print some of the file paths
    print("Here are a few file paths from the dataset:")
    for i in range(min(5, len(ds))):  # show up to 5
        print(ds.filenames[i])

if __name__ == "__main__":
    main()
