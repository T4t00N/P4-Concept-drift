import torch
import os
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from YOLOv8_dataloader import YOLOv8COCODataset
from torchvision import transforms

# Define transforms (optional)
transform = transforms.Compose([
    transforms.Resize((640,640)),
    transforms.ToTensor(),
])

def visualize_image_with_bboxes(image_path, json_file, root_dir, transform=None):
    # Initialize the dataset
    dataset = YOLOv8COCODataset(
        json_file=json_file,
        root_dir=root_dir,
        transform=None
    )

    # Iterate through the dataset and find the image
    for data in dataset:
        img, bboxes, img_path = data
        if img_path == image_path:  # Check if this is the correct image
            print(f"Found image: {img_path}")
            break
    else:
        print(f"Image path {image_path} not found in the dataset.")
        return
    
    # Load image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Plot the image
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img, cmap='gray')
    
    # Iterate over bounding boxes and plot them
    bboxes_list = bboxes.tolist()
    
    for bbox in bboxes_list:
        print(f"bbox: {bbox}")
        
        class_id, x_center, y_center, width, height = bbox

        # Convert from relative to absolute values (in pixels)
        x_min = int((x_center - width / 2) * img.shape[1])
        y_min = int((y_center - height / 2) * img.shape[0])
        box_width = int(width * img.shape[1])
        box_height = int(height * img.shape[0])

        # Create a rectangle for the bounding box
        rect = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Optionally, add the label (category)
        ax.text(x_min, y_min, f"Class {int(class_id)}", color='r', fontsize=12, verticalalignment='top')

    # Display the image with bounding boxes
    plt.show()

if __name__ == "__main__":
    # User input for the image path
    image_path_input = input("Enter the image path you want to visualize: ")

    # Define the dataset and path to the JSON file and image folder
    json_file = "harborfront/Test.json"
    root_dir = "harborfront/"

    # Visualize the image with bounding boxes
    visualize_image_with_bboxes(image_path_input, json_file, root_dir, transform)

