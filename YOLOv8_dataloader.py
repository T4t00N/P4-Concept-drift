import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image  # Added for converting arrays to PIL images
from torchvision import transforms  # For data augmentation (optional)

class YOLOv8COCODataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        """
        json_file: Path to COCO annotations file (e.g., test.json, train.json, etc.)
        root_dir: Path to dataset root (where 'frames/' is located)
        transform: Data augmentation / preprocessing (optional)
        """
        self.root_dir = root_dir
        self.transform = transform

        # Load JSON
        with open(json_file, "r") as f:
            self.coco_data = json.load(f)
        
        # Create lookup dictionaries
        self.image_id_to_annotations = {}
        for ann in self.coco_data["annotations"]:
            image_id = ann["image_id"]
            bbox = ann["bbox"]
            category_id = ann["category_id"]

            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            
            # COCO bbox format: [x_min, y_min, width, height]
            self.image_id_to_annotations[image_id].append((bbox, category_id))

        # Create image list, ensuring the clip folders exist
        self.image_list = []
        for img in self.coco_data["images"]:
            img_path = os.path.join(self.root_dir, img["file_name"])
            img_id = img["id"]
            img_width, img_height = img["width"], img["height"]
            
            # Include only images that have annotations and whose directories exist
            clip_dir = os.path.dirname(img_path)
            if os.path.exists(clip_dir):  # Check if the clip folder exists
                if img_id in self.image_id_to_annotations:
                    self.image_list.append((img_path, img_id, img_width, img_height))
            else:
                continue
               # print(f"Warning: Skipping missing directory {clip_dir}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path, img_id, img_width, img_height = self.image_list[idx]

        # Load thermal image in grayscale mode
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Ensures a single-channel image
    
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert to a 3D tensor (1, H, W) to match PyTorch expectations
        img = np.expand_dims(img, axis=0)  # Shape becomes (1, H, W)

        # Convert the image to PIL Image before applying transforms
        img = Image.fromarray(img[0])  # Convert to PIL Image (single channel)

        # Get bounding boxes
        bboxes = []
        for (bbox, category_id) in self.image_id_to_annotations[img_id]:
            x_min, y_min, width, height = bbox

            # Convert to YOLO format: [class_id, x_center, y_center, width, height]
            x_center = (x_min + width / 2) / img_width
            y_center = (y_min + height / 2) / img_height
            width = width / img_width
            height = height / img_height

            bboxes.append([category_id, x_center, y_center, width, height])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)  # Apply transform to PIL Image

        # Convert the image back to a tensor after applying transforms
        img = torch.tensor(np.array(img), dtype=torch.float32)

        # Convert to the proper shape (C, H, W)
        img = img.unsqueeze(0)  # Shape becomes (1, H, W)

        # Debugging line to check what is being returned
        # print(f"Returning img: {img.shape}, bboxes: {bboxes.shape}, image path: {img_path}")

        return img, bboxes, img_path  # Ensure only two values are returned
