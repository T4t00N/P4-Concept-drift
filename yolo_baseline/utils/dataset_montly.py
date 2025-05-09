import math
import os
import random
import cv2
import numpy
import torch
import re
from PIL import Image
from torch.utils import data
from tqdm import tqdm

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment, month_filter=None):
        self.params = params
        self.augment = augment
        self.input_size = input_size

        # Filter by month if specified
        if month_filter:
            filenames = self.filter_by_month(filenames, month_filter)
            print(f"Filtered to {len(filenames)} images from month: {month_filter}")

        # Count images by month
        self.count_images_by_month(filenames)

        # Read labels
        cache = self.load_label(filenames, month_filter)
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = numpy.array(shapes, dtype=numpy.float64)
        self.filenames = list(cache.keys())
        self.n = len(shapes)
        self.indices = range(self.n)

    def __getitem__(self, index):
        index = self.indices[index]

        # Load image
        image, shape = self.load_image(index)
        h, w = image.shape[:2]

        # Resize
        image, ratio, pad = resize_static(image, self.input_size)
        shapes = shape, ((h / shape[0], w / shape[1]), pad)

        # Get labels and transform them from normalized [x,y,w,h] to pixel [x1,y1,x2,y2]
        label = self.labels[index].copy()
        if label.size:
            label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])

        # Convert [x1,y1,x2,y2] back to normalized [x,y,w,h]
        nl = len(label)
        if nl:
            label[:, 1:5] = xy2wh(label[:, 1:5], image.shape[1], image.shape[0])

        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)

        # Add channel dimension for grayscale
        sample = image[numpy.newaxis, :, :]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample), target, shapes

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i], cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[:2]
        return image, (h, w)

    @staticmethod
    def filter_by_month(filenames, month):
        """Filter filenames by month (format: MM) using the full path"""
        month_pattern = re.compile(r'2021' + month)
        filtered_files = [f for f in filenames if month_pattern.search(f)]
        return filtered_files

    @staticmethod
    def count_images_by_month(filenames):
        """Count images by month using the full path"""
        month_counts = {}
        for f in filenames:
            match = re.search(r'2021(\d{2})', f)
            if match:
                month = match.group(1)
                month_counts[month] = month_counts.get(month, 0) + 1
        print("\n--- Image Count by Month ---")
        for month, count in sorted(month_counts.items()):
            print(f"Month {month}: {count} images")
        print("---------------------------\n")
        return month_counts

    @staticmethod
    def collate_fn(batch):
        samples, targets, shapes = zip(*batch)
        for i, item in enumerate(targets):
            item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

    @staticmethod
    def load_label(filenames, month_filter=None, person_only=False):
        # Setup cache directory
        cache_dir = "monthly_cache"
        os.makedirs(cache_dir, exist_ok=True)

        base_name = os.path.basename(os.path.dirname(filenames[0]))
        month_suffix = f"_{month_filter}" if month_filter else ""
        label_cache_filename = f"{base_name}{month_suffix}_label_cache_02.pt"
        label_cache_path = os.path.join(cache_dir, label_cache_filename)
        print(f"Cache path: {label_cache_path}")

        # If the label cache already exists, load it
        if os.path.exists(label_cache_path):
            print(f"Loading label cache from {label_cache_path}")
            full_cache = torch.load(label_cache_path)
            filtered_cache = {k: v for k, v in full_cache.items() if k in filenames}
            print(f"Loaded and filtered cache to {len(filtered_cache)} entries\n")
            return filtered_cache

        # Otherwise build it from scratch
        print(f"Cache not found at {label_cache_path}")
        print(f"Creating label cache for {len(filenames)} images…")
        label_dict = {}
        for filename in tqdm(filenames, total=len(filenames), desc="Building cache"):
            try:
                with open(filename, 'rb') as f:
                    img = Image.open(f)
                    img.verify()
                shape = img.size

                # derive label path
                img_folder = os.sep + "images" + os.sep
                lbl_folder = os.sep + "labels" + os.sep
                split = filename.rsplit(img_folder, 1)
                lbl_path = lbl_folder.join(split).rsplit('.', 1)[0] + '.txt'

                # read or init empty
                if os.path.isfile(lbl_path):
                    lines = open(lbl_path).read().strip().splitlines()
                    arr = numpy.array([l.split() for l in lines], dtype=numpy.float32)
                    # dedupe + sanitize...
                    if arr.size:
                        _, idx = numpy.unique(arr, axis=0, return_index=True)
                        arr = arr[idx]
                else:
                    arr = numpy.zeros((0, 5), dtype=numpy.float32)

                if arr.size:
                    # Define the number of expected classes from params if available,
                    # otherwise default to 2 (0 and 1) as per your YAML.
                    # This assumes params['names'] is accessible or self.params is available
                    # If not directly available, you might need to pass num_classes to load_label
                    # For now, we'll hardcode based on your stated intent for 2 classes.
                    num_expected_classes = 2  # Based on your YAML (0: background, 1: person)
                    max_valid_class_index = num_expected_classes - 1

                    # The first column in 'arr' is the class_id.
                    # Filter out labels with class indices outside the expected range [0, max_valid_class_index].
                    valid_class_indices_mask = (arr[:, 0] >= 0) & (arr[:, 0] <= max_valid_class_index)

                    if not numpy.all(valid_class_indices_mask):
                        print(
                            f"WARNING: File {lbl_path} contains class indices outside the expected range [0, {max_valid_class_index}].")
                        print(
                            f"Original labels in this file (class_id is the first column):\n{arr[~valid_class_indices_mask]}")
                        arr = arr[valid_class_indices_mask]
                        if not arr.size:
                            print(
                                f"INFO: After filtering, no valid labels (classes 0-{max_valid_class_index}) remain in {lbl_path}.")
                            # Ensure arr is correctly shaped if empty
                            arr = numpy.zeros((0, 5), dtype=numpy.float32)
                        else:
                            print(f"INFO: Keeping the following valid labels in {lbl_path}:\n{arr}")

                    # Ensure 'arr' has the correct 2D shape even if it becomes empty or has one row after filtering
                    if arr.ndim == 1 and arr.size > 0:  # handles case of a single valid label becoming a 1D array
                        arr = arr.reshape(1, -1)
                    elif not arr.size:  # handles case where all labels were invalid or file was empty
                        arr = numpy.zeros((0, 5), dtype=numpy.float32)
                # ---- END: Insert new code block here ----

                label_dict[filename] = [arr, shape]
            except Exception:
                continue

        # Save to cache & report
        torch.save(label_dict, label_cache_path)
        print(f"Saved new label cache ({len(label_dict)} entries) to {label_cache_path}\n")
        return label_dict


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # [x, y, w, h] normalized -> [x1, y1, x2, y2] in pixels
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
    return y


def xy2wh(x, w=640, h=640):
    # [x1, y1, x2, y2] in pixels -> [x, y, w, h] normalized
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)

    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y


def resize_static(image, input_size):
    # Strictly static resize with no random interpolation
    shape = image.shape[:2]
    r = min(input_size / shape[0], input_size / shape[1])
    r = min(r, 1.0)  # only downscale if needed
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:
        image = cv2.resize(image, dsize=pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image, (r, r), (w, h)

