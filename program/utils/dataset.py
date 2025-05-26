import math
import os
import random
import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data
from tqdm import tqdm
from pathlib import Path

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment, cache_base_dir=""):
        self.params = params
        self.augment = augment
        self.input_size = input_size

        # --- Cache directory setup ---
        # If cache_base_dir is provided, use it. Otherwise, default to a hidden folder
        # in the current working directory, which is usually writable.
        if cache_base_dir:
            self.cache_dir = Path(cache_base_dir) / "filtered_cache"
        else:
            self.cache_dir = Path('./.yolo_cache') / "filtered_cache"

        self.cache_dir.mkdir(parents=True, exist_ok=True) # Ensure the cache directory exists

        # --- MODIFICATION START ---
        # Define your specific, fixed cache path here
        self.fixed_label_cache_path = Path("/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset/filtered_cache/train_label_cache.pt")
        # Ensure the parent directory for the fixed cache path exists
        self.fixed_label_cache_path.parent.mkdir(parents=True, exist_ok=True)
        # --- MODIFICATION END ---


        # Read labels
        # load_label now needs access to self.cache_dir
        cache = self.load_label(filenames)
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
        # Check if image was loaded successfully
        if image is None:
            raise FileNotFoundError(f"Error loading image: {self.filenames[i]}. Please check file path and integrity.")
        h, w = image.shape[:2]
        return image, (h, w)

    @staticmethod
    def collate_fn(batch):
        samples, targets, shapes = zip(*batch)
        for i, item in enumerate(targets):
            item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

    def load_label(self, filenames, person_only=False):
        # --- MODIFICATION START ---
        # Use the fixed path instead of dynamically generating it
        label_cache_path = self.fixed_label_cache_path
        # --- MODIFICATION END ---

        # If the label cache already exists, load it
        if label_cache_path.exists():
            try:
                cached_data = torch.load(label_cache_path)
                # Simple check for cache validity (number of files and content)
                # You might want to remove or adjust this validity check if your fixed cache
                # doesn't necessarily contain all 'filenames' from the current run,
                # especially if it's a pre-generated "master" cache.
                if isinstance(cached_data, dict) and len(cached_data) == len(filenames) and all(
                        f in cached_data for f in filenames):
                    print(f"Loaded label cache from {label_cache_path}")
                    return cached_data
                else:
                    print(f"Cache {label_cache_path} is invalid or incomplete. Regenerating...")
            except Exception as e:
                print(f"Error loading cache from {label_cache_path}: {e}. Regenerating...")

        print(f"Creating label cache in {label_cache_path}")

        # Build up the label dictionary
        label_dict = {}
        # Iterate over filenames using tqdm for progress bar
        for filename in tqdm(filenames, desc="Processing labels"):
            try:
                with open(filename, 'rb') as f:
                    image_pil = Image.open(f)
                    image_pil.verify()

                shape = image_pil.size
                assert (shape[0] > 9) & (shape[1] > 9), f"Image size {shape} <10 pixels: {filename}"
                assert image_pil.format.lower() in FORMATS, f"Invalid image format {image_pil.format}: {filename}"

                image_folder = f"{os.sep}images{os.sep}"
                label_folder = f"{os.sep}filtered_labels{os.sep}"

                lbl_path_str = str(filename)
                if image_folder in lbl_path_str:
                    split_part = lbl_path_str.rsplit(image_folder, 1)
                    swapped_folder_path = label_folder.join(split_part)
                    base_label_path = swapped_folder_path.rsplit('.', 1)[0]
                    label_path = f"{base_label_path}.txt"
                else:
                    label_path = os.path.splitext(filename)[0] + ".txt"

                label = numpy.zeros((0, 5), dtype=numpy.float32)
                if os.path.isfile(label_path):
                    with open(label_path, 'r') as f:
                        raw_lines = f.read().strip().splitlines()
                        parsed_labels = []
                        for line in raw_lines:
                            if len(line.strip()) > 0:
                                try:
                                    parts = line.strip().split()
                                    if len(parts) == 5:
                                        cls = int(parts[0])
                                        coords = [float(p) for p in parts[1:5]]
                                        parsed_labels.append([cls] + coords)
                                    else:
                                        print(
                                            f"Warning: Malformed label line in {label_path}: '{line.strip()}'. Skipping.")
                                except ValueError:
                                    print(
                                        f"Warning: Could not parse numerical values in {label_path}: '{line.strip()}'. Skipping.")

                        if parsed_labels:
                            label = numpy.array(parsed_labels, dtype=numpy.float32)

                    nl = len(label)
                    if nl:
                        assert label.shape[1] == 5, f"Labels in {label_path} require 5 columns (class, x, y, w, h)"
                        assert (label >= 0).all(), f"Negative label values found in {label_path}"
                        assert (label[:, 1:] <= 1).all(), f"Non-normalized coordinates found in {label_path}"
                        _, unique_idx = numpy.unique(label, axis=0, return_index=True)
                        if len(unique_idx) < nl:
                            label = label[unique_idx]

                label_dict[filename] = [label, shape]

            except FileNotFoundError as e:
                print(
                    f"Warning: Image file not found for label processing: {filename}. Error: {e}. Skipping this entry.")
                continue
            except Exception as e:
                print(f"Error processing {filename}: {e}. Skipping this entry.")
                continue

        # Save to cache file
        try:
            torch.save(label_dict, label_cache_path)
            print(f"Saved label cache to {label_cache_path}")
        except Exception as e:
            print(f"Error saving label cache to {label_cache_path}: {e}")
            print(f"Please ensure write permissions for: {label_cache_path.parent}")

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

    # Calculate new unpadded shape
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Calculate padding for letterbox
    dw, dh = (input_size - new_unpad[0]) / 2, (input_size - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:  # if not already the target unpadded shape
        image = cv2.resize(image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Apply padding
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image, (r, r), (left, top) # Return image, ratio_tuple, pad_tuple