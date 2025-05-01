import math
import os
import random
import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data
from tqdm import tqdm

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.augment = augment
        self.input_size = input_size

        # Read labels
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
        h, w = image.shape[:2]
        return image, (h, w)

    @staticmethod
    def collate_fn(batch):
        samples, targets, shapes = zip(*batch)
        for i, item in enumerate(targets):
            item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

    @staticmethod
    def load_label(filenames, person_only=False):
        # Setup cache directory
        cache_dir = "/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset/filtered_cache"
        #print("Cache directory:", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        is_val_split = 'val' in filenames[0] or 'test' in filenames[0]
        label_cache_path = os.path.join(
            cache_dir,
            'val_label_cache.pt' if is_val_split else 'train_label_cache.pt')
        #print("Cache path:", label_cache_path)
        # Build the path for the person-only label cache
        #person_cache_filename = f"{base_name}_person_label_cache.pt"
        #person_cache_path = os.path.join(cache_dir, person_cache_filename)
        #print("Person cache path:", person_cache_path)

        # If the label cache already exists, load it

        if os.path.exists(label_cache_path):
            return torch.load(label_cache_path)
        else:
            print(f"Creating label from {label_cache_path}")

        # Build up the label dictionary
        label_dict = {}
        for filename in tqdm(filenames, desc="Processing labels"):
            try:
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()

                shape = image.size
                assert (shape[0] > 9) & (shape[1] > 9), f"Image size {shape} <10 pixels"
                assert image.format.lower() in FORMATS, f"Invalid image format {image.format}"

                # Construct label path from the image path
                image_folder = f"{os.sep}images{os.sep}"
                label_folder = f"{os.sep}filtered_labels{os.sep}"
                split_part = filename.rsplit(image_folder, 1)
                swapped_folder_path = label_folder.join(split_part)
                base_label_path = swapped_folder_path.rsplit('.', 1)[0]
                label_path = f"{base_label_path}.txt"

                if os.path.isfile(label_path):
                    with open(label_path) as f:
                        raw_lines = f.read().strip().splitlines()
                        label = [line.split() for line in raw_lines if len(line)]
                        label = numpy.array(label, dtype=numpy.float32)

                    nl = len(label)
                    if nl:
                        assert label.shape[1] == 5, "Labels require 5 columns"
                        assert (label >= 0).all(), "Negative label values"
                        assert (label[:, 1:] <= 1).all(), "Non-normalized coordinates"
                        # Remove any duplicate rows
                        _, unique_idx = numpy.unique(label, axis=0, return_index=True)
                        if len(unique_idx) < nl:
                            label = label[unique_idx]
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)

                if filename:
                    label_dict[filename] = [label, shape]

            except FileNotFoundError:
                pass

        # Save to cache file
        torch.save(label_dict, label_cache_path)
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