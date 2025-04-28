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

# Assume wh2xy, xy2wh, resize_static functions are defined as before

class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.augment = augment
        self.input_size = input_size

        # Read labels using the modified load_label method
        cache = self.load_label(filenames, save_interval=50000) # Pass interval
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = numpy.array(shapes, dtype=numpy.float64)
        self.filenames = list(cache.keys()) # filenames are keys of the final cache
        self.n = len(self.filenames) # Use length of final cache
        self.indices = range(self.n)

    def __getitem__(self, index):
        try:
            index = self.indices[index]
            filename = self.filenames[index]
            label = self.labels[index].copy()
            cached_shape = tuple(self.shapes[index])  # From cache

            # Load image
            image, shape = self.load_image_from_path(filename)
            print(f"Loading image: {filename}, cached shape: {cached_shape}, loaded shape: {shape}")  # Debug

            # Validate image and shapes
            if image is None:
                raise ValueError(f"Failed to load image: {filename}")
            if cached_shape != shape:
                raise ValueError(f"Shape mismatch for {filename}: cached {cached_shape}, loaded {shape}")

            # Resize
            result = resize_static(image, self.input_size)
            if result is None:
                raise ValueError(f"resize_static returned None for {filename}, shape: {shape}")
            image_resized, ratio, pad = result

            h_orig, w_orig = shape
            h_resized, w_resized = image_resized.shape[:2]

            # Transform labels
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], w=w_orig, h=h_orig, pad_w=0, pad_h=0)
                label[:, 1:5:2] *= ratio[0]
                label[:, 3:5:2] *= ratio[1]
                label[:, 1:5:2] += pad[0]
                label[:, 3:5:2] += pad[1]

            if len(label):
                label[:, 1:5] = xy2wh(label[:, 1:5], w=w_resized, h=h_resized)

            target = torch.zeros((len(label), 6))
            if len(label):
                target[:, 1:] = torch.from_numpy(label)

            sample = numpy.ascontiguousarray(image_resized[numpy.newaxis, :, :])
            shapes_info = (shape, (ratio, pad))

            return torch.from_numpy(sample), target, shapes_info
        except Exception as e:
            print(f"Error in __getitem__ for index {index}, file {filename}: {e}")
            raise

    # Modified load_image to accept path directly
    def load_image_from_path(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image {path}")
        h, w = image.shape[:2]
        return image, (h, w)

    # Keep original load_image if needed elsewhere, or adapt __getitem__ further
    def load_image(self, i):
        # This is likely redundant now if __getitem__ uses load_image_from_path
        return self.load_image_from_path(self.filenames[i])

    @staticmethod
    def collate_fn(batch):
        samples, targets, shapes = zip(*batch)
        for i, item in enumerate(targets):
            item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

    @staticmethod
    def load_label(filenames, save_interval=50000, person_only=False): # Add save_interval
        # Setup cache directory
        cache_dir = "/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset/filtered_cache"
        os.makedirs(cache_dir, exist_ok=True)

        # Build base name from the parent folder of the first filename
        if not filenames:
             return {} # Handle empty input list
        base_name = os.path.basename(os.path.dirname(filenames[0]))

        # Build the path for the label cache (consider person_only if implementing)
        cache_suffix = "_person_label_cache.pt" if person_only else "_label_cache.pt"
        label_cache_filename = f"{base_name}{cache_suffix}"
        label_cache_path = os.path.join(cache_dir, label_cache_filename)

        # --- Caching Logic Modification ---
        label_dict = {}
        processed_filenames = set()

        # 1. Try to load existing cache (partial or complete)
        if os.path.exists(label_cache_path):
            try:
                print(f"Attempting to load cache from {label_cache_path}...")
                label_dict = torch.load(label_cache_path)
                # Ensure loaded keys are strings (sometimes needed across versions)
                label_dict = {str(k): v for k, v in label_dict.items()}
                processed_filenames = set(label_dict.keys())
                print(f"Successfully loaded {len(processed_filenames)} items from cache.")
            except Exception as e:
                print(f"Warning: Could not load or parse cache file {label_cache_path}. Rebuilding cache. Error: {e}")
                # If loading fails, delete the corrupt cache file to start fresh
                try:
                    os.remove(label_cache_path)
                    print(f"Removed corrupt cache file: {label_cache_path}")
                except OSError as oe:
                    print(f"Error removing corrupt cache file: {oe}")
                label_dict = {}
                processed_filenames = set()

        # 2. Determine files still needing processing
        # Ensure input filenames are also strings for comparison
        filenames_str = [str(f) for f in filenames]
        files_to_process = [f for f in filenames_str if f not in processed_filenames]

        if not files_to_process:
            print("All required files found in cache.")
            # Ensure the returned dict only contains keys from the original input `filenames`
            # This handles cases where the cache might have more items than needed for the current run.
            final_dict = {f: label_dict[f] for f in filenames_str if f in label_dict}
            return final_dict

        # 3. Process remaining files
        print(f"Need to process {len(files_to_process)} more files...")
        num_initially_processed = len(processed_filenames)

        # Use enumerate starting from the number already processed for interval checks
        # but iterate over the list of files that *need* processing.
        process_iterator = tqdm(files_to_process, desc="Processing labels")
        for i, filename in enumerate(process_iterator):
            # Calculate total count for interval check
            current_total_processed = num_initially_processed + i + 1

            try:
                # --- Start of individual file processing ---
                # Verify image integrity and get shape
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify() # Verify DDS/PNG/JPG etc.

                shape = image.size # Use PIL Image size
                assert (shape[0] > 9) & (shape[1] > 9), f"Image size {shape} <10 pixels for {filename}"
                assert image.format.lower() in FORMATS, f"Invalid image format {image.format} for {filename}"

                # Construct label path
                image_folder = f"{os.sep}images{os.sep}"
                label_folder = f"{os.sep}filtered_labels{os.sep}"
                # Robust path splitting and joining
                parts = filename.split(image_folder)
                if len(parts) != 2:
                     print(f"Warning: Could not determine label path structure for {filename}. Skipping label processing.")
                     label = numpy.zeros((0, 5), dtype=numpy.float32) # Assign empty label
                else:
                    base_path_for_label = parts[0] + label_folder + parts[1]
                    base_label_path = os.path.splitext(base_path_for_label)[0]
                    label_path = f"{base_label_path}.txt"

                    if os.path.isfile(label_path):
                        try:
                            with open(label_path) as f:
                                raw_lines = f.read().strip().splitlines()
                                # Handle empty lines or lines with just whitespace
                                label = [line.split() for line in raw_lines if line.strip()]
                                if not label: # Handle empty file
                                     label = numpy.zeros((0, 5), dtype=numpy.float32)
                                else:
                                     label = numpy.array(label, dtype=numpy.float32)

                            # Validation (only if labels were loaded)
                            if label.size > 0:
                                assert label.shape[1] == 5, f"Labels require 5 columns, got {label.shape[1]} in {label_path}"
                                assert (label >= 0).all(), f"Negative label values found in {label_path}"
                                # Clamp normalized coordinates strictly between 0 and 1
                                label[:, 1:] = numpy.clip(label[:, 1:], 0.0, 1.0)
                                #assert (label[:, 1:] <= 1.0).all(), f"Non-normalized coordinates (>1) found in {label_path}" # Clip handles > 1

                                # Remove duplicate rows
                                nl = len(label)
                                _, unique_idx = numpy.unique(label, axis=0, return_index=True)
                                if len(unique_idx) < nl:
                                    label = label[unique_idx]
                        except Exception as e:
                             print(f"Error reading or parsing label file {label_path}: {e}. Assigning empty label.")
                             label = numpy.zeros((0, 5), dtype=numpy.float32)
                    else:
                        # Label file doesn't exist
                        label = numpy.zeros((0, 5), dtype=numpy.float32)

                # Add processed data to dictionary
                if filename: # Ensure filename is not empty
                    label_dict[filename] = [label, shape]
                # --- End of individual file processing ---

                # 4. Intermediate Save Logic
                if save_interval > 0 and current_total_processed % save_interval == 0:
                    print(f"\nProcessed {current_total_processed} total items. Saving intermediate cache...")
                    try:
                        torch.save(label_dict, label_cache_path)
                        print(f"Intermediate cache saved successfully to {label_cache_path}")
                    except Exception as e:
                        print(f"Error saving intermediate cache: {e}")

            except FileNotFoundError:
                # This might catch the Image.open if the image file is missing
                print(f"Warning: Image file not found: {filename}. Skipping.")
                continue # Skip to the next file
            except Exception as e:
                print(f"Error processing file {filename}: {e}. Skipping.")
                # Remove potentially corrupt entry if it was added before error
                if filename in label_dict:
                     del label_dict[filename]
                continue # Skip to the next file

        # 5. Final Save (after loop finishes)
        print(f"\nFinished processing loop. Processed {len(files_to_process)} new files.")
        print(f"Total items in dictionary: {len(label_dict)}. Saving final cache...")
        try:
            torch.save(label_dict, label_cache_path)
            print(f"Final cache saved successfully to {label_cache_path}")
        except Exception as e:
            print(f"Error saving final cache: {e}")

        # Ensure the returned dict only contains keys from the original input `filenames`
        final_dict = {f: label_dict[f] for f in filenames_str if f in label_dict}
        print(f"Returning final cache with {len(final_dict)} items.")
        return final_dict


# --- Define helper functions wh2xy, xy2wh, resize_static here ---
def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # [x, y, w, h] normalized -> [x1, y1, x2, y2] in pixels
    y = numpy.copy(x)
    # x_center = x[:, 0], y_center = x[:, 1], width = x[:, 2], height = x[:, 3]
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # x1 = w * (x_center - width/2) + pad_w
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # y1 = h * (y_center - height/2) + pad_h
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # x2 = w * (x_center + width/2) + pad_w
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # y2 = h * (y_center + height/2) + pad_h
    return y

def xy2wh(x, w=640, h=640):
    # [x1, y1, x2, y2] in pixels -> [x, y, w, h] normalized
    # Clip first to ensure valid coordinates within image bounds
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w) # Clip x coordinates to [0, w]
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h) # Clip y coordinates to [0, h]

    y = numpy.zeros_like(x) # Initialize with zeros
    # Calculate differences safely after clipping
    dw = x[:, 2] - x[:, 0]
    dh = x[:, 3] - x[:, 1]

    # Calculate centers and dimensions, normalize by image size
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x_center normalized
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y_center normalized
    y[:, 2] = dw / w  # width normalized
    y[:, 3] = dh / h  # height normalized
    return y

def resize_static(image, input_size):
    shape = image.shape[:2]  # current shape [height, width]
    h, w = shape

    # Calculate resize ratio, keeping aspect ratio
    r = min(input_size / h, input_size / w)