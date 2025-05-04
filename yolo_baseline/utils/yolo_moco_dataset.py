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
    def __init__(self, filenames, input_size, params, augment, cluster_label):
        self.params = params
        self.augment = augment # Although passed, it's not used in this version
        self.input_size = input_size
        self.cluster_label = cluster_label

        # Read labels - Call the static method correctly using the Class name
        cache = Dataset.load_label(filenames, cluster_label=self.cluster_label)
        # Ensure cache is not empty before trying to unpack
        if not cache:
             raise ValueError(f"Label cache for cluster '{cluster_label}' is empty or failed to load.")
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
        if image is None: # Handle case where image loading fails
             print(f"Warning: Failed to load image at index {index}, filename: {self.filenames[index]}. Skipping.")
             # Return dummy data or raise an error, depending on desired behavior
             # Returning dummy data might hide issues
             # Example: Return None to be filtered in collate_fn or handled by the training loop
             # Or return tensors of zeros:
             dummy_sample = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
             dummy_target = torch.zeros((0, 6), dtype=torch.float32)
             dummy_shapes = ((0, 0), ((0, 0), (0, 0)))
             return dummy_sample, dummy_target, dummy_shapes


        h, w = image.shape[:2]

        # Resize
        image, ratio, pad = resize_static(image, self.input_size)
        shapes = shape, ((h / shape[0] if shape[0] else 0, w / shape[1] if shape[1] else 0), pad) # Add checks for zero division

        # Get labels and transform them from normalized [x,y,w,h] to pixel [x1,y1,x2,y2]
        label = self.labels[index].copy()
        if label.size and w > 0 and h > 0: # Add checks for valid w, h
            label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])

        # Convert [x1,y1,x2,y2] back to normalized [x,y,w,h]
        nl = len(label)
        if nl and image.shape[1] > 0 and image.shape[0] > 0: # Add checks for valid image dims
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
        # Add error handling for image reading
        try:
            image = cv2.imread(self.filenames[i], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"cv2.imread failed to load image: {self.filenames[i]}")
            h, w = image.shape[:2]
            return image, (h, w)
        except Exception as e:
            print(f"Error loading image {self.filenames[i]}: {e}")
            return None, (0, 0) # Return None image and zero shape on error

    @staticmethod
    def load_label(filenames, cluster_label, person_only=False):
        cluster_label_str = str(cluster_label).strip()  # ← removes stray spaces/new‑lines
        is_original_val = cluster_label_str == "original_val"  # keep as boolean so we can re‑use
        # --------------------------------------------------------------------

        # keep using the _normalised_ string from here on
        cluster_label = cluster_label_str
        # --- This static method is now correctly placed within the outer Dataset class ---
        cache_dir = "filtered_cache"
        os.makedirs(cache_dir, exist_ok=True)

        if is_original_val:
            # pre‑built cache lives here ⬇
            label_cache_path = (
                "/ceph/project/P4-concept-drift/final_yolo_data_format/"
                "YOLOv8-pt/Dataset/filtered_cache/val_label_cache.pt"
            )
            cache_dir = os.path.dirname(label_cache_path)
            os.makedirs(cache_dir, exist_ok=True)

        base_name = "unknown_set"
        if filenames:
            try:
                first_dir = os.path.dirname(filenames[0])
                if first_dir: # Check if dirname is not empty
                    base_name = os.path.basename(first_dir)
                    if not base_name:
                        base_name = "root_set" # Handle files directly in root
                else: # Handle case where filename might not have a directory part
                    base_name = "no_dir_set"
            except IndexError:
                base_name = "error_set"
        else:
            base_name = "empty_set"

        safe_cluster_label = str(cluster_label).replace('/', '_').replace('\\', '_')
        label_cache_filename = f"{base_name}_cluster_{safe_cluster_label}_label_cache.pt"
        label_cache_path = os.path.join(cache_dir, label_cache_filename)

        if os.path.exists(label_cache_path):
            try:
                print(f"Loading label cache for cluster '{cluster_label}' from {label_cache_path}")
                return torch.load(label_cache_path)
            except Exception as e:
                 print(f"Warning: Failed to load cache file {label_cache_path}: {e}. Recreating cache.")
                 # Optionally delete the corrupted cache file: os.remove(label_cache_path)


        print(f"Creating label cache for cluster '{cluster_label}' at {label_cache_path}")
        label_dict = {}
        pbar = tqdm(filenames, desc=f"Processing labels for cluster {cluster_label}")
        for filename in pbar:
            try:
                label_path = filename.replace('images', 'labels').replace(os.path.splitext(filename)[1], '.txt')

                if os.path.exists(label_path):
                    with open(label_path) as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        # Add check for empty lines or lines with wrong number of elements
                        l_valid = []
                        for item in l:
                            if len(item) == 5: # Expecting class + 4 coordinates
                                try:
                                     # Validate numeric conversion
                                     vals = [float(v) for v in item]
                                     l_valid.append(vals)
                                except ValueError:
                                     print(f"Warning: Non-numeric value found in label file {label_path}, line: '{' '.join(item)}'. Skipping line.")
                                     continue
                            else:
                                print(f"Warning: Invalid line format in label file {label_path}: '{' '.join(item)}'. Skipping line.")
                                continue

                        if person_only:
                             l_valid = [x for x in l_valid if int(x[0]) == 0]

                        if l_valid: # Only create array if valid lines were found
                            label = numpy.array(l_valid, dtype=numpy.float32)
                        else:
                            label = numpy.empty((0, 5), dtype=numpy.float32)

                else:
                    label = numpy.empty((0, 5), dtype=numpy.float32)

                shape = (0, 0) # Default shape
                try:
                    # Using PIL to get shape without loading full image data
                    with Image.open(filename) as img:
                        shape = img.size[::-1]  # Get (height, width)
                except FileNotFoundError:
                    print(f"Warning: Image file not found: {filename}. Skipping.")
                    continue # Skip this file entirely if image doesn't exist
                except Exception as e:
                    print(f"Warning: Could not read image shape for {filename}: {e}. Storing with 0 shape.")
                    # Keep label but with zero shape - might need filtering later

                # Store label and shape if shape is valid (or handle invalid shape cases)
                # You might decide to skip files with invalid shapes entirely
                if shape[0] > 0 and shape[1] > 0:
                    label_dict[filename] = [label, shape]
                else:
                    print(f"Skipping file {filename} due to invalid shape {shape} or image loading error.")
                    # Alternatively, store with zero shape if needed:
                    # label_dict[filename] = [label, shape]

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                # Continue to the next file

        if not label_dict:
            print(f"Warning: No valid labels/images processed for cluster {cluster_label}. Cache file will not be saved.")
            return {}

        try:
            print(f"Saving label cache with {len(label_dict)} entries to {label_cache_path}")
            torch.save(label_dict, label_cache_path)
        except Exception as e:
             print(f"Error saving cache file {label_cache_path}: {e}")
             # Consider returning the dictionary anyway, even if saving failed
        return label_dict


    @staticmethod
    def collate_fn(batch):
        # Filter out None samples potentially returned by __getitem__ on error
        batch = [b for b in batch if b is not None and b[0] is not None]
        if not batch: # If all samples in batch failed
             # Return empty tensors or raise error
             return torch.empty(0), torch.empty(0), []

        samples, targets, shapes = zip(*batch)
        for i, item in enumerate(targets):
            item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes


    # --- The nested Dataset class definition should be REMOVED from here ---
    # @staticmethod
    # class Dataset(data.Dataset):
    #     ... (rest of the nested class code was here) ...


# --- Utility functions remain unchanged ---
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
    # Ensure w and h are positive to avoid division by zero or negative strides
    if w <= 0 or h <= 0:
        print(f"Warning: xy2wh called with non-positive dimensions w={w}, h={h}. Returning original input.")
        return x # Or handle as error

    x_copy = numpy.copy(x) # Work on a copy
    # Clip coordinates to be within image bounds [0, w-1] and [0, h-1]
    x_copy[:, [0, 2]] = numpy.clip(x_copy[:, [0, 2]], 0, w - 1E-3)
    x_copy[:, [1, 3]] = numpy.clip(x_copy[:, [1, 3]], 0, h - 1E-3)

    y = numpy.zeros_like(x_copy) # Initialize output array
    y[:, 0] = ((x_copy[:, 0] + x_copy[:, 2]) / 2) / w  # Normalized center x
    y[:, 1] = ((x_copy[:, 1] + x_copy[:, 3]) / 2) / h  # Normalized center y
    y[:, 2] = (x_copy[:, 2] - x_copy[:, 0]) / w      # Normalized width
    y[:, 3] = (x_copy[:, 3] - x_copy[:, 1]) / h      # Normalized height
    return y



def resize_static(image, input_size):
    shape = image.shape[:2] # (height, width)
    if shape[0] == 0 or shape[1] == 0: # Check for invalid shape
        print(f"Warning: resize_static called with invalid shape {shape}. Returning dummy image.")
        # Return a black image of the target size
        dummy_image = numpy.zeros((input_size, input_size), dtype=image.dtype)
        return dummy_image, (0, 0), (input_size/2, input_size/2) # Or handle differently


    # Calculate resize ratio, ensuring it doesn't exceed 1.0 (no upscaling)
    r = min(input_size / shape[0], input_size / shape[1])
    r = min(r, 1.0)

    # New dimensions after resizing
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # (width, height)

    # Calculate padding
    dw = (input_size - new_unpad[0]) / 2
    dh = (input_size - new_unpad[1]) / 2

    # Resize if necessary
    if shape[::-1] != new_unpad: # If new size is different from original
        try:
            image = cv2.resize(image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
             print(f"Error during cv2.resize: {e}. Original shape: {shape}, target size: {new_unpad}")
             # Handle error: return a dummy image or raise
             dummy_image = numpy.zeros((input_size, input_size), dtype=image.dtype)
             return dummy_image, (0, 0), (input_size/2, input_size/2)


    # Calculate border padding amounts
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Add padding
    try:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) # Use value=0 for black border
    except cv2.error as e:
        print(f"Error during cv2.copyMakeBorder: {e}. Image shape: {image.shape}, padding: ({top},{bottom},{left},{right})")
        # Handle error: return a dummy image or raise
        dummy_image = numpy.zeros((input_size, input_size), dtype=image.dtype)
        return dummy_image, (r, r), (dw, dh)


    return image, (r, r), (dw, dh) # Return original ratio 'r' and padding 'dw', 'dh'