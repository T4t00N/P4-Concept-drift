import os
import pandas as pd

# Define cache file path
cache_file = os.path.join("harborfrontV2", "cached_dataframe.pkl")

# Check if cached DataFrame exists
if os.path.exists(cache_file):
    print(f"Found cached DataFrame at {cache_file}. Loading...")
    df = pd.read_pickle(cache_file)
    print("Cached DataFrame loaded. Sample:")
    print(df.head())
else:
    print("No cached DataFrame found. Proceeding with data processing...")

import json


def load_metadata(dataset_root):
    print("Loading metadata from JSON files...")
    json_files = ["train.json", "valid.json", "test.json"]
    all_images = []
    all_annotations = []

    for json_file in json_files:
        json_path = os.path.join(dataset_root, json_file)
        if not os.path.exists(json_path):
            print(f"  - {json_file} not found, skipping...")
            continue

        print(f"  - Loading {json_file}...")
        with open(json_path, "r") as f:
            data = json.load(f)

        all_images.extend(data["images"])
        print(f"    - Found {len(data['images'])} images in {json_file}")

        people_anns = [ann for ann in data["annotations"] if ann["category_id"] == 1]
        all_annotations.extend(people_anns)
        print(f"    - Found {len(people_anns)} people annotations in {json_file}")

    print(f"Total images loaded: {len(all_images)}")
    print(f"Total people annotations loaded: {len(all_annotations)}")
    return all_images, all_annotations


# Load metadata if no cache
if not os.path.exists(cache_file):
    dataset_root = r"C:\Users\anto3\harborfrontV2"
    images_metadata, people_annotations = load_metadata(dataset_root)

from datetime import datetime

if not os.path.exists(cache_file):
    print("Matching annotations to images...")
    image_info = {}
    for img in images_metadata:
        image_id = img["id"]
        file_name = img["file_name"]
        date_str = file_name.split("/")[1]
        time_str = file_name.split("/")[2].split("_")[-1]
        try:
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
        except ValueError:
            print(f"  - Warning: Could not parse datetime from {file_name}, skipping...")
            continue
        image_info[image_id] = {
            "path": os.path.join(dataset_root, file_name),
            "datetime": dt,
            "people_bboxes": []
        }

    for ann in people_annotations:
        image_id = ann["image_id"]
        if image_id in image_info:
            image_info[image_id]["people_bboxes"].append(ann["bbox"])

    image_data = list(image_info.values())
    image_data.sort(key=lambda x: x["datetime"])
    print(f"Total images with metadata: {len(image_data)}")
    print(f"Images with people: {sum(1 for entry in image_data if entry['people_bboxes'])}")

if not os.path.exists(cache_file):
    print("Defining background region (pavement)...")
    background_roi = (100, 50, 284, 200)
    print(f"Background ROI set to: {background_roi}")

import cv2
import numpy as np


def compute_people_intensities(image_path, people_bboxes, background_roi):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"    - Warning: Failed to load image {image_path}")
        return None

    intensities = {}
    people_intensities = []
    for bbox in people_bboxes:
        x, y, w, h = map(int, bbox)
        region = img[y:y + h, x:x + w]
        if region.size > 0:
            people_intensities.append(np.mean(region))
        else:
            print(f"    - Warning: Empty region for bbox {bbox} in {image_path}")

    intensities["people"] = np.mean(people_intensities) if people_intensities else np.nan
    x, y, w, h = background_roi
    background_region = img[y:y + h, x:x + w]
    intensities["background"] = np.mean(background_region) if background_region.size > 0 else np.nan
    return intensities


if not os.path.exists(cache_file):
    print("Computing intensities for people regions...")
    data = []
    total_images = len(image_data)
    for i, entry in enumerate(image_data, 1):
        if not entry["people_bboxes"]:
            continue
        print(f"  - Processing image {i}/{total_images}: {entry['path']} ({i / total_images * 100:.1f}% complete)")
        intensities = compute_people_intensities(entry["path"], entry["people_bboxes"], background_roi)
        if intensities and not np.isnan(intensities["people"]):
            data.append({
                "datetime": entry["datetime"],
                "image_path": entry["path"],
                **intensities
            })
    print(f"Finished processing. Total images with people data: {len(data)}")

if not os.path.exists(cache_file):
    print("Converting data to DataFrame...")
    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    print("DataFrame created. Sample:")
    print(df.head())

    # Add season and time-of-day columns
    print("Adding season and time-of-day columns to DataFrame...")
    def get_season(dt):
        month = dt.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    def get_time_of_day(dt):
        hour = dt.hour
        if 0 <= hour < 6:
            return "Night"
        elif 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        else:
            return "Evening"

    df["season"] = df.index.map(get_season)
    df["time_of_day"] = df.index.map(get_time_of_day)
    print("Season and time-of-day columns added. Sample:")
    print(df[["people", "background", "season", "time_of_day"]].head())
if not os.path.exists(cache_file):
    print(f"Saving DataFrame to {cache_file}...")
    df.to_pickle(cache_file)
    print("DataFrame cached successfully.")

import matplotlib.pyplot as plt

# Plot 1: Average Intensity of People Regions Over Time
print("Generating Plot 1: Average Intensity of People Regions Over Time...")
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["people"], label="People", marker='o', markersize=3, color='blue')
plt.xlabel("DateTime")
plt.ylabel("Average Intensity")
plt.title("Average Intensity of People Regions Over Time")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Plot 1 completed.")

# Plot 2: Histogram of Average Intensity of People Regions
print("Generating Plot 2: Histogram of Average Intensity of People Regions...")
plt.figure(figsize=(8, 6))
plt.hist(df["people"], bins=30, color='blue', alpha=0.7, density=True)
plt.xlabel("Average Intensity")
plt.ylabel("Density")
plt.title("Histogram of Average Intensity of People Regions")
plt.grid(True)
plt.tight_layout()
plt.show()
print("Plot 2 completed.")

# Plot 3 (Season): Scatter Plot of People Intensity vs. Background Intensity (Colored by Season)
print("Generating Plot 3: Scatter Plot of People Intensity vs. Background Intensity (Colored by Season)...")
season_colors = {
    "Winter": "blue",
    "Spring": "green",
    "Summer": "red",
    "Fall": "orange"
}
plt.figure(figsize=(8, 6))
for season, color in season_colors.items():
    season_data = df[df["season"] == season]
    plt.scatter(season_data["background"], season_data["people"],
                label=season, color=color, alpha=0.6)
plt.xlabel("Background Intensity (Pavement)")
plt.ylabel("People Intensity")
plt.title("People Intensity vs. Background Intensity (by Season)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("Plot 3 (Season) completed.")

# Plot 3 (Time of Day): Scatter Plot of People Intensity vs. Background Intensity (Colored by Time of Day)
print("Generating Plot 3: Scatter Plot of People Intensity vs. Background Intensity (Colored by Time of Day)...")
time_of_day_colors = {
    "Night": "darkblue",
    "Morning": "lightblue",
    "Afternoon": "orange",
    "Evening": "purple"
}
plt.figure(figsize=(8, 6))
for time_of_day, color in time_of_day_colors.items():
    time_data = df[df["time_of_day"] == time_of_day]
    plt.scatter(time_data["background"], time_data["people"],
                label=time_of_day, color=color, alpha=0.6)
plt.xlabel("Background Intensity (Pavement)")
plt.ylabel("People Intensity")
plt.title("People Intensity vs. Background Intensity (by Time of Day)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("Plot 3 (Time of Day) completed.")

# Final Confirmation
print("Analysis complete! All plots have been generated.")

