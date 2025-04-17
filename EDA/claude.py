import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from datetime import datetime
import cv2
from scipy import ndimage


class InfraredImageAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.json_paths = [
            os.path.join(base_path, 'train.json'),
            os.path.join(base_path, 'valid.json'),
            os.path.join(base_path, 'test.json')
        ]

        # Load and process metadata
        self.metadata = self.load_and_process_metadata()

        # Prepare DataFrame for analysis
        self.prepare_dataframe()

    def load_and_process_metadata(self):
        """
        Load metadata from COCO-like JSON files
        """
        processed_metadata = []

        for json_path in self.json_paths:
            if not os.path.exists(json_path):
                continue

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Extract images from COCO-like structure
                images = data.get('images', [])
                for image in images:
                    # Extract filename and datetime
                    filename = image.get('file_name', '')
                    datetime_str = image.get('date_captured', '')

                    # Add additional metadata
                    metadata_entry = {
                        'filename': filename,
                        'datetime': datetime_str,
                        'width': image.get('width', 0),
                        'height': image.get('height', 0)
                    }

                    # Add weather metadata if available
                    if 'meta' in image:
                        metadata_entry.update(image['meta'])

                    processed_metadata.append(metadata_entry)

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

        return processed_metadata

    def prepare_dataframe(self):
        # Create DataFrame from processed metadata
        self.df = pd.DataFrame(self.metadata)

        # Convert datetime
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])

        # Sort by datetime
        self.df = self.df.sort_values('datetime')

    def create_region_masks(self, image_path):
        """
        Automatically create region masks for different areas in the image

        :param image_path: Path to the infrared image
        :return: Dictionary of region masks
        """
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Could not read image: {image_path}")
            return {}

        # Basic image processing
        # Use adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(cleaned)

        # Create masks for different regions
        masks = {}

        # Water (typically lower intensity regions)
        water_mask = (img < np.percentile(img, 20)).astype(np.uint8)
        masks['water'] = water_mask == 1

        # Pavement (middle intensity regions)
        pavement_mask = np.zeros_like(img, dtype=bool)
        mid_low = np.percentile(img, 40)
        mid_high = np.percentile(img, 60)
        pavement_mask[(img >= mid_low) & (img <= mid_high)] = True
        masks['pavement'] = pavement_mask

        # Pole (high intensity, narrow regions)
        pole_mask = np.zeros_like(img, dtype=bool)
        pole_candidates = labels[img > np.percentile(img, 80)]
        if len(pole_candidates) > 0:
            # Find narrow, high-intensity regions
            unique, counts = np.unique(pole_candidates, return_counts=True)
            pole_label = unique[np.argmin(counts)]
            pole_mask = (labels == pole_label) & (img > np.percentile(img, 80))
        masks['pole'] = pole_mask

        # Objects (medium-high intensity, distinct from background)
        objects_mask = np.zeros_like(img, dtype=bool)
        for label in range(1, num_labels):
            region = (labels == label)
            # Filter objects based on size and intensity
            if (np.sum(region) > 50 and  # Minimum size
                    np.sum(region) < 500 and  # Maximum size
                    np.mean(img[region]) > np.percentile(img, 60)):
                objects_mask |= region
        masks['objects'] = objects_mask

        # Background (lowest intensity regions)
        background_mask = (img < np.percentile(img, 10)).astype(np.uint8)
        masks['background'] = background_mask == 1

        return masks

    def plot_intensity_over_time(self, plot_filename=None):
        """
        Plot average intensity for different regions over time
        """
        plt.figure(figsize=(15, 6))

        # Compute intensities for each image
        intensities = []
        for _, row in self.df.iterrows():
            # Construct full image path
            image_path = os.path.join(self.base_path, row['filename'])

            # Create masks for this image
            masks = self.create_region_masks(image_path)

            # Compute intensities
            if not masks:
                continue

            image_intensities = {}
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for region, mask in masks.items():
                image_intensities[region] = np.mean(img[mask])

            image_intensities['datetime'] = row['datetime']
            intensities.append(image_intensities)

        # Convert to DataFrame
        intensity_df = pd.DataFrame(intensities)

        # Plot each region
        regions = ['water', 'pavement', 'pole', 'objects', 'background']
        for region in regions:
            plt.plot(intensity_df['datetime'], intensity_df[region],
                     label=region, marker='o', linestyle='-', alpha=0.7)

        plt.title('Average Pixel Intensity by Region Over Time')
        plt.xlabel('Date and Time')
        plt.ylabel('Average Pixel Intensity')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save or show plot
        if plot_filename:
            plt.savefig(plot_filename)
        else:
            plt.show()

    def plot_intensity_histogram(self, plot_filename=None):
        """
        Plot histogram of intensities for different regions
        """
        plt.figure(figsize=(15, 6))

        # Compute intensities for each image
        intensities = []
        for _, row in self.df.iterrows():
            # Construct full image path
            image_path = os.path.join(self.base_path, row['filename'])

            # Create masks for this image
            masks = self.create_region_masks(image_path)

            # Compute intensities
            if not masks:
                continue

            image_intensities = {}
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for region, mask in masks.items():
                image_intensities[region] = np.mean(img[mask])

            intensities.append(image_intensities)

        # Convert to DataFrame
        intensity_df = pd.DataFrame(intensities)

        # Plot histograms
        regions = ['water', 'pavement', 'pole', 'objects', 'background']
        for i, region in enumerate(regions, 1):
            plt.subplot(1, len(regions), i)
            sns.histplot(intensity_df[region], kde=True)
            plt.title(f'Intensity Distribution - {region}')
            plt.xlabel('Average Pixel Intensity')
            plt.ylabel('Frequency')

        plt.tight_layout()

        # Save or show plot
        if plot_filename:
            plt.savefig(plot_filename)
        else:
            plt.show()

    def plot_region_vs_background(self, plot_filename=None):
        """
        Scatter plot of region intensities vs background intensity
        """
        plt.figure(figsize=(15, 6))

        # Compute intensities for each image
        intensities = []
        for _, row in self.df.iterrows():
            # Construct full image path
            image_path = os.path.join(self.base_path, row['filename'])

            # Create masks for this image
            masks = self.create_region_masks(image_path)

            # Compute intensities
            if not masks:
                continue

            image_intensities = {}
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for region, mask in masks.items():
                image_intensities[region] = np.mean(img[mask])

            intensities.append(image_intensities)

        # Convert to DataFrame
        intensity_df = pd.DataFrame(intensities)

        # Scatter plot for each region against background
        regions = ['water', 'pavement', 'pole', 'objects']
        colors = ['blue', 'gray', 'yellow', 'green']
        for region, color in zip(regions, colors):
            plt.scatter(intensity_df['background'], intensity_df[region],
                        label=region, alpha=0.7, color=color)

        plt.title('Region Intensity vs Background Intensity')
        plt.xlabel('Background Pixel Intensity')
        plt.ylabel('Region Pixel Intensity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save or show plot
        if plot_filename:
            plt.savefig(plot_filename)
        else:
            plt.show()

    def generate_weather_summary(self):
        """
        Generate summary of weather conditions
        """
        # Weather columns to summarize
        weather_columns = [
            'Temperature', 'Humidity', 'Precipitation latest 10 min',
            'Dew Point', 'Wind Direction', 'Wind Speed',
            'Sun Radiation Intensity', 'Min of sunshine latest 10 min'
        ]

        # Create summary
        summary = {}
        for col in weather_columns:
            if col in self.df.columns:
                summary[col] = {
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'std': self.df[col].std()
                }

        return summary


# Example usage
if __name__ == "__main__":
    # Replace with your actual base path to the project directory
    base_path = r'C:\Users\anto3\harborfrontV2'

    # Initialize analyzer
    analyzer = InfraredImageAnalyzer(base_path)

    # Generate plots
    analyzer.plot_intensity_over_time()
    analyzer.plot_intensity_histogram()
    analyzer.plot_region_vs_background()

    # Generate weather summary
    weather_summary = analyzer.generate_weather_summary()
    print("Weather Summary:")
    for key, value in weather_summary.items():
        print(f"{key}:")
        for stat, val in value.items():
            print(f"  {stat}: {val}")