import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# Set the base directory
BASE_DIR = r"C:\Users\anto3\harborfrontV2"

# Paths to JSON files
TRAIN_JSON = os.path.join(BASE_DIR, "train.json")
VAL_JSON = os.path.join(BASE_DIR, "valid.json")
TEST_JSON = os.path.join(BASE_DIR, "test.json")

# Create output directory for plots
PLOTS_DIR = os.path.join(BASE_DIR, "people_count_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Load the JSON files
def load_json_data(file_paths):
    all_data = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
                    print(f"Successfully loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    return all_data


# Extract image data with metadata and annotations
def extract_data(json_data):
    df_rows = []

    for data in json_data:
        # Create a mapping from image_id to annotations
        image_to_annotations = defaultdict(list)
        for annotation in data.get('annotations', []):
            image_to_annotations[annotation['image_id']].append(annotation)

        for img in data.get('images', []):
            # Basic image info
            row = {
                'id': img['id'],
                'file_name': img['file_name'],
                'date_captured': img['date_captured'],
            }

            # Extract date components
            dt = datetime.strptime(img['date_captured'], '%Y-%m-%dT%H:%M:%S')
            row['date'] = dt.date()
            row['hour'] = dt.hour
            row['weekday'] = dt.weekday()
            row['is_weekend'] = dt.weekday() >= 5  # 5=Saturday, 6=Sunday

            # Count person annotations
            annotations = image_to_annotations.get(img['id'], [])
            row['person_count'] = sum(1 for ann in annotations if ann['category_id'] == 1)

            df_rows.append(row)

    return pd.DataFrame(df_rows)


# Plot people count by hour of day, comparing weekdays and weekends
def plot_people_count_weekday_vs_weekend(df):
    """Create a plot comparing people count by hour for weekdays vs weekends"""
    fig = plt.figure(figsize=(14, 8))

    # Group by hour and is_weekend, calculate mean person counts
    hourly_comparison = df.groupby(['hour', 'is_weekend']).agg({
        'person_count': 'mean'
    }).reset_index()

    # Separate weekday and weekend data
    weekday_data = hourly_comparison[hourly_comparison['is_weekend'] == False]
    weekend_data = hourly_comparison[hourly_comparison['is_weekend'] == True]

    # Plot
    plt.plot(weekday_data['hour'], weekday_data['person_count'], 'b-',
             label='Weekdays (Mon-Fri)', linewidth=2.5, marker='o', markersize=8)
    plt.plot(weekend_data['hour'], weekend_data['person_count'], 'r-',
             label='Weekends (Sat-Sun)', linewidth=2.5, marker='s', markersize=8)

    plt.title('People Detection by Hour: Weekdays vs Weekends', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Average Number of People Detected', fontsize=14)
    plt.xticks(range(0, 24, 1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)

    # Find peak hours safely
    if not weekday_data.empty:
        weekday_max_hour = weekday_data.loc[weekday_data['person_count'].idxmax(), 'hour']
        weekday_max_count = weekday_data['person_count'].max()

        plt.annotate(f'Weekday peak: {weekday_max_count:.1f}',
                     xy=(weekday_max_hour, weekday_max_count),
                     xytext=(5, 10), textcoords='offset points',
                     fontsize=12, fontweight='bold', color='blue')

    if not weekend_data.empty:
        weekend_max_hour = weekend_data.loc[weekend_data['person_count'].idxmax(), 'hour']
        weekend_max_count = weekend_data['person_count'].max()

        plt.annotate(f'Weekend peak: {weekend_max_count:.1f}',
                     xy=(weekend_max_hour, weekend_max_count),
                     xytext=(5, 10), textcoords='offset points',
                     fontsize=12, fontweight='bold', color='red')

    return fig  # Return the figure object instead of plt


# Main function
def main():
    # Define paths to your JSON files
    json_paths = [TRAIN_JSON, VAL_JSON, TEST_JSON]

    # Load data
    print("Loading JSON data...")
    json_data = load_json_data(json_paths)

    if not json_data:
        print("No data was loaded. Please check the file paths.")
        return

    # Extract and process data
    print("Processing data...")
    df = extract_data(json_data)
    print(f"Processed {len(df)} images")

    if len(df) == 0:
        print("No data was processed. Please check the JSON structure.")
        return

    # Generate and save the plot
    print("Generating people count comparison plot...")

    try:
        fig = plot_people_count_weekday_vs_weekend(df)
        fig.savefig(os.path.join(PLOTS_DIR, 'people_count_weekday_vs_weekend.png'),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure object, not the plt module
        print("Plot completed: People count comparison by hour (weekday vs weekend)")
        print(f"Plot saved to '{PLOTS_DIR}'")

    except Exception as e:
        print(f"Error generating plot: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()