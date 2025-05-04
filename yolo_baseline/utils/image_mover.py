import os
import shutil
import datetime
import sys

# Define source and destination folders
source_folder = "test"
destination_folder = "test2"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get list of files in the source folder
files = os.listdir(source_folder)
total_files = len(files)

# Process each file with progress indicator
for index, file in enumerate(files, start=1):
    # Check if the file is a .jpg image
    if file.endswith(".jpg"):
        try:
            # Extract the month (characters 4 and 5, 0-based indexing)
            month_num = file[4:6]
            # Convert month number to abbreviation (e.g., "01" to "Jan")
            month_abbr = datetime.datetime.strptime(month_num, "%m").strftime("%b")
            # Define the path for the month subfolder
            month_folder = os.path.join(destination_folder, month_abbr)
            # Create the month subfolder if it doesn't exist
            os.makedirs(month_folder, exist_ok=True)
            # Define full source and destination file paths
            source_file = os.path.join(source_folder, file)
            destination_file = os.path.join(month_folder, file)
            # Copy the file to the month subfolder
            shutil.copy(source_file, destination_file)
        except ValueError:
            # Skip files with invalid month numbers and print a warning
            print(f"\nInvalid month in file name: {file}")
            continue

    # Update and display progress indicator
    percent_complete = (index / total_files) * 100
    sys.stdout.write(f"\rProcessed {index} of {total_files} files ({percent_complete:.2f}% complete)")
    sys.stdout.flush()

print("\nProcessing complete.")
