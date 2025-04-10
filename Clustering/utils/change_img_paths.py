import csv

# Define the old and new path patterns
old_path = "/ceph/project/P4-concept-drift/YOLOv8-/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset/images/test/"
new_path = "/ceph/project/P4-concept-drift/final_yolo_data_format/YOLOv8-pt/Dataset/images/test/"

# Input and output file names
input_file = "/Users/jens-jakobskotingerslev/Downloads/super_clusters2.csv"  # Replace with your input CSV file name
output_file = "/Users/jens-jakobskotingerslev/Downloads/super_clusters3.csv"  # Replace with your desired output CSV file name

# Read the input CSV and write to new CSV with modified paths
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if len(row) >= 2:  # Ensure row has at least 2 columns (path and cluster)
            # Replace the old path pattern with new path pattern
            new_path_full = row[0].replace(old_path, new_path)
            # Write the modified row (new path and cluster)
            writer.writerow([new_path_full, row[1]])

print(f"CSV file has been processed. Output saved to {output_file}")