import pandas as pd
from pathlib import Path

def images_in_cluster(csv_path: str, label: int):
    """Return a list of image paths belonging to one cluster."""
    df = pd.read_csv(csv_path)
    return df[df.cluster_label == label].image_path.tolist()

def write_list(img_paths, outfile: str):
    """Save one path per line—YOLO style."""
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        for p in img_paths:
            f.write(str(p) + "\n")