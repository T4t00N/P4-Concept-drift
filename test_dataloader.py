from torch.utils.data import DataLoader
from torchvision import transforms
from YOLOv8_dataloader import YOLOv8COCODataset

# Define transformations ensuring single-channel images remain valid
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLO standard size
    transforms.ToTensor()  # Converts image to tensor (keeps 1-channel format)
])

dataset = YOLOv8COCODataset(
    json_file=r"harborfront/Test.json",
    root_dir=r"harborfront/",
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

# print(f"simple dataloader print: {}")

# Test dataloader
for data in dataloader:
    images, targets, img_path = data[0]
    print("Batch size:", len(images))
    print("First image shape:", images.shape)  # Should be (1, 640, 640)
    print("First bbox:", targets)
    print("First image path:", img_path)
    break
