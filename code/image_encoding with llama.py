import os
import numpy as np
from PIL import Image
from modelscope import AutoModel, AutoProcessor
# from transformers import CLIPModel, CLIPProcessor
import torch

model_path = "AI-ModelScope/clip-vit-large-patch14"
model = AutoModel.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image_dir = "D:/pycharm_project/LightGCN-PyTorch-master/data/cellphones/cellphones/image"
num_images = 10429
output_npy_path = "D:/pycharm_project/LightGCN-PyTorch-master/data/cellphones/cellphones/item_image_features.npy"

for i in range(num_images):
    image_path = os.path.join(image_dir, f"{i}.jpg")
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs).cpu().numpy()

        if i == 0:
            np.save(output_npy_path, image_features)
        else:
            existing_features = np.load(output_npy_path)
            new_features = np.vstack((existing_features, image_features))
            np.save(output_npy_path, new_features)

        print(f"Processed image {i}.jpg")
    except FileNotFoundError:
        image_features = torch.zeros((1, 768)).cpu().numpy()
        existing_features = np.load(output_npy_path)
        new_features = np.vstack((existing_features, image_features))
        np.save(output_npy_path, new_features)

        print(f"Image {i}.jpg not found. Skipping...")


