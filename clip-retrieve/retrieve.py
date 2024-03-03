import torch
import clip
import json
from PIL import Image
from karpathy_splits import COCOKSDataset
from tqdm import tqdm

# Function to compute the cosine similarity
def cosine_similarity(x1, x2):
    return torch.mm(x1, x2.T) / (torch.norm(x1, dim=1, keepdim=True) * torch.norm(x2, dim=1, keepdim=True).T)

# Function to encode images using the CLIP model
def encode_images(image_paths, model, preprocess, device):
    features = []
    for image_path in tqdm(image_paths):
        with Image.open(image_path) as img:
            image = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        features.append(image_features)
    return torch.cat(features, dim=0)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

ks = COCOKSDataset()
train_imgs = ks.get_train_data()
train_features = encode_images(train_imgs.values(), model, preprocess, device)

test_imgs = ks.get_test_data()
test_features = encode_images(test_imgs.values(), model, preprocess, device)

most_similar_set = {}

# search for the most similar 32 samples
for test_id, test_feature in tqdm(zip(test_imgs.keys(), test_features)):
    similarities = cosine_similarity(test_feature.unsqueeze(0), train_features)
    top_similar_indices = torch.topk(similarities, 32, dim=1, largest=True, sorted=True).indices.squeeze(0)
    most_similar_sample_ids = [list(train_imgs.keys())[i] for i in top_similar_indices]
    most_similar_set[test_id] = most_similar_sample_ids

# Save to JSON file
with open("train_set_clip.json", "w") as f:
    json.dump(most_similar_set, f, indent=4)
