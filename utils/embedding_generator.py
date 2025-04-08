import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

def generate_embeddings(image_dir, user_id):
    embeddings = []
    labels = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (160, 160))
        flat = img.flatten()
        embeddings.append(flat)
        labels.append(user_id)

    data = {
        'embeddings': embeddings,
        'labels': labels
    }

    output_path = f"temp/pkl/{user_id}.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    return output_path
