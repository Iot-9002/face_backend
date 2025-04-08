import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
from mtcnn import MTCNN
import os

def generate_embeddings(image_dir, user_id):
    embedder = FaceNet()
    detector = MTCNN()
    embeddings = []
    labels = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Detect faces using MTCNN
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_image)

        if not faces:
            continue

        for face in faces:
            x, y, w, h = face['box']
            face_img = rgb_image[y:y+h, x:x+w]

            # Resize to 160x160 for FaceNet
            face_img = cv2.resize(face_img, (160, 160))

            # Extract embedding
            embedding = embedder.embeddings(np.array([face_img]))[0]

            # Store embeddings and labels
            embeddings.append(embedding)
            labels.append(user_id)

    # Save embeddings to a .pkl file
    data = {
        'embeddings': embeddings,
        'labels': labels
    }
    output_path = f"temp/pkl/{user_id}.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    return output_path
