from flask import Flask, request, jsonify
import os
import uuid

from utils.face_extraction import extract_faces_from_video
from utils.augmentation import augment_faces
from utils.embedding_generator import generate_embeddings
from utils.uploader import upload_pkl_to_cloudinary

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    user_id = request.form.get('user_id')

    if not file or not user_id:
        return jsonify({'error': 'Missing video or user_id'}), 400

    video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
    file.save(video_path)

    try:
        faces_dir = extract_faces_from_video(video_path, user_id)
        augmented_dir = augment_faces(faces_dir)
        pkl_path = generate_embeddings(augmented_dir, user_id)
        cloud_url = upload_pkl_to_cloudinary(pkl_path, user_id)
        return jsonify({'cloud_url': cloud_url}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
