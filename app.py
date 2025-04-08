from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

from utils.face_extraction import extract_faces_from_video
from utils.augmentation import augment_faces
from utils.embedding_generator import generate_embeddings
from utils.uploader import upload_pkl_to_cloudinary

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit for the uploaded video
ALLOWED_EXTENSIONS = {'mp4'}  # Only allow .mp4 files

app = Flask(__name__)
CORS(app)  # Enables cross-origin requests

# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to compress .pkl file if it exceeds 9MB
import gzip
def compress_pkl_file(pkl_path):
    file_size = os.path.getsize(pkl_path)
    if file_size > 9 * 1024 * 1024:  # If the file is larger than 9MB
        compressed_path = pkl_path + ".gz"
        with open(pkl_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        return compressed_path
    return pkl_path

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    user_id = request.form.get('user_id')

    if not file or not user_id:
        return jsonify({'error': 'Missing video or user_id'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type, only .mp4 files are allowed'}), 400

    if file.content_length > MAX_FILE_SIZE:
        return jsonify({'error': f'File size exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB'}), 400

    # Save the uploaded video file
    video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
    file.save(video_path)

    try:
        # Step 1: Extract faces from the video
        faces_dir = extract_faces_from_video(video_path, user_id)

        # Step 2: Augment faces
        augmented_dir = augment_faces(faces_dir)

        # Step 3: Generate embeddings and save as .pkl
        pkl_path = generate_embeddings(augmented_dir, user_id)

        # Step 4: Compress .pkl file if it exceeds 9MB
        compressed_pkl_path = compress_pkl_file(pkl_path)

        # Step 5: Upload .pkl (or compressed .pkl) to Cloudinary
        cloud_url = upload_pkl_to_cloudinary(compressed_pkl_path, user_id)

        return jsonify({'message': 'Video processed successfully', 'cloud_url': cloud_url}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
