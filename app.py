import os
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS  # 👈 NEW

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # 👈 NEW: This allows requests from other domains (like your frontend)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        file = request.files.get('video')
        user_id = request.form.get('user_id')

        print("📥 Upload request received.")
        print("🔑 User ID:", user_id)
        print("📁 File:", file.filename if file else "No file")

        if not file or not user_id:
            print("❌ Missing video or user_id.")
            return jsonify({'error': 'Missing video or user_id'}), 400

        # Save video
        video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
        file.save(video_path)
        print(f"✅ Video saved to: {video_path}")

        # Import heavy modules inside the route
        from utils.face_extraction import extract_faces_from_video
        from utils.augmentation import augment_faces
        from utils.embedding_generator import generate_embeddings
        from utils.uploader import upload_pkl_to_cloudinary

        # Step 1: Extract faces
        print("🧠 Extracting faces...")
        faces_dir = extract_faces_from_video(video_path, user_id)
        print(f"✅ Faces extracted to: {faces_dir}")

        # Step 2: Augment faces
        print("🎨 Augmenting faces...")
        augmented_dir = augment_faces(faces_dir)
        print(f"✅ Faces augmented to: {augmented_dir}")

        # Step 3: Generate embeddings
        print("🔍 Generating embeddings...")
        pkl_path = generate_embeddings(augmented_dir, user_id)
        print(f"✅ Embeddings saved to: {pkl_path}")

        # Step 4: Upload PKL to Cloudinary
        print("☁️ Uploading to Cloudinary...")
        cloud_url = upload_pkl_to_cloudinary(pkl_path, user_id)
        print(f"✅ Uploaded to Cloudinary: {cloud_url}")

        return jsonify({'cloud_url': cloud_url}), 200

    except Exception as e:
        print("❌ An error occurred:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
