import cloudinary.uploader

def upload_pkl_to_cloudinary(pkl_path, user_id):
    result = cloudinary.uploader.upload_large(
        pkl_path,
        resource_type="raw",
        public_id=f"users/{user_id}/embeddings"
    )
    return result.get('secure_url')
