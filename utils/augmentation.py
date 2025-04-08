import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

def augment_faces(input_dir):
    output_dir = input_dir.replace("faces", "augmented_faces")
    os.makedirs(output_dir, exist_ok=True)

    # Define augmentation pipeline
    augmenters = iaa.Sequential([
        iaa.Affine(rotate=(-15, 15)),  # Rotation (±15°)
        iaa.Affine(scale=(0.9, 1.1)),  # Zoom (±10%)
        iaa.Multiply((0.7, 1.3)),      # Brightness Adjustment
        iaa.Fliplr(0.5),               # Horizontal Flip (50% chance)
        iaa.AdditiveGaussianNoise(scale=(10, 20)),  # Gaussian Noise
        iaa.Cutout(nb_iterations=2, size=0.2)       # Cutout/Masking
    ])

    # Process images in input_dir and augment
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            augmented_images = augmenters(image=img)

            # Save original and augmented images
            for i, augmented_image in enumerate(augmented_images):
                augmented_img_path = os.path.join(output_dir, f"{filename.split('.')[0]}_aug_{i}.jpg")
                cv2.imwrite(augmented_img_path, augmented_image)

    return output_dir
