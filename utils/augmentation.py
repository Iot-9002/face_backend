import os
import cv2
import numpy as np
from pathlib import Path

def augment_faces(input_dir):
    output_dir = input_dir.replace("faces", "augmented_faces")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # Original
            cv2.imwrite(os.path.join(output_dir, f"{filename}"), img)

            # Flip
            flipped = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(output_dir, f"{filename}_flip.jpg"), flipped)

            # Brightness
            bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
            cv2.imwrite(os.path.join(output_dir, f"{filename}_bright.jpg"), bright)

            # Rotation
            h, w = img.shape[:2]
            matrix = cv2.getRotationMatrix2D((w/2, h/2), 15, 1)
            rotated = cv2.warpAffine(img, matrix, (w, h))
            cv2.imwrite(os.path.join(output_dir, f"{filename}_rotate.jpg"), rotated)

    return output_dir
