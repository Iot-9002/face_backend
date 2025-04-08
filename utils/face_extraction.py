import cv2
import os
import mediapipe as mp

def extract_faces_from_video(video_path, user_id):
    output_dir = f"temp/faces/{user_id}"
    os.makedirs(output_dir, exist_ok=True)

    mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                face = frame[y:y+h_box, x:x+w_box]
                if face.size > 0:
                    face_path = os.path.join(output_dir, f"face_{count}.jpg")
                    cv2.imwrite(face_path, face)
                    count += 1
    cap.release()
    return output_dir
