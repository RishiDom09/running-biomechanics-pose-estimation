#mediapipe landmark extraction 


import cv2 
import os 
import csv 
import numpy as np
import mediapipe as mp 


#input folder and output csv folder

video_folder  = '/Users/rishi/Desktop/hs_women'     # input videos
output_folder = '/Users/rishi/hs_women'             # output CSVs


os.makedirs(output_folder, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)

LANDMARKS_TO_KEEP = {
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_foot_index", "right_foot_index"
}

for filename in os.listdir(video_folder):
    if filename.lower().endswith((".mp4", ".mov")):
        video_path = os.path.join(video_folder, filename)
        print(f"Processing {video_path}")

        csv_path = os.path.join(
            output_folder, 
            f"{os.path.splitext(filename)[0]}.csv"
        )

        with open(csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            header = ['frame', 'downward_velocity']
            for lm in mp_pose.PoseLandmark:
                name = lm.name.lower()
                if name in LANDMARKS_TO_KEEP:
                    header += [
                        f"{name}_x",
                        f"{name}_y",
                        f"{name}_z",
                        f"{name}_visibility"
                    ] 
                    csv_writer.writerow(header)

                    cap = cv2.VideoCapture(video_path)
                    frame_idx = 0 
                    prev_ankle_y = None 

                    while cap.isOpened():
                        success, frame = cap.read()
                        if not success:
                            break 
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(frame_rgb)

                            if not results.pose_landmarks:
                                frame_idx += 1
                                continue

                            landmarks = results.pose_landmarks.landmark
                            ankle = landmarks[mp_pose.PoseLamndmark.LEFT_ANKLE.value]
                            ankle_y = ankle.y

                            if prev_ankle_y is None:
                                downward_velocity = 0.0
                            else:
                                downward_velocity = ankle_y - prev_ankle_y
                            
                            prev_ankle_y = ankle_y
                            row = [frame_idx, downward_velocity]

                            for i, lm in enumerate(landmarks):
                                name = mp_pose.Poselandmark(i).name.lower()
                                if name in LANDMARKS_TO_KEEP:
                                    row += [lm.x, lm.y, lm.z, lm.visibility]

                            csv_writer.writerow(row)
                        except Exception as e:
                            print(f"Error processing frame {frame_idx} of {filename}: {e}")

                        frame_idx += 1

                    cap.release()
print("Done")

