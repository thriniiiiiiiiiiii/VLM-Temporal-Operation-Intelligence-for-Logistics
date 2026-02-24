import os
import json
import cv2
import numpy as np
from pathlib import Path

def create_mock_video(path, duration_sec=10, fps=25, size=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    
    for i in range(duration_sec * fps):
        # Generate a distinct pattern for each frame to help entropy sampling
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50 + (i % 100)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Add some "action" for entropy
        if 50 <= i <= 150: # "Tape" operation simulation
            cv2.rectangle(frame, (100, 100), (200, 200), (0, 0, 255), -1)
        out.write(frame)
    out.release()

def create_mock_annotations(path):
    # Matches the structure in data_pipeline.py _parse_annotation
    data = {
        "annotations": [
            {"operation": "Box Setup", "start_frame": 0, "end_frame": 49, "start_time": 0.0, "end_time": 1.96},
            {"operation": "Tape", "start_frame": 50, "end_frame": 150, "start_time": 2.0, "end_time": 6.0},
            {"operation": "Put Items", "start_frame": 151, "end_frame": 249, "start_time": 6.04, "end_time": 10.0}
        ]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    root = Path("mock_data")
    subjects = ["U0101", "U0102", "U0103", "U0104", "U0105", "U0106", "U0107", "U0108"]
    
    for sub in subjects:
        sub_dir = root / sub / "S0100"
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Video
        video_dir = sub_dir / "kinect" / "rgb"
        video_dir.mkdir(parents=True, exist_ok=True)
        create_mock_video(video_dir / "kinect_rgb.mp4")
        
        # Annotation
        anno_dir = sub_dir / "operation"
        anno_dir.mkdir(parents=True, exist_ok=True)
        create_mock_annotations(anno_dir / "annotations.json")
        
    print(f"Mock dataset created at {root}")

if __name__ == "__main__":
    main()
