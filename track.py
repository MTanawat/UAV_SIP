from collections import defaultdict
import cv2
import numpy as np
from ultralytics import RTDETR
import os                             # <--- MODIFIED (For creating directories)
from pathlib import Path           # <--- MODIFIED (For handling file paths)

# --- Configuration ---
MODEL_PATH = "/project/lt200246-mmacma/nuke/swamp/UAV-DETR/train1280_pad/exp6/weights/best.pt"
INPUT_VIDEO_DIR = "/project/lt200246-mmacma/nuke/swamp/DRENet_hyperconfig/test_video/" # <--- MODIFIED
OUTPUT_DIR = "/project/lt200246-mmacma/nuke/swamp/UAV-DETR/my_video_results1280_pad"  # <--- MODIFIED (This is your new directory)

# Config for Goal 2:
CONF_THRES = 0.05  # <--- MODIFIED
IOU_THRES = 0.5   # <--- MODIFIED
# ---------------------

# Load your custom YOLOv12 model
model = RTDETR(MODEL_PATH)  # <--- MODIFIED: Load with RTDETR

# Create the output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True) # <--- MODIFIED

# Find all video files in the input directory
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv'] # <--- MODIFIED
video_files = []
for ext in video_extensions:
    video_files.extend(list(Path(INPUT_VIDEO_DIR).glob(ext)))

print(f"Found {len(video_files)} videos to process. Saving results to: {OUTPUT_DIR}")

# <--- MODIFIED (Start loop over all found videos)
for video_path in video_files:
    video_path_str = str(video_path)
    output_video_name = video_path.name  # e.g., "camera1.mp4"
    output_video_path = str(Path(OUTPUT_DIR) / output_video_name)
    
    print(f"\nProcessing: {video_path.name}")

    # Open the video file
    cap = cv2.VideoCapture(video_path_str) # <--- MODIFIED (Use loop variable)

    # --- Setup Video Writer ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height)) # <--- MODIFIED (Use new output path)
    # -------------------------

    # Store the track history (reset for each new video)
    track_history = defaultdict(lambda: []) # <--- MODIFIED (Moved inside loop)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO tracking on the frame, persisting tracks between frames
            result = model.track(
                frame, 
                persist=True, 
                conf=CONF_THRES, # <--- MODIFIED (Pass config)
                iou=IOU_THRES,    # <--- MODIFIED (Pass config)
                tracker="bytetrack_config.yaml", # <--- ADD THIS
                verbose=False
            )[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the bounding boxes and track IDs on the frame
                frame = result.plot()

                # Plot the tracks (the "tail" behind the object)
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # --- Save the frame ---
            out.write(frame)
            # ----------------------

            # We remove the cv2.imshow() and waitKey() for faster batch processing

        else:
            # Break the loop if the end of the video is reached
            break

    # Release everything for this video
    out.release()
    cap.release()
    print(f"Finished. Saved to: {output_video_path}")
    # <--- MODIFIED (End of the main video loop)

cv2.destroyAllWindows()
print("\nAll videos processed successfully.")