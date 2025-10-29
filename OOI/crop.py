import cv2
import random
import numpy as np
from ultralytics import YOLO
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
# Helper: deterministic color for each track ID

def color_for_id(track_id):
    rng = random.Random(int(track_id))
    return (rng.randint(30, 230), rng.randint(30, 230), rng.randint(30, 230))

# Load model and tracker
def model_tracker():

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)
    class_names = model.names
    return model , tracker , class_names

#video setup
def video_setup(video_path, output_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        exit()

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    PROCESS_EVERY_SEC = True
    stride = int(max(10, round(fps))) if PROCESS_EVERY_SEC else 1

    crop_height = height
    crop_width  = (crop_height * 9) // 16 
    crop_width  = min(crop_width, width - (width % 2))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_width, crop_height))
    return cap, out, width, height, fps, crop_width ,stride , crop_height

#processing
def video_process(video_path, output_path):
    model, tracker, class_names = model_tracker()    
    cap, out, width, height, fps, crop_width, stride, crop_height = video_setup(video_path, output_path)
    
    # Use original fps for smooth playback
    fps_out = fps
    frame_count = 0
    frame_skip = 4
    count = 0
    track_count = {} 

    crop_y1 = 0 # top fixed as 0
    crop_y2 = crop_height  # vertical bounds fixed

    out = cv2.VideoWriter("out/chef.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_width, crop_height))

    smooth_queue = deque(maxlen=20)
    last_crop_x1 = width // 2

    # Processing frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break
        frame_count += 1
        
        # Process every stride frames for detection, but write every frame
        if frame_count % stride == 0:
            count += 1

            #  YOLO detection
            results = model.predict(frame, verbose=False)[0]
            detections = []
            obj_centers = []  # collect center coordinates of detected objects eg.[(100,50),(150,67),...]
            for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                conf = float(score) if score is not None else 0.0
                detections.append(([x1, y1, x2, y2], conf, int(cls)))
                obj_centers.append(((x1 + x2) // 2, (y1 + y2) // 2))  # object center
            print(f"Frame {frame_count} (processed frame count {count}): {len(detections)} detections")

            #  DeepSORT tracking
            tracks = tracker.update_tracks(detections, frame=frame)
            #  Compute horizontal moving crop box (fixed 9:16, fixed height)
            if obj_centers:
                avg_x = int(sum([c[0] for c in obj_centers]) / len(obj_centers))
            else:
                avg_x = width // 2  # default center
            smooth_queue.append(avg_x)
            smooth_avg_x = int(np.mean(smooth_queue))
            last_crop_x1 = smooth_avg_x - crop_width // 2
            last_crop_x1 = max(0, min(last_crop_x1, width - crop_width))  # keep crop box inside frame
        
        # Use the last computed crop position for smooth movement
        crop_x1 = last_crop_x1
        crop_x2 = crop_x1 + crop_width # right boundary of the crop box

        # # draw crop box
        # cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 255), 2)
        # cv2.putText(frame, "Crop Box", (crop_x1, crop_y1 + 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw tracked objects (only when we have tracks from processing)
        if frame_count % stride == 0:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                # print(track_id)

                l, t, r, b = map(int, track.to_ltrb())
                color = color_for_id(track_id)
                cls = track.det_class if hasattr(track, "det_class") else -1
                conf = track.det_conf if hasattr(track, "det_conf") else 0.0
                class_name = class_names.get(cls, "Unknown")
                if track_id not in track_count:
                    track_count[track_id] = [0, class_name]
                track_count[track_id][0] += 1

                # draw objects if they overlap with crop box
            #     if not (r < crop_x1 or l > crop_x2 or b < crop_y1 or t > crop_y2):
            #         conf = float(score) if score is not None else 0.0
            #         label = f"ID:{track_id} {class_name} {conf:.2f}"
            #         cv2.rectangle(frame, (l, t), (r, b), color, 2)
            #         cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        cropping = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # changed: ensure dtype/contiguity
        if cropping.dtype != np.uint8:
            cropping = cropping.astype(np.uint8)
        cropping = np.ascontiguousarray(cropping)
        out.write(cropping)
    # Print tracking summary
    sorting = sorted(track_count.items(), key=lambda x: int(x[0]))
    for trk_id, (cont, cls_name) in sorting:
        print(trk_id, cont, cls_name)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Tracking finished. Output saved as out/chef.mp4")


if __name__ == "__main__":
    video_process(
        video_path=r"E:\Internship-Works\OOI\Video_Dataset\chef_ad.mp4", output_path="out/chef.mp4" )
