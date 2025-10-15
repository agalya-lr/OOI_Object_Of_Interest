import cv2
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings

# Suppress pkg_resources warning
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

# Helper: deterministic color for each track ID
def color_for_id(track_id):
    rng = random.Random(int(track_id))
    return (rng.randint(30, 230), rng.randint(30, 230), rng.randint(30, 230))

# Load YOLO + DeepSORT
model = YOLO("yolov8n.pt")  # pretrained on COCO
tracker = DeepSort(max_age=30)
class_names = model.names  # COCO class names dictionary

# Video input/output
video_path = r"Video_Dataset/myvideo_padded.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("out/output_tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
count = 0
track_count = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    frame_count += 1
    if frame_count % int(fps) != 0:
        continue

    count += 1
    results = model.predict(frame, verbose=False)[0]

    # Prepare detections
    detections = []
    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        conf = float(score) if score is not None else 0.0
        detections.append(([x1, y1, x2, y2], conf, int(cls)))

    print(f"Frame {frame_count} (processed frame count {count}): {len(detections)} detections")

    # Update DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        track_count[track_id] = track_count.get(track_id, 0) + 1

        l, t, r, b = map(int, track.to_ltrb())
        color = color_for_id(track_id)

        cls = track.det_class if hasattr(track, "det_class") else -1
        conf = track.det_conf if hasattr(track, "det_conf") else 0.0
        class_name = class_names.get(cls, "Unknown")

        label = f"ID:{track_id} {class_name} {conf:.2f}"
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

for tid, cnt in track_count.items():
    print(f"Track ID {tid} appeared in {cnt} processed frames")

cap.release()
out.release()
cv2.destroyAllWindows()
print("Tracking finished. Output saved as output_tracked.mp4")
