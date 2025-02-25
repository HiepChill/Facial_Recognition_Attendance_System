import face_recognition
import argparse
import pickle
import cv2
import time
import imutils
import csv
import os
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to the output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face detection model to use: cnn or hog")
args = vars(ap.parse_args())

# Load encodings
print("[INFO] loading encodings...")
data = pickle.load(open(args["encodings"], "rb"))

# Initialize video stream
print("[INFO] starting video stream...")
video = cv2.VideoCapture(0)
video_writer = None

time.sleep(2.0)

# Prepare log directory and file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "attendance_log.csv")

# Initialize attendance log
today = datetime.now().strftime("%Y-%m-%d")
attendance = {}

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

        names.append(name)

        # Log attendance
        if name != "Unknown":
            if name not in attendance:
                attendance[name] = {"Time In": datetime.now().strftime("%H:%M:%S"), "Time Out": ""}
            else:
                attendance[name]["Time Out"] = datetime.now().strftime("%H:%M:%S")

    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if video_writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)

    if video_writer is not None:
        video_writer.write(frame)

    if args["display"] > 0:
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
video.release()
cv2.destroyAllWindows()
if video_writer is not None:
    video_writer.release()

# Write attendance log
with open(log_file, mode="w", newline="") as csvfile:
    log_writer = csv.writer(csvfile)
    log_writer.writerow(["Name", "Date", "Time In", "Time Out"])
    for name, times in attendance.items():
        log_writer.writerow([name, today, times["Time In"], times["Time Out"]])

print(f"[INFO] Attendance log saved to {log_file}")
