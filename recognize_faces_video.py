import face_recognition
import argparse
import pickle
import cv2
import time
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to the output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection_method", type=str, default="hog", help="face detection model to use: cnn or hog")
args = vars(ap.parse_args())

# Load known faces and encodings
print("[INFO] loading encodings...")
data = pickle.load(open(args["encodings"], "rb"))

# Start video stream
print("[INFO] starting video stream...")
video = cv2.VideoCapture(0)
writer = None
time.sleep(2.0)

frame_count = 0  # Counter to process every N frames
process_every_n_frames = 2  # Adjust for performance

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = imutils.resize(frame, width=500)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every N frames
    if frame_count % process_every_n_frames == 0:
        boxes = face_recognition.face_locations(rgb_small, model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb_small, boxes)
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

    # Scale bounding boxes back to original frame size
    for ((top, right, bottom, left), name) in zip(boxes, names):
        scale = frame.shape[1] / float(small_frame.shape[1])
        top = int(top * scale)
        right = int(right * scale)
        bottom = int(bottom * scale)
        left = int(left * scale)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Initialize VideoWriter after getting frame size
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

    # Display the output frame
    if args["display"] > 0:
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_count += 1

video.release()
cv2.destroyAllWindows()

if writer is not None:
    writer.release()
