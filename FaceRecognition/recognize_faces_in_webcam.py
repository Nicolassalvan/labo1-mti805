import face_recognition
import imutils
import pickle
import cv2
import os

# Path to face encodings
faceenc_path = "FaceRecognition/face_enc"

# Find path of XML files containing Haarcascade files
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheye = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye.xml"
cascPathsmile = os.path.dirname(cv2.__file__) + "/data/haarcascade_smile.xml"

# Load Haarcascade classifiers
faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheye)
smileCascade = cv2.CascadeClassifier(cascPathsmile)

# Load the known faces and embeddings saved in the last file
data = pickle.loads(open(faceenc_path, "rb").read())

print("Streaming started")
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # Set width
video_capture.set(4, 480)  # Set height

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get the facial embeddings for faces in the input
    encodings = face_recognition.face_encodings(rgb)
    names = []

    # Loop over the facial embeddings for multiple faces
    for encoding in encodings:
        # Compare encodings with those in data["encodings"]
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

    # Loop over the detected faces
    for ((x, y, w, h), name) in zip(faces, names):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(5, 5),
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect smiles within the face ROI
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    # Display the output
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()