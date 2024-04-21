import cv2
import random

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture("teste2.mp4")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        # if (random.random() > 0.5):
        #     colored_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        #     # Modify the HSV values as per your desired color
        #     colored_face[:, :, 0] += random.randint(0, 180)
        #     colored_face = cv2.cvtColor(colored_face, cv2.COLOR_HSV2RGB)
        #     frame[y:y+h, x:x+w] = colored_face
        # else:
        frame[y:y+h, x:x +
                  w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (63, 63), 0)

        # background_image[y:y+h, x:x+w] = face_region
        # faceC = random.randint(0, 10)
        # cv2.imshow(f"Face ~> {faceC}", face_region)

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
