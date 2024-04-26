import cv2
import random
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture("pedestres.mp4")
# video_capture = cv2.VideoCapture(0)

filter = "none"
detect = True

options = "A - Blur\n S - Color\nD- Canny\n F - Blow Out\nG - Blow In\n H - Invert\n J - Contornos \n K - Detection On/Off\n Q - Quit\nL - Reset Filters"

def nothing(inp):
    pass

cv2.namedWindow("controls")

cv2.createTrackbar("r", "controls", 255, 1, nothing)
cv2.setTrackbarMax("r", "controls", 255)
cv2.setTrackbarMin("r", "controls", 0)

cv2.createTrackbar("g", "controls", 0, 1, nothing)
cv2.setTrackbarMax("g", "controls", 255)
cv2.setTrackbarMin("g", "controls", 0)

cv2.createTrackbar("b", "controls", 0, 1, nothing)
cv2.setTrackbarMax("b", "controls", 255)
cv2.setTrackbarMin("b", "controls", 0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error opening video")
        break

    if (detect):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            if filter == "blur":
                frame[y:y+h, x:x +
                      w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (63, 63), 0)
            elif filter == "color":
                colored_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
                # Modify the HSV values as per your desired color
                colored_face[:, :, 0] += random.randint(0, 180)
                frame[y:y+h, x:x+w] = colored_face
            elif filter == "contornos":
                r = cv2.getTrackbarPos("r", "controls")
                g = cv2.getTrackbarPos("g", "controls")
                b = cv2.getTrackbarPos("b", "controls")

                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

                _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(face_region, contours, -1, (b,g,r), thickness=2)
                frame[y:y+h, x:x + w] = face_region
            elif filter == "blowout":
                frame[y:y+h, x:x +
                      w] = cv2.dilate(face_region, np.ones((17, 17), dtype=np.uint8))
            elif filter == "blowin":
                frame[y:y+h, x:x +
                      w] = cv2.erode(face_region, np.ones((17, 17), dtype=np.uint8))

    if (filter == "canny"):
        edges = cv2.Canny(frame, threshold1=50, threshold2=150)

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        r = cv2.getTrackbarPos("r", "controls") / 255.0
        g = cv2.getTrackbarPos("g", "controls") / 255.0
        b = cv2.getTrackbarPos("b", "controls") / 255.0

        white = np.ones_like(edges_bgr, dtype=np.float64) * 255

        white[:, :, 0] *= b  # Blue channel
        white[:, :, 1] *= g  # Green channel
        white[:, :, 2] *= r  # Red channel

        white_uint8 = white.astype(np.uint8)

        colored_edges = cv2.bitwise_and(white_uint8, edges_bgr)

        frame = colored_edges

    elif filter == "invert":
        frame = cv2.bitwise_not(frame)

    cv2.putText(frame, f"Filter: {filter}", ((
        frame.shape[1]//2)-100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    y0 = frame.shape[0] - 300
    for i, line in enumerate(options.split('\n')):
        y = y0 + (30 * i)
        cv2.putText(frame, line, (50, y),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Video', frame)

    # Key mapping
    key = cv2.waitKey(1) & 0xFF

    if (key == ord("q")):
        break
    elif key == ord("a"):
        filter = "blur"
    elif key == ord("s"):
        filter = "color"
    elif key == ord("d"):
        filter = "canny"
    elif key == ord("f"):
        filter = "blowout"
    elif key == ord("g"):
        filter = "blowin"
    elif key == ord("h"):
        filter = "invert"
    elif key == ord("j"):
        filter = "contornos"
    elif key == ord("k"):
        detect = not detect
    elif key == ord("l"):
        filter = "none"
    

video_capture.release()
cv2.destroyAllWindows()