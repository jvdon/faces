import cv2
import random
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture("teste_face_only.mp4")

filter = "none"
detect = True

options = "A - Blur\n S - Color\nD- Canny\n F - Blow Out\nG - Blow In\n H - Invert\nJ - Detection On/Off\n Q - Quit\nL - Reset Filters"

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    if(detect):
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
            elif filter == "blowout":
                frame[y:y+h, x:x+w] = cv2.dilate(face_region, np.ones((17, 17), dtype=np.uint8))
            elif filter == "blowin":
                frame[y:y+h, x:x+w] = cv2.erode(face_region, np.ones((17, 17), dtype=np.uint8))

    if(filter == "canny"):
        edges = cv2.Canny(frame, threshold1=50, threshold2=150)
        rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # RGB for matplotlib, BGR for imshow() !
        # step 2: now all edges are white (255,255,255). to make it red, multiply with another array:
        rgb *= np.array((0,1, 0),np.uint8)
        
        frame = rgb
    elif filter == "invert":
        frame =  cv2.bitwise_not(frame)

    
    cv2.putText(frame, f"Filter: {filter}", ((frame.shape[1]//2)-100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    
    y0 = frame.shape[0] - 300
    for i, line in enumerate(options.split('\n')):
        y = y0 + (30 * i)
        cv2.putText(frame, line, (50, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    
    
    if(key == ord("q")):
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
        detect = not detect
    elif key == ord("l"):
        filter =  "none"
    

video_capture.release()
cv2.destroyAllWindows()
