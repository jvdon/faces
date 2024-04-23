import cv2
import random
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture("teste_face_only.mp4")

filter = "none"

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

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
        rgb *= np.array((0,1, 0),np.uint8) # set g and b to 0, leaves red :)
         # step 3: compose:
        # out = np.bitwise_or(frame, rgb)
        frame = rgb
    

    
    cv2.putText(frame, f"Filter: {filter}", (200, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    
    
    if(key == ord("q")):
        print(cv2.waitKey(1) & 0xFF)
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
    elif key == ord("l"):
        filter =  "none"
    

video_capture.release()
cv2.destroyAllWindows()
