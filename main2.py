import cv2
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

video_capture = cv2.VideoCapture('teste2.mp4')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Detection
    results = face_detection.process(rgb_frame)

    # Extract face regions if faces are detected
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            face_region = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            # Display the face region (optional)
            if face_region.shape[0] <= 0 or face_region.shape[1] <= 0:
                continue
        
            frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = cv2.GaussianBlur(face_region, (63, 63), 0)
            # cv2.imshow('Face Region', face_region)

    # Display the processed frame (with bounding boxes if needed)
    cv2.imshow('Video', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()