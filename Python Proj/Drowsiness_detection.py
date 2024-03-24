import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = load_model("D:\Project\Python Proj\Model\model.h5")

mixer.init()
sound = mixer.Sound(r'D:\Project\Python Proj\alarm.wav')

def detect_faces():
    cap = cv2.VideoCapture(0)
    Score = 0
    while True:
        ret, frame = cap.read()
        height, width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, pt1=(ex, ey), pt2=(ex + ew, ey + eh), color=(255, 0, 0), thickness=3)

            # Preprocessing steps
            eye = frame[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            
            # Model prediction
            prediction = model.predict(eye)

            # If eyes are closed
            if prediction[0][0] > 0.30:
                cv2.putText(frame, 'closed', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                            color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, 'Score' + str(Score), (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                Score += 1
                if Score > 5:
                    try:
                        sound.play()
                    except:
                        pass

            # If eyes are open
            elif prediction[0][1] > 0.90:
                cv2.putText(frame, 'open', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                            color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, 'Score' + str(Score), (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                Score = Score-1
                if (Score<0):
                    Score = 0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
