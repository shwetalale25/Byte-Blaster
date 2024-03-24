import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

# Labels for distraction classes
labels = {0: 'Safe driving', 1: 'Distracted'}

def detect_distraction():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (299, 299))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_resized = np.expand_dims(frame_resized, axis=0)
        frame_resized = preprocess_input(frame_resized)
        prediction = model.predict(frame_resized)
        label = labels[np.argmax(prediction)]
        cv2.putText(frame,"Activity:{}".format(label),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        ret, buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
