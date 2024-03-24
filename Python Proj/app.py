from flask import Flask, render_template, Response
from Drowsiness_detection import detect_faces
from distraction_detection import detect_distraction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/distraction')
def distraction():
    return render_template('distraction.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/distraction_feed')
def distraction_feed():
    return Response(detect_distraction(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
