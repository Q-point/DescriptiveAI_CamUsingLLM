from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import time
import requests

# OpenAI GPT parameters
prompt_prefix = 'llava:13b'
prompt_suffix = 'Provide a single sentence description of this image:'
response_text = ""
api_endpoint = "http://localhost:11434/api/generate"


app = Flask(__name__)


global_frame = None

def generate_frames():
    camera = cv2.VideoCapture(0)
    global global_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            global_frame = frame  # Update global frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global global_frame
    response_text = ""  # Initialize response_text

    if global_frame is not None:
        ret, img_buffer = cv2.imencode('.jpg', global_frame)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')

        api_url = api_endpoint
        payload = {
            "model": prompt_prefix,
            "prompt": "what is in this picture?",
            "stream": False,
            "images": [img_base64]
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', "")

            print(response_text, end='', flush=True)
        return jsonify(response_text=response_text)  # Return JSON response with key 'response_text'



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)  

