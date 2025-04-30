# flask_zmq_server.py
import zmq
import torch
import numpy as np
import threading
import time
import cv2
from flask import Flask, render_template, Response, request, jsonify
from external.tamapp.efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
import json

app = Flask(__name__)

# Global state
tam_checkpoint = "external/tamapp/checkpoints/efficienttam_ti_512x512.pt"
model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"
classes = [0, 1, 2, 3]
click_points = {cls: [] for cls in classes}
current_class = 0

reset_flags= {
    "reset" : False,
    "reset_class" : 0,
    "current_frame_idx" : 0
}
latest_mask_bytes = None

no_obj_points = np.array([[0, 0]], dtype=np.float32)
no_obj_labels = np.array([-1], dtype=np.int32)

@app.route('/')
def index():
    return render_template('app_index.html')

@app.route('/video_feed')
def video_feed():
    def stream():
        while True:
            if latest_mask_bytes:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + latest_mask_bytes + b'\r\n')
            time.sleep(0.033)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click', methods=['POST'])
def handle_click():
    global click_points, current_class
    data = request.json
    click_points[current_class].append([data['x'], data['y']])
    return jsonify({'status': 'success'})

@app.route('/reset_class', methods=['POST'])
def reset_class():
    global reset_flags
    data = request.json
    cls = data['class']
    if cls in classes:
        reset_flags['reset'] = True
        reset_flags['reset_class'] = cls
    return jsonify({'status': 'success'})

@app.route('/change_class', methods=['POST'])
def change_class():
    global current_class
    data = request.json
    current_class = data['class']
    return jsonify({'status': 'success'})

def process_mask_frame(frame):
    global latest_mask_bytes
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, out_mask_logits = predictor.track(frame)

    new_input = False
    first_hit = True
    Gathered_matrix = {cls: {'points': [], 'labels': [], 'first_hit': []} for cls in classes}

    for cls in classes:
        if click_points[cls]:
            new_input = True
            for point in click_points[cls]:
                Gathered_matrix[cls]['points'].append(point)
                Gathered_matrix[cls]['labels'].append(1)
                Gathered_matrix[cls]['first_hit'].append(first_hit)
                first_hit = False
            click_points[cls] = []

    if new_input or reset_flags["reset"]:
        for cls in classes:
            if Gathered_matrix[cls]['points']:
                points = np.array(Gathered_matrix[cls]['points'], dtype=np.float32)
                labels = np.array(Gathered_matrix[cls]['labels'], dtype=np.int32)
                first_hit = np.array(Gathered_matrix[cls]['first_hit'], dtype=bool)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictor.add_new_points_during_track(cls, points, labels, first_hit[0], frame)
        if reset_flags["reset"]:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.add_new_points(
                    frame_idx=reset_flags["current_frame_idx"],
                    obj_id=reset_flags["reset_class"],
                    points=no_obj_points,
                    labels=no_obj_labels,
                    new_input=True
                )
            reset_flags["reset"] = False

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, out_mask_logits = predictor.finalize_new_input()

    mask_logits = out_mask_logits.cpu().numpy()
    frame_with_mask = apply_mask_to_frame(frame, mask_logits[:4])
    
    cv2.putText(frame_with_mask, f"Selected Class: {current_class}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, cls in enumerate(classes):
        color = (0, 255, 0) if cls == current_class else (200, 200, 200)
        cv2.putText(frame_with_mask, f"Class {cls}", (10, 60 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    ret, buffer = cv2.imencode('.jpg', frame_with_mask)
    if ret:
        latest_mask_bytes = buffer.tobytes()

    binary_mask = (mask_logits[0] > 0).astype(np.uint8) * 255
    for i in range(1, len(classes)):
        binary_mask = cv2.bitwise_or(binary_mask, (mask_logits[i] > 0).astype(np.uint8) * 255)

    return binary_mask

def apply_mask_to_frame(frame, mask_logits):
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
    mask_colored = np.zeros_like(frame, dtype=np.uint8)
    for i in range(mask_logits.shape[0]):
        mask = (mask_logits[i] > 0).astype(np.uint8) * 255
        for j in range(3):
            mask_colored[:, :, j] = np.clip(mask_colored[:, :, j] + mask * (colors[i][j] / 255), 0, 255)
    return cv2.addWeighted(frame, 0.6, mask_colored.astype(np.uint8), 0.4, 0)

def recv_image(socket):
    img_bytes = socket.recv_multipart()
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return image

def recv_array():
    meta_bytes, data_bytes = socket.recv_multipart()
    meta = json.loads(meta_bytes.decode('utf-8'))
    dtype = np.dtype(meta['dtype'])
    shape = tuple(meta['shape'])
    return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

def send_mask(array: np.ndarray, meta: dict={}):
    meta.update({'dtype': str(array.dtype), 'shape': array.shape})
    socket.send_multipart([
        json.dumps(meta).encode('utf-8'),  # 헤더: JSON 문자열
        array.tobytes()                    # 본문: 실제 데이터
    ])

    return array, meta

def zmq_loop():
    global reset_flags
    image = recv_array()
    binary_mask = predictor.load_first_frame(image, 4)
    # success, encoded_img = cv2.imencode('.jpg', binary_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    # if not success:
    #     raise RuntimeError("이미지 인코딩 실패")
    # socket.send_multipart(encoded_img.tobytes())
    send_mask(binary_mask.cpu().numpy())
    
    reset_flags["current_frame_idx"] +=1
    
    while True:
        image = recv_array()
        out_mask_logits = process_mask_frame(image)
        # success, encoded_img = cv2.imencode('.jpg', out_mask_logits, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        # if not success:
        #     raise RuntimeError("이미지 인코딩 실패")
        # socket.send_multipart(encoded_img.tobytes())
        # socket.send_pyobj(out_mask_logits_np)  # now it's bytes

        send_mask(out_mask_logits)
        reset_flags["current_frame_idx"] += 1
        
if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000)).start()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:1113")
    predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint)
    zmq_loop()