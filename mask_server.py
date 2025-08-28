# text_mask_server.py
import zmq
import torch
import numpy as np
import threading
import time
import cv2
import json
import argparse
from flask import Flask, render_template, Response, request, jsonify
from external.tamapp.efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor

app = Flask(__name__)

# ------------------- Predictor Init -------------------
tam_checkpoint = "external/tamapp/checkpoints/efficienttam_ti.pt"
model_cfg = "configs/efficienttam/efficienttam_ti.yaml"
predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint)

classes = [0, 1, 2, 3]
click_points = {cls: [] for cls in classes}
box_prompts   = {cls: [] for cls in classes}

current_class = 0
reset_flags= {"reset": False, "reset_class": 0, "current_frame_idx": 0}

# text state
active_text_prompt = ""
text_prompts_log = []

latest_mask_bytes = None
no_obj_points = np.array([[0, 0]], dtype=np.float32)
no_obj_labels = np.array([-1], dtype=np.int32)

# ------------------- Flask Routes -------------------

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

@app.route('/box', methods=['POST'])
def handle_box():
    global box_prompts, current_class
    data = request.json or {}
    x1, y1, x2, y2 = float(data['x1']), float(data['y1']), float(data['x2']), float(data['y2'])
    box_prompts[current_class].append((x1, y1, x2, y2))
    return jsonify({'status': 'success'})

@app.route('/text_prompt', methods=['POST'])
def text_prompt():
    global active_text_prompt, text_prompts_log, current_class
    data = request.json or {}
    text = (data.get('text') or "").strip()
    if not text:
        return jsonify({'status':'error','message':'Empty text'}), 400
    active_text_prompt = text
    text_prompts_log.append({
        "frame_idx": reset_flags["current_frame_idx"],
        "text": text,
        "class_id": current_class,
        "source": "text-sticky"
    })
    return jsonify({'status': 'success', 'active_text': active_text_prompt})

@app.route('/reset_class', methods=['POST'])
def reset_class():
    global reset_flags, active_text_prompt
    data = request.json
    cls = data['class']
    if cls in classes:
        reset_flags['reset'] = True
        reset_flags['reset_class'] = cls
        active_text_prompt = ""   # clear sticky text
    return jsonify({'status': 'success'})

@app.route('/change_class', methods=['POST'])
def change_class():
    global current_class
    data = request.json
    current_class = data['class']
    return jsonify({'status': 'success'})

# ------------------- Processing -------------------

def process_mask_frame(frame):
    global latest_mask_bytes
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, out_mask_logits = predictor.track(frame)

    new_input = False
    first_hit = True
    Gathered = {cls: {'points': [], 'labels': [], 'boxes': [], 'first_hit': []} for cls in classes}

    for cls in classes:
        if click_points[cls]:
            new_input = True
            for point in click_points[cls]:
                Gathered[cls]['points'].append(point)
                Gathered[cls]['labels'].append(1)
                Gathered[cls]['first_hit'].append(first_hit)
                first_hit = False
            click_points[cls] = []

    for cls in classes:
        if box_prompts[cls]:
            new_input = True
            for (x1, y1, x2, y2) in box_prompts[cls]:
                Gathered[cls]['boxes'].append([x1, y1, x2, y2])
                Gathered[cls]['first_hit'].append(first_hit)
                first_hit = False
            box_prompts[cls] = []

    if new_input or reset_flags["reset"]:
        for cls in classes:
            if Gathered[cls]['points']:
                points = np.array(Gathered[cls]['points'], dtype=np.float32)
                labels = np.array(Gathered[cls]['labels'], dtype=np.int32)
                first_hit = np.array(Gathered[cls]['first_hit'], dtype=bool)
                predictor.add_new_prompts_during_track(cls, points=points, labels=labels,
                                                       first_hit=first_hit[0], frame=frame)
            elif Gathered[cls]['boxes']:
                boxes = np.array(Gathered[cls]['boxes'], dtype=np.float32)
                first_hit = np.array(Gathered[cls]['first_hit'], dtype=bool)
                predictor.add_new_prompts_during_track(cls, boxes=boxes,
                                                       first_hit=first_hit[0], frame=frame)

        if reset_flags["reset"]:
            predictor.add_new_prompts(
                frame_idx=reset_flags["current_frame_idx"],
                obj_id=reset_flags["reset_class"],
                points=no_obj_points,
                labels=no_obj_labels,
                new_input=True
            )
            reset_flags["reset"] = False

        _, _, out_mask_logits = predictor.finalize_new_input()

    mask_logits = out_mask_logits.cpu().numpy()
    frame_with_mask = apply_mask_to_frame(frame, mask_logits[:4])

    frame_with_mask_rgb = cv2.cvtColor(frame_with_mask, cv2.COLOR_BGR2RGB)
    ret, buffer = cv2.imencode('.jpg', frame_with_mask_rgb)
    if ret:
        latest_mask_bytes = buffer.tobytes()

    # Consolidated binary mask
    binary_mask = (mask_logits[0] > 0).astype(np.uint8) * 255
    for i in range(1, len(classes)):
        binary_mask = cv2.bitwise_or(binary_mask, (mask_logits[i] > 0).astype(np.uint8) * 255)

    return binary_mask

def apply_mask_to_frame(frame, mask_logits):
    colors = [(0,255,0),(0,0,255),(255,0,0),(0,255,255)]
    mask_colored = np.zeros_like(frame, dtype=np.uint8)
    for i in range(mask_logits.shape[0]):
        mask = (mask_logits[i] > 0).astype(np.uint8) * 255
        for j in range(3):
            mask_colored[:,:,j] = np.clip(mask_colored[:,:,j] + mask * (colors[i][j] / 255), 0, 255)
    return cv2.addWeighted(frame, 0.6, mask_colored.astype(np.uint8), 0.4, 0)

# ------------------- ZMQ Helpers -------------------

def recv_array(socket):
    meta_bytes, data_bytes = socket.recv_multipart()
    meta = json.loads(meta_bytes.decode('utf-8'))
    dtype = np.dtype(meta['dtype'])
    shape = tuple(meta['shape'])
    return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

def send_mask(socket, array: np.ndarray, meta: dict={}):
    meta.update({'dtype': str(array.dtype), 'shape': array.shape})
    socket.send_multipart([
        json.dumps(meta).encode('utf-8'),
        array.tobytes()
    ])
    return array, meta

def send_text_prompt(socket, text: str, frame_idx: int, class_id: int):
    payload = {"frame_idx": frame_idx, "class_id": class_id, "text": text}
    socket.send_json(payload)

def send_mask_and_text(socket, mask: np.ndarray, text: str, frame_idx: int, class_id: int):
    meta = {
        "dtype": str(mask.dtype),
        "shape": mask.shape,
        "frame_idx": frame_idx,
        "class_id": class_id,
        "text": text
    }
    socket.send_multipart([
        json.dumps(meta).encode('utf-8'),
        mask.tobytes()
    ])

# ------------------- Main Loop -------------------

def zmq_loop(socket, use_text, use_masks, use_dual):
    """Strict REP semantics: recv -> send -> recv -> send ..."""
    predictor_loaded = False

    while True:
        # 1) RECV: get the next image from the client (blocks until available)
        image = recv_array(socket)

        # 2) Lazy-load first frame
        if not predictor_loaded:
            predictor.load_first_frame(image, len(classes))
            predictor_loaded = True
            reset_flags["current_frame_idx"] = 0

        reset_flags["current_frame_idx"] += 1
        frame_idx = reset_flags["current_frame_idx"]
        mask = process_mask_frame(image)
        if use_dual:
            send_mask_and_text(socket, mask, active_text_prompt or "",
                               frame_idx, current_class)

        elif use_masks:
            send_mask(socket, mask)

        elif use_text:
            send_text_prompt(socket, active_text_prompt or "",
                             frame_idx, current_class)

        reset_flags["current_frame_idx"] += 1

# ------------------- Entrypoint -------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_text', action='store_true', help='Enable text prompt forwarding')
    parser.add_argument('--use_masks', action='store_true', help='Enable mask forwarding')
    parser.add_argument('--use_dual', action='store_true', help='Enable dual forwarding')
    args = parser.parse_args()

    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000)).start()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:1113")

    zmq_loop(socket, args.use_text, args.use_masks, args.use_dual)
