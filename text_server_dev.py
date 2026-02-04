# text_mask_server.py  (text-only version)
import zmq
import torch
import numpy as np
import threading
import time
import json
import argparse
from flask import Flask, render_template, request, jsonify

# Tokenizer
from transformers import BertTokenizerFast

app = Flask(__name__)

# ------------------- State -------------------
classes = [0, 1, 2, 3]
current_class = 0

# "reset"은 텍스트 sticky를 clear하는 용도로만 유지
reset_flags = {"reset": False, "reset_class": 0, "current_frame_idx": 0}

# ------------------- Text (Sticky) State -------------------
active_text_prompt = ""  # sticky (string)
text_prompts_log = []

# ------------------- BERT Tokenizer -------------------
tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-small")
MAX_TEXT_LEN = 16

ORD = ["first", "second", "third", "fourth", "fifth"]

def numeric_to_text(n1: int, n2: int, n3: int) -> str:
    """
    n1: 0/1 -> "blue block" / "orange block"
    n2: 0 -> no text, 1~5 -> "{ordinal} from the left"
    n3: 0 -> no text, 1~5 -> "{ordinal} from the top"
    join with comma into one string
    """
    parts = []

    if n1 == 0:
        parts.append("blue block")
    elif n1 == 1:
        parts.append("orange block")

    if 1 <= n2 <= 5:
        parts.append(f"{ORD[n2-1]} from the left")

    if 1 <= n3 <= 5:
        parts.append(f"{ORD[n3-1]} from the top")

    return ", ".join(parts)

@torch.no_grad()
def tokenize_prompt(text: str):
    """
    returns:
      input_ids:       (MAX_TEXT_LEN,) np.int64
      attention_mask:  (MAX_TEXT_LEN,) np.int64
    """
    enc = tokenizer(
        text if text is not None else "",
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0].to(torch.int64).cpu().numpy()
    attention_mask = enc["attention_mask"][0].to(torch.int64).cpu().numpy()
    return input_ids, attention_mask

# ------------------- Flask Routes -------------------

@app.route("/")
def index():
    return render_template("app_index.html")

# NEW: numeric prompt endpoint
@app.route("/numeric_prompt", methods=["POST"])
def numeric_prompt():
    """
    Expected JSON:
      { "n1": int, "n2": int, "n3": int }
    Sticky: active_text_prompt 유지. reset_class에서만 clear.
    """
    global active_text_prompt, text_prompts_log, current_class
    data = request.json or {}

    try:
        n1 = int(data.get("n1", 0))
        n2 = int(data.get("n2", 0))
        n3 = int(data.get("n3", 0))
    except Exception:
        return jsonify({"status": "error", "message": "n1/n2/n3 must be integers"}), 400

    text = numeric_to_text(n1, n2, n3).strip()
    if not text:
        return jsonify({"status": "error", "message": "Empty prompt after conversion"}), 400

    active_text_prompt = text
    text_prompts_log.append(
        {
            "frame_idx": reset_flags["current_frame_idx"],
            "numeric": {"n1": n1, "n2": n2, "n3": n3},
            "text": text,
            "class_id": current_class,
            "source": "numeric-sticky",
        }
    )
    return jsonify({"status": "success", "active_text": active_text_prompt})

@app.route("/reset_class", methods=["POST"])
def reset_class():
    """
    mask가 없으니, 여기서는 '해당 클래스 리셋' 의미를
    sticky text clear 용도로만 사용.
    """
    global reset_flags, active_text_prompt
    data = request.json or {}
    cls = data.get("class", None)
    if cls in classes:
        reset_flags["reset"] = True
        reset_flags["reset_class"] = cls
        active_text_prompt = ""  # clear sticky text
    return jsonify({"status": "success"})

@app.route("/change_class", methods=["POST"])
def change_class():
    global current_class
    data = request.json or {}
    cls = data.get("class", 0)
    if cls in classes:
        current_class = cls
    return jsonify({"status": "success", "current_class": current_class})

# ------------------- ZMQ Helpers -------------------

def recv_array(socket):
    meta_bytes, data_bytes = socket.recv_multipart()
    meta = json.loads(meta_bytes.decode("utf-8"))
    dtype = np.dtype(meta["dtype"])
    shape = tuple(meta["shape"])
    return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

def send_text_tokens(socket, text: str, frame_idx: int, class_id: int):
    input_ids, attention_mask = tokenize_prompt(text)
    payload = {
        "frame_idx": frame_idx,
        "class_id": class_id,
        "text": text,
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "max_length": MAX_TEXT_LEN,
        "tokenizer": "prajjwal1/bert-small",
    }
    socket.send_json(payload)

# ------------------- Main Loop -------------------

def zmq_loop_text_only(socket):
    """Strict REP semantics: recv -> send -> recv -> send ..."""
    while True:
        # 1) RECV: get the next image from the client (blocks until available)
        _image = recv_array(socket)  # 이미지 자체는 여기서는 사용 안 함

        # 2) frame idx update
        reset_flags["current_frame_idx"] += 1
        frame_idx = reset_flags["current_frame_idx"]

        # 3) sticky text
        text = active_text_prompt or ""

        # 4) SEND: tokens only
        send_text_tokens(socket, text, frame_idx, current_class)

# ------------------- Entrypoint -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_text",
        action="store_true",
        help="(kept for compatibility) Enable tokenized prompt forwarding",
    )
    args = parser.parse_args()

    # Flask UI thread
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True
    ).start()

    # ZMQ REP server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:1113")

    zmq_loop_text_only(socket)
