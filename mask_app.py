import zmq
import torch
import numpy as np
import threading
import time
import cv2
import json
import argparse
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QGridLayout, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QCursor
from threading import Lock 

# 외부 모듈 (경로에 따라 수정 필요)
try:
    from external.tamapp.efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
except ImportError:
    print("Warning: external module 'efficient_track_anything' not found. This code will only run if you have the external dependencies.")
    
# ------------------- GLOBAL STATE & CONFIG -------------------
tam_checkpoint = "external/tamapp/checkpoints/efficienttam_ti.pt"
model_cfg = "configs/efficienttam/efficienttam_ti.yaml"

# Global state shared between the GUI (Main Thread) and the ZMQ Worker
classes = [0, 1, 2, 3]
click_points = {cls: [] for cls in classes}
box_prompts = {cls: [] for cls in classes}
current_class = 0
reset_flags = {"reset": False, "reset_class": 0, "current_frame_idx": 0}
active_text_prompt = ""

# Mutex for thread-safe access to shared state
state_mutex = Lock()

# ------------------- ZMQ Helper Functions -------------------

def recv_array(socket):
    """Receives a NumPy array over a ZMQ multipart message."""
    meta_bytes, data_bytes = socket.recv_multipart()
    meta = json.loads(meta_bytes.decode('utf-8'))
    dtype = np.dtype(meta['dtype'])
    shape = tuple(meta['shape'])
    return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

def send_mask(socket, array: np.ndarray, meta: dict={}):
    """Sends a NumPy array over ZMQ."""
    meta.update({'dtype': str(array.dtype), 'shape': array.shape})
    socket.send_multipart([
        json.dumps(meta).encode('utf-8'),
        array.tobytes()
    ])
    return array, meta

def send_mask_and_text(socket, mask: np.ndarray, text: str, frame_idx: int, class_id: int):
    """Sends a NumPy mask and text prompt data together over ZMQ."""
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

# ------------------- Image Processing Logic (Modified for PyQt) -------------------

def apply_mask_to_frame(frame, mask_logits):
    """
    마스크 영역만 블렌딩하고, 배경은 원본 화질을 100% 유지.
    여기서만 mask_logits의 채널/배치 차원을 2D (H, W)로 줄여서
    frame (H, W, 3) 과 boolean mask가 맞도록 처리합니다.
    """
    colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]  # BGR
    vis_frame = frame.copy()
    H, W = frame.shape[:2]

    for i in range(mask_logits.shape[0]):
        mask = mask_logits[i]

        # ---- 강건하게 2D (H, W) 마스크로 변환 ----
        if mask.ndim == 3:
            # 예상되는 케이스: (1, H, W) 또는 (H, W, 1)
            if mask.shape == (1, H, W):
                mask_2d = mask[0]               # (H, W)
            elif mask.shape == (H, W, 1):
                mask_2d = mask[:, :, 0]         # (H, W)
            else:
                # 그 외 예외 케이스는 squeeze로 최대한 맞춰봄
                mask_2d = np.squeeze(mask)
        elif mask.ndim == 2:
            mask_2d = mask
        else:
            # 예기치 않은 차원 수: squeeze 후 여전히 2D가 아니면 스킵
            mask_2d = np.squeeze(mask)
            if mask_2d.ndim != 2:
                print(f"[apply_mask_to_frame] Unexpected mask shape: {mask.shape}")
                continue

        # 여기까지 오면 mask_2d는 (H, W) 여야 함
        if mask_2d.shape != (H, W):
            # 크기가 프레임과 안 맞으면 안전하게 스킵
            print(f"[apply_mask_to_frame] Mask/frame size mismatch: mask {mask_2d.shape}, frame {(H, W)}")
            continue

        mask_bool = mask_2d > 0
        if not np.any(mask_bool):
            continue

        color = colors[i % len(colors)]

        # vis_frame[mask_bool] 의 shape: (num_pixels, 3)
        roi = vis_frame[mask_bool]
        blended = (roi * 0.6 + np.array(color, dtype=roi.dtype) * 0.4).astype(np.uint8)
        vis_frame[mask_bool] = blended

    return vis_frame


# ------------------- ZMQ Worker Thread -------------------

class ZmqWorker(QThread):
    """ZMQ 통신과 무거운 이미지 처리(Tracking)를 담당하는 쓰레드."""
    
    # PyQt Signal: 처리된 프레임을 메인 GUI로 보내기 위한 신호
    frame_updated = pyqtSignal(np.ndarray)
    
    def __init__(self, use_text, use_masks, use_dual, parent=None):
        super().__init__(parent)
        self.use_text = use_text
        self.use_masks = use_masks
        self.use_dual = use_dual
        self.predictor = None
        self.predictor_loaded = False
        self.context = None
        self.socket = None
        self.no_obj_points = np.array([[0, 0]], dtype=np.float32)
        self.no_obj_labels = np.array([-1], dtype=np.int32)
        self.classes = classes # Use global classes

    def init_predictor(self):
        """Lazy initialization of the heavy tracking model."""
        try:
            self.predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint)
            print("Tracking Predictor loaded successfully.")
        except Exception as e:
            print(f"Error loading predictor: {e}")
            self.predictor = None

    def process_mask_frame(self, frame):
        """
        Tracking 모델을 실행하고, 마스크를 프레임에 합성합니다.
        imencode 없이 순수 NumPy 배열을 반환합니다.
        """
        global click_points, box_prompts, reset_flags, active_text_prompt

        if self.predictor is None:
             # 임시로 원본 프레임만 반환
             return frame 

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_mask_logits = self.predictor.track(frame)

        new_input = False
        first_hit = True
        # 상태를 안전하게 읽기 위해 Mutex 사용
        with state_mutex:
            
            # 1. Gathered inputs from the GUI
            Gathered = {cls: {'points': [], 'labels': [], 'boxes': [], 'first_hit': []} for cls in self.classes}
            
            for cls in self.classes:
                if click_points[cls]:
                    new_input = True
                    for point in click_points[cls]:
                        Gathered[cls]['points'].append(point)
                        Gathered[cls]['labels'].append(1)
                        Gathered[cls]['first_hit'].append(first_hit)
                        first_hit = False
                    click_points[cls] = [] # Consume points
                
                if box_prompts[cls]:
                    new_input = True
                    for box in box_prompts[cls]:
                        Gathered[cls]['boxes'].append(box)
                        Gathered[cls]['first_hit'].append(first_hit)
                        first_hit = False
                    box_prompts[cls] = [] # Consume boxes

            # 2. Apply Prompts to Predictor
            if new_input or reset_flags["reset"]:
                for cls in self.classes:
                    if Gathered[cls]['points']:
                        points = np.array(Gathered[cls]['points'], dtype=np.float32)
                        labels = np.array(Gathered[cls]['labels'], dtype=np.int32)
                        first_hit = np.array(Gathered[cls]['first_hit'], dtype=bool)
                        self.predictor.add_new_prompts_during_track(cls, points=points, labels=labels,
                                                                    first_hit=first_hit[0], frame=frame)
                    elif Gathered[cls]['boxes']:
                        boxes = np.array(Gathered[cls]['boxes'], dtype=np.float32)
                        first_hit = np.array(Gathered[cls]['first_hit'], dtype=bool)
                        self.predictor.add_new_prompts_during_track(cls, boxes=boxes,
                                                                    first_hit=first_hit[0], frame=frame)

                if reset_flags["reset"]:
                    self.predictor.add_new_prompts(
                        frame_idx=reset_flags["current_frame_idx"],
                        obj_id=reset_flags["reset_class"],
                        points=self.no_obj_points,
                        labels=self.no_obj_labels,
                        new_input=True
                    )
                    reset_flags["reset"] = False
                    active_text_prompt = "" # Reset text on full reset

                _, _, out_mask_logits = self.predictor.finalize_new_input()

        # 3. Apply Mask to Frame
        mask_logits = out_mask_logits.cpu().numpy()
        frame_with_mask = apply_mask_to_frame(frame, mask_logits[:4])

        # 4. Prepare Mask for ZMQ Response
        # Consolidated binary mask (보내는 용도)
        binary_mask = (mask_logits[0] > 0).astype(np.uint8) * 255
        for i in range(1, len(self.classes)):
            binary_mask = cv2.bitwise_or(binary_mask, (mask_logits[i] > 0).astype(np.uint8) * 255)

        # frame_with_mask는 BGR 순서 (OpenCV 기본)
        return frame_with_mask, binary_mask

    def run(self):
        """The main loop for ZMQ communication and image processing."""
        self.init_predictor()
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        try:
            self.socket.bind("tcp://*:1113")
            print("ZMQ REP socket bound to tcp://*:1113")
        except zmq.error.ZMQError as e:
            print(f"ZMQ Error binding socket: {e}. Check if another process is running.")
            return

        while not self.isInterruptionRequested():
            try:
                # 1) RECV: get the next image from the client (blocks until available)
                image = recv_array(self.socket)

                # 2) Lazy-load first frame
                if not self.predictor_loaded and self.predictor is not None:
                    self.predictor.load_first_frame(image, len(self.classes))
                    self.predictor_loaded = True
                    reset_flags["current_frame_idx"] = 0
                    print("First frame loaded and tracking initialized.")

                reset_flags["current_frame_idx"] += 1
                frame_idx = reset_flags["current_frame_idx"]
                
                # 3) PROCESS
                frame_with_mask, mask_to_send = self.process_mask_frame(image)
                
                # 4) SEND ZMQ RESPONSE
                with state_mutex:
                    text_prompt = active_text_prompt

                if self.use_dual:
                    send_mask_and_text(self.socket, mask_to_send, text_prompt or "",
                                       frame_idx, current_class)
                elif self.use_masks:
                    send_mask(self.socket, mask_to_send)
                # elif use_text: (Text-only is not implemented for REP in this structure)

                # 5) PyQt GUI Update (Signal)
                self.frame_updated.emit(frame_with_mask) # BGR NumPy array

            except zmq.error.ContextTerminated:
                print("ZMQ Context Terminated. Exiting Worker.")
                break
            except Exception as e:
                print(f"An error occurred in ZMQ loop: {e}")
                time.sleep(0.1)


# ------------------- PyQt GUI -------------------

class VideoWidget(QLabel):
    """영상을 표시하고 마우스 클릭 이벤트를 처리하는 위젯."""
    
    # 텍스트 박스 사용을 위한 박스 드래그 플래그
    is_drawing = False
    start_point = None
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setAlignment(Qt.AlignCenter)
        self.setText("Waiting for video stream...")
        self.setCursor(QCursor(Qt.CrossCursor)) # 십자선 커서
        self.last_frame = None

    def mousePressEvent(self, event):
        """마우스 클릭 (점 프롬프트) 또는 박스 드래그 시작."""
        if event.button() == Qt.LeftButton:
            # 클릭 또는 박스 시작
            self.is_drawing = True
            self.start_point = event.pos()

    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 (점 프롬프트 완료 또는 박스 완료)."""
        if event.button() == Qt.LeftButton and self.is_drawing:
            end_point = event.pos()
            
            # 좌표계 변환 (픽셀 단위)
            frame_w, frame_h = self.last_frame.shape[1], self.last_frame.shape[0]
            label_w, label_h = self.width(), self.height()
            
            # 1. Affine Transformation을 고려한 실제 프레임 내 좌표 계산
            # 현재 위젯은 Qt.AlignCenter로 중앙 정렬되어 있다고 가정
            
            # 실제 프레임에 맞춰 위젯에 표시되는 영역 계산
            w_ratio = frame_w / label_w
            h_ratio = frame_h / label_h
            scale = max(w_ratio, h_ratio)

            disp_w = frame_w / scale
            disp_h = frame_h / scale
            
            x_offset = (label_w - disp_w) / 2
            y_offset = (label_h - disp_h) / 2
            
            def get_frame_coord(p):
                # 표시 영역 내 좌표
                x_disp = p.x() - x_offset
                y_disp = p.y() - y_offset
                
                # 원본 프레임 좌표
                x_frame = int(x_disp * scale)
                y_frame = int(y_disp * scale)
                
                # 범위 보정
                x_frame = max(0, min(x_frame, frame_w))
                y_frame = max(0, min(y_frame, frame_h))
                return x_frame, y_frame

            start_x, start_y = get_frame_coord(self.start_point)
            end_x, end_y = get_frame_coord(end_point)
            
            # 2. 클릭 vs. 박스 판정
            if abs(start_x - end_x) < 5 and abs(start_y - end_y) < 5:
                # 짧은 클릭 (Point Prompt)
                with state_mutex:
                    global click_points, current_class
                    click_points[current_class].append([start_x, start_y])
                    print(f"Click Prompt added: ({start_x}, {start_y}) for Class {current_class}")
            else:
                # 드래그 (Box Prompt)
                x1 = min(start_x, end_x)
                y1 = min(start_y, end_y)
                x2 = max(start_x, end_x)
                y2 = max(start_y, end_y)
                
                with state_mutex:
                    global box_prompts
                    box_prompts[current_class].append((x1, y1, x2, y2))
                    print(f"Box Prompt added: ({x1}, {y1}, {x2}, {y2}) for Class {current_class}")
                    
            self.is_drawing = False
            self.start_point = None

    def update_frame(self, frame_bgr):
        """BGR NumPy 배열을 받아 QLabel에 표시합니다."""
        if frame_bgr is None:
            return
        
        self.last_frame = frame_bgr # 좌표 변환을 위해 원본 저장
        
        # 1. BGR -> RGB 변환 (PyQt는 RGB를 기대함)
        rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # 2. QImage로 변환
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 3. 위젯 크기에 맞게 스케일링 후 표시
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        
    def resizeEvent(self, event):
        """위젯 크기 변경 시 이미지도 다시 스케일링하여 표시."""
        if self.last_frame is not None:
            self.update_frame(self.last_frame)
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("PyQt TAM Tracker GUI (High Quality)")
        self.setGeometry(100, 100, 1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        self.video_widget = VideoWidget(self)
        self.layout.addWidget(self.video_widget, 4) # 4/5 공간 할당
        
        self.create_controls_panel()
        
        self.args = args
        self.worker = None
        self.start_zmq_worker()

    def create_controls_panel(self):
        """컨트롤 버튼 및 입력창을 설정합니다."""
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setAlignment(Qt.AlignTop)
        
        # --- Class Selection Group ---
        class_group = QGroupBox("Select Object Class")
        class_layout = QGridLayout()
        self.class_buttons = []
        for i in classes:
            btn = QPushButton(f"Class {i}")
            btn.setCheckable(True)
            if i == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, cls=i: self.change_class(cls))
            class_layout.addWidget(btn, i // 2, i % 2)
            self.class_buttons.append(btn)
        class_group.setLayout(class_layout)
        controls_layout.addWidget(class_group)

        # --- Text Prompt Group ---
        text_group = QGroupBox("Text Prompt (Sticky)")
        text_layout = QVBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter object name (e.g., 'cat')")
        self.text_input.returnPressed.connect(self.send_text_prompt)
        self.text_send_button = QPushButton("Activate Text Prompt")
        self.text_send_button.clicked.connect(self.send_text_prompt)
        self.active_text_label = QLabel(f"Active Text: (None)")
        
        text_layout.addWidget(self.text_input)
        text_layout.addWidget(self.text_send_button)
        text_layout.addWidget(self.active_text_label)
        text_group.setLayout(text_layout)
        controls_layout.addWidget(text_group)
        
        # --- Reset/Info Group ---
        reset_group = QGroupBox("Actions")
        reset_layout = QVBoxLayout()
        
        self.reset_button = QPushButton("Reset Current Class")
        self.reset_button.clicked.connect(self.reset_current_class)
        reset_layout.addWidget(self.reset_button)
        
        self.status_label = QLabel("Status: Ready")
        reset_layout.addWidget(self.status_label)
        
        reset_group.setLayout(reset_layout)
        controls_layout.addWidget(reset_group)
        
        self.layout.addWidget(controls_panel, 1) # 1/5 공간 할당

    def start_zmq_worker(self):
        """ZMQ Worker 쓰레드를 시작하고 시그널을 연결합니다."""
        self.worker = ZmqWorker(self.args.use_text, self.args.use_masks, self.args.use_dual)
        self.worker.frame_updated.connect(self.video_widget.update_frame)
        self.worker.start()

    def closeEvent(self, event):
        """창 종료 시 쓰레드를 안전하게 종료합니다."""
        print("Stopping ZMQ Worker...")
        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait(2000) # 2초 대기
        super().closeEvent(event)

    def change_class(self, cls):
        """현재 클래스 ID를 변경하고 버튼 UI를 업데이트합니다."""
        with state_mutex:
            global current_class
            current_class = cls
        
        for btn in self.class_buttons:
            btn.setChecked(False)
        self.class_buttons[cls].setChecked(True)
        self.status_label.setText(f"Status: Class changed to {cls}")

    def send_text_prompt(self):
        """텍스트 프롬프트를 활성화합니다."""
        text = self.text_input.text().strip()
        if not text:
            self.active_text_label.setText(f"Active Text: (None)")
            return
            
        with state_mutex:
            global active_text_prompt
            active_text_prompt = text
            
        self.active_text_label.setText(f"Active Text: {text}")
        print(f"Text Prompt Activated: '{text}' for Class {current_class}")

    def reset_current_class(self):
        """현재 클래스의 트래킹을 리셋합니다."""
        with state_mutex:
            global reset_flags, current_class, click_points, box_prompts
            reset_flags['reset'] = True
            reset_flags['reset_class'] = current_class
            
            # GUI에 남아있을 수 있는 미사용 프롬프트도 초기화
            click_points[current_class] = []
            box_prompts[current_class] = []
            
        self.status_label.setText(f"Status: Class {current_class} Reset triggered")
        print(f"Reset triggered for Class {current_class}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_text', action='store_true', help='Enable text prompt forwarding')
    parser.add_argument('--use_masks', action='store_true', help='Enable mask forwarding')
    parser.add_argument('--use_dual', action='store_true', help='Enable dual forwarding (Mask + Text)')
    args = parser.parse_args()

    # ZMQ-REP 통신은 반드시 마스크(혹은 텍스트) 응답이 필요하므로, 최소한 하나는 활성화되어야 함
    if not (args.use_text or args.use_masks or args.use_dual):
        print("ERROR: ZMQ REP requires a response. Please specify at least one flag: --use_masks, --use_text, or --use_dual.")
        sys.exit(1)

    app = QApplication(sys.argv)
    
    # BGR-to-RGB 변환 시 색상 공간 오차를 줄이기 위해 Qt 이미지 캐시 비활성화 (선택 사항)
    app.setAttribute(Qt.AA_DontShowIconsInMenus, True) 
    
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec_())
