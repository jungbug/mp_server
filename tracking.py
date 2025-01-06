from flask import Flask, Response, request, jsonify, make_response
import base64, ssl, json
import cv2
import time
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import os

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

app = Flask(__name__)

# ==== 전역 변수 ====
framesA = []         # 영상 A의 모든 프레임(JPEG 바이너리) 리스트
current_frame_idx = 0  # 현재 몇 번째 프레임을 송출 중인지
switch_to_ai = False   # False면 원본 영상 재생, True면 AI 결과 재생
fps = 30               # 영상의 FPS (A와 AI 결과 동일 가정)

# YOLO 모델 로드
det_model = YOLO("yolov8n.pt")

# ==== MJPEG 프레임 로드 함수 ====
def load_video_frames(video_path):
    """ OpenCV로 영상을 읽어들여 각 프레임을 JPEG로 인코딩한 뒤 바이트로 저장 """
    cap = cv2.VideoCapture(video_path)
    loaded_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ret_jpg, jpg_buffer = cv2.imencode('.jpg', frame)
        if ret_jpg:
            loaded_frames.append(jpg_buffer.tobytes())
        else:
            continue

    cap.release()
    return loaded_frames

input_image = cv2.imread('test.png')

input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (224, 224))

similarity_model = tf.keras.models.load_model('binary97.h5')
max_cosine_distance = 0.7  # 거리 임계값
nn_budget = None  # 특징 벡터 크기 제한
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

track_paths = {}
similar_tracks = set()

# 유사도가 높은 트랙 ID만 추적
tracked_similar_id = None

frame_count_ = 0

# ==== AI 모델로 프레임 처리 ====
def process_frame_with_ai(frame):
    """AI 모델로 프레임을 처리하여 박스와 결과를 렌더링"""
    result = det_model(frame)
    processed_frame = frame.copy()
    detections = []
    for box in result[0].boxes:
        if box.cls == 0:  # 'person' 클래스
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            bbox = [x1, y1, x2 - x1, y2 - y1]  # x, y, width, height
            
            # Detection 객체 생성 (feature 없이)
            detections.append(Detection(bbox, conf, 'person', []))
            
    # 트래커 업데이트
    tracker.predict()  # 다음 상태 예측
    tracker.update(detections)  # 탐지 결과 업데이트
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 5:
            print(f"Track {track.track_id} not confirmed or outdated.")
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()  # Bounding box 좌표

        # 만약 유사도 모델로 확인된 트랙이 있으면 해당 트랙만 추적
        # bbox = (x1, y1, w, h)라고 가정
        x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # 이미지 크롭
        try:
            crop_image = frame[y1:h, x1:w]
            crop_image = cv2.resize(crop_image, (224, 224))
            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            #cv2.imwrite('output_image.jpq', crop_image)
            #break
            combined_image = np.hstack((input_image, crop_image))
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
            combined_image = tf.image.resize(combined_image, (224, 224))
            combined_image = np.expand_dims(combined_image, axis=0)

            person_score = similarity_model.predict(combined_image)
            print(person_score)

            if person_score >= 0.90:
                tracked_similar_id = track_id

            # 특정 트랙만 추적
            if tracked_similar_id == track_id:
                # 추적한 사람 그리기
                cv2.rectangle(processed_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Score: {person_score}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # 트랙 경로 그리기
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)

                # 트랙의 경로 저장
                if track_id not in track_paths:
                    track_paths[track_id] = []
                track_paths[track_id].append((center_x, center_y))

                # 트랙 경로를 선으로 그리기
                for j in range(1, len(track_paths[track_id])):
                    cv2.line(processed_frame, track_paths[track_id][j - 1], track_paths[track_id][j], (0, 0, 255), 2)
        except:
            continue

    ret_jpg, jpg_buffer = cv2.imencode('.jpg', processed_frame)
    return jpg_buffer.tobytes() if ret_jpg else None

# ==== 실제 스트리밍(제너레이터) ====
def generate_mjpeg():
    global current_frame_idx, switch_to_ai

    while True:
        if not switch_to_ai:
            # 원본 영상 프레임 송출
            frame_data = framesA[current_frame_idx]
        else:
            # AI 처리 결과 프레임 송출
            frame = cv2.imdecode(np.frombuffer(framesA[current_frame_idx], np.uint8), cv2.IMREAD_COLOR)
            frame_data = process_frame_with_ai(frame) or framesA[current_frame_idx]
             
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        current_frame_idx += 1

        # 마지막 프레임에 도달 시 다시 0으로 되돌아감
        if current_frame_idx >= len(framesA):
            current_frame_idx = 0

        time.sleep(2.0 / fps)

# ==== Flask 라우트 ====
@app.route('/')
def index():
    return """
    <html>
      <head><title>MJPEG Test</title></head>
      <body>
        <h1>MJPEG Streaming Demo</h1>
        <img src="/video_feed" />
        <br/><br/>
        <button onclick="fetch('/switch?to=ai')">Switch to AI Processed</button>
        <button onclick="fetch('/switch?to=original')">Switch to Original</button>
        <button onclick="fetch('/reset')">Reset Frame Index</button>
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch')
def switch():
    """원본 또는 AI 처리 영상으로 스위치"""
    global switch_to_ai
    mode = request.args.get('to', 'original')
    switch_to_ai = (mode == 'ai')
    return f"Switched to {'AI Processed' if switch_to_ai else 'Original'} Video"

@app.route('/reset')
def reset_frame_index():
    """현재 프레임 인덱스를 0으로 리셋"""
    global current_frame_idx
    current_frame_idx = 0
    return "Frame index reset to 0."
    
@app.route('/upload', methods=['POST'])
def process_images():
    try:
        data = request.json
        images = data.get('images', [])

        if not images:
            return jsonify({"success": False, "message": "No images provided"}), 400
        
        # 이미지를 저장하거나 처리
        save_dir = "uploaded_images"
        os.makedirs(save_dir, exist_ok=True)
        # 이전 이미지 삭제
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # 파일 삭제
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")

        for idx, image_b64 in enumerate(images):
            image_data = base64.b64decode(image_b64)
            image_path = os.path.join(save_dir, f"image_{idx + 1}.jpg")
            with open(image_path, "wb") as f:
                f.write(image_data)

        return jsonify({"success": True, "message": "Images processed successfully."})
    except Exception as e:
        print("Error processing images:", str(e))
        return jsonify({"success": False, "message": "Error processing images."}), 500



# ==== 초기화 단계 ====
if __name__ == '__main__':
    framesA = load_video_frames('test.mp4')  # 원본 영상 로드
    app.run(host='0.0.0.0', port=4000, debug=True)
