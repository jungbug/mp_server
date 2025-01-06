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

#############################
# 예시로 추가한 라이브러리들 #
#############################
import imageio.v3 as iio
import matplotlib.pyplot as plt

app = Flask(__name__)

# ==== 전역 변수 ====
framesA = []           # 영상 A의 모든 프레임(JPEG 바이너리) 리스트
current_frame_idx = 0  # 현재 몇 번째 프레임을 송출 중인지
switch_to_ai = False   # False면 원본 영상 재생, True면 AI 결과 재생
fps = 30               # 영상의 FPS (A와 AI 결과 동일 가정)

# YOLO 모델 로드
det_model = YOLO("yolov8n.pt")

# DEEPSORT 설정
max_cosine_distance = 0.7
nn_budget = None
model_filename = 'mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# 타겟 유사도 비교 threshold
sim_threshold = 0.09  # 예시

#######################################
# 다수결 앙상블 함수(투표 방식 + 타이브레이크)
#######################################
def ensemble_similar_id(candidates_dicts):
    """
    여러 target 후보(candidates) 딕셔너리를 받아
    1) 다수결(투표) 방식으로 가장 많이 등장한 ID 선정
    2) 만약 최다 득표가 동률이면, 각 target과의 평균 유사도(더 낮을수록 비슷함)가 가장 낮은 ID 선정
    :param candidates_dicts: 예) [ {id1: sim1, id2: sim2}, {id1: sim3, id3: sim4}, ... ]
    :return: 최종 picked_id
    """
    import numpy as np
    
    # 모든 candidates에서 가장 작은 유사도(=가장 비슷한) ID를 골라 리스트에 담기
    # (각 target 딕셔너리에서 argmin에 해당하는 ID)
    all_ids = []
    for candidates in candidates_dicts:
        if not candidates:
            continue
        # candidates = {id1: sim1, id2: sim2, ...}
        best_sim_val = np.sort(list(candidates.values()))[0]  # 가장 낮은(sim) 값
        picked_id = similar_id(candidates, best_sim_val)      # 그 값에 해당하는 ID
        all_ids.append(picked_id)

    # 만약 candidates 자체가 아무것도 없었다면(None 방지)
    if not all_ids:
        return None

    # 다수결(투표) 로직
    unique_ids = list(set(all_ids))  # 고유 ID 추출
    id_to_index = {uid: idx for idx, uid in enumerate(unique_ids)}  # ID -> 인덱스 매핑

    one_hot_counts = np.zeros(len(unique_ids), dtype=int)
    for pid in all_ids:
        one_hot_counts[id_to_index[pid]] += 1

    max_count = np.max(one_hot_counts)
    # 동률인지 확인
    tie_indices = np.where(one_hot_counts == max_count)[0]

    # 동률 아님 -> 그냥 최다 득표 ID 리턴
    if len(tie_indices) == 1:
        final_id = unique_ids[tie_indices[0]]
    else:
        # 동률 -> 평균 유사도(낮을수록 좋음)로 결정
        tie_ids = [unique_ids[i] for i in tie_indices]
        
        best_id = None
        best_score = float('inf')
        
        # tie_ids 중에서 모든 target과의 유사도 평균이 가장 낮은(=비슷한) ID 찾기
        for tid in tie_ids:
            # tid가 각 target 딕셔너리에 있을 경우만 모아서 평균
            scores = []
            for cand_dict in candidates_dicts:
                if tid in cand_dict:
                    scores.append(cand_dict[tid])
            if len(scores) > 0:
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_id = tid
        final_id = best_id

    return final_id


def similar_id(dictionary, value, threshold=1e-5):
    """
    dictionary에서 value와 가장 가까운 값을 갖는 key를 찾되,
    threshold 내에 들어올 때만 반환. threshold보다 크면 None
    """
    import numpy as np
    
    r = None
    last = np.inf
    for k in dictionary.keys():
        if abs(value - dictionary[k]) < last and abs(value - dictionary[k]) < threshold:
            last = abs(value - dictionary[k])
            r = k
    return r


def detect_person(img, conf_threshold=0.5, ret_coord=False):
    result = det_model.cpu()(img, verbose=False)
    
    boxes_out = np.array([np.concatenate([
        np.array(box.xywh[0])[:2] - np.array(box.xywh[0])[2:] / 2,
        np.array(box.xywh[0])[2:]
    ]) for box in result[0].boxes if box.cls == 0 and box.conf >= conf_threshold])

    conf_out = np.array([
        np.array(box.conf)
        for box in result[0].boxes if box.cls == 0 and box.conf >= conf_threshold
    ])

    boxes = np.array([
        np.array([
            box.xyxyn[0][1],
            box.xyxyn[0][0],
            box.xyxyn[0][3],
            box.xyxyn[0][2]
        ]) for box in result[0].boxes if box.cls == 0 and box.conf >= conf_threshold
    ])

    _img = tf.repeat(img[None], len(boxes), axis=0)
    cropped_image = tf.image.crop_and_resize(_img, boxes, range(len(boxes)), (128, 64))

    if ret_coord:
        return cropped_image, boxes_out, conf_out
    else:
        return cropped_image


###############################
# AI 처리 (사람 감지 + 트래킹) #
###############################
def process_frame_with_ai(frame):
    global tracker  # 이미 생성해둔 전역 tracker 사용
    processed_frame = frame.copy()

    # DeepSORT 처리
    crops, boxes, scores = detect_person(frame, ret_coord=True)
    names = ['person' for _ in range(len(boxes))]
    features = encoder(frame, boxes)

    detections = [
        Detection(bbox, score, class_name, feature)
        for bbox, score, class_name, feature in zip(boxes, scores, names, features)
    ]

    tracker.predict()
    tracker.update(detections)

    tracked_bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 5:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        tracking_id = track.track_id
        tracked_bboxes.append(bbox.tolist() + [tracking_id])

    # ---------------------------------------------------------
    # 아래 부분: 여러 target과 비교하기 위한 candidates_list 로직 예시
    # ---------------------------------------------------------

    # 현재 프레임에서 각 ID별 crop들
    objects = {}  
    for t in tracked_bboxes:
        x1, y1, x2, y2, pid = t
        # tf.image.crop_and_resize를 위해 (y1, x1, y2, x2) 순으로 만들어줌
        # 정규화 필요하므로 frame 크기로 나누기
        frame_h, frame_w = frame.shape[:2]
        _t = [y1 / frame_h, x1 / frame_w, y2 / frame_h, x2 / frame_w]

        cropped = tf.image.crop_and_resize(
            frame[None],
            np.array([_t]),
            [0],
            (128, 64)
        )[0].numpy()

        if pid not in objects:
            objects[pid] = []
        objects[pid].append(cropped)

    # -----------------------------------
    # 아래는 "여러 개의 target" 비교 예시
    # -----------------------------------
    # 실제 코드는, target 이미지를 여러분이 따로 로드해서
    # 각 프레임마다 반복하거나, 처음에 미리 feature만 뽑아놓고 비교하는 방식 등
    # 상황에 따라 다양하게 구현할 수 있습니다.

    # 예: target 이미지 여러 장 로드했다고 가정
    #    target_images = [iio.imread(path)[..., :3], iio.imread(path2)[..., :3], ...]
    # 그 각각에 대해 candidates를 구한 뒤,
    # candidates_list에 append -> ensemble_similar_id로 최종 picked_id 결정

    # 여기선 예시로 target_images를 전역이든, 다른 전처리에 있든 있다고 치고,
    # 아래처럼 임시로 하나 만든다고 가정합니다.
    dummy_target_path = "test.png"   # 예시
    # 이미지 로드 (실제로는 iio.imread 등을 사용)
    if os.path.exists(dummy_target_path):
        target_img = cv2.imread(dummy_target_path)[..., :3]
    else:
        target_img = frame[0:128, 0:64, :]  # 그냥 임시 잘라서 씀

    # 여러 target이 있다고 가정하여 리스트로 만듦 (여기서는 2개 예시)
    target_images = [target_img, target_img]

    # target 별 candidates 딕셔너리를 모으기
    candidates_list = []
    for target in target_images:
        objects_sim = {}
        for pid, cropped_list in objects.items():
            ret = []
            for o in cropped_list:
                feature1 = encoder(np.array(target), [[0, 0, 64, 128]])
                feature2 = encoder(np.array(o), [[0, 0, 64, 128]])
                ret.append(np.mean((feature1 - feature2) ** 2) ** 0.5)
            objects_sim[pid] = ret

        # 각 target에 대해서, 하위 10퍼센트 구간의 평균 유사도가 sim_threshold 이하인 애들만 후보
        candidates = {}
        for k in objects_sim.keys():
            perc = np.percentile(objects_sim[k], 10)
            sim = np.mean(np.array(objects_sim[k])[objects_sim[k] <= perc])
            if sim <= sim_threshold:
                candidates[k] = sim

        candidates_list.append(candidates)

    # candidates_list: [ {id1: sim1, id2: sim2}, {id1: sim3, id3: sim4}, ... ]
    # 앙상블로 최종 picked_id 결정
    picked_id = ensemble_similar_id(candidates_list)
    if picked_id is not None:
        # picked_id를 이용해서 BBox 색상 변화(초록색) 등 표시
        for t in tracked_bboxes:
            x1, y1, x2, y2, pid = t
            if pid == picked_id:
                bbox_color = (0, 255, 0)  # 초록색
            else:
                bbox_color = (0, 0, 255)  # 빨간색
            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)
    else:
        # picked_id를 못찾았거나 후보가 없으면 그냥 빨간색
        for t in tracked_bboxes:
            x1, y1, x2, y2, pid = t
            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # 최종 jpeg 인코딩
    ret_jpg, jpg_buffer = cv2.imencode('.jpg', processed_frame)
    return jpg_buffer.tobytes() if ret_jpg else None


# ==== MJPEG 프레임 로드 함수 ====
def load_video_frames(video_path):
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
        if current_frame_idx >= len(framesA):
            current_frame_idx = 0

        time.sleep(2.0 / fps)  # fps=30 가정 -> 약 2프레임씩 건너뛰는 설정 예시


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
    global switch_to_ai
    mode = request.args.get('to', 'original')
    switch_to_ai = (mode == 'ai')
    return f"Switched to {'AI Processed' if switch_to_ai else 'Original'} Video"


@app.route('/reset')
def reset_frame_index():
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


# ==== 메인 ====
if __name__ == '__main__':
    # 원본 영상 로드
    framesA = load_video_frames('test.mp4')
    app.run(host='0.0.0.0', port=4000, debug=True)
