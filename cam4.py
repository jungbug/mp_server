from flask import Flask, Response, request, jsonify, make_response
import base64, ssl, json
import cv2
import time
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image
from flask_cors import CORS

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

app = Flask(__name__)
CORS(app)
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

def similar_id(dictionary, value, threshold=1e-5):
    r=None
    last=np.inf
    for k in dictionary.keys():
        if np.abs(value-dictionary[k])<last and np.abs(value-dictionary[k])<threshold:
            last=np.abs(value-dictionary[k])
            r=k
            
    return r

# def detect_person(img, conf_threshold=0.5, ret_coord=False):
#     result=det_model.cpu()(img, verbose=False)
#     # print('1')
#     boxes_out=np.array([np.concatenate([np.array(box.xywh[0])[:2]-np.array(box.xywh[0])[2:]/2,np.array(box.xywh[0])[2:]]) for box in result[0].boxes if box.cls==0 and box.conf>=conf_threshold])
#     conf_out=np.array([np.array(box.conf) for box in result[0].boxes if box.cls==0 and box.conf>=conf_threshold])
#     # print('2')

#     boxes=np.array([np.array([box.xyxyn[0][1],box.xyxyn[0][0],box.xyxyn[0][3],box.xyxyn[0][2]]) for box in result[0].boxes if box.cls==0 and box.conf>=conf_threshold])
#     # print('3')
 
#     _img=tf.repeat(img[None], len(boxes), axis=0)
#     # print('4')

#     cropped_image=tf.image.crop_and_resize(_img, boxes, range(len(boxes)), (128,64))
    
#     if ret_coord:
#         # print('5')
#         return tf.image.crop_and_resize(_img, boxes, range(len(boxes)), (128,64)), boxes_out, conf_out
#     else:
#         # print('6')
#         return tf.image.crop_and_resize(_img, boxes, range(len(boxes)), (128,64))



def detect_person(img, conf_threshold=0.5, ret_coord=False):
    result = det_model(img)
    boxes_out = np.array([
        np.concatenate([
            np.array(box.xywh[0])[:2] - np.array(box.xywh[0])[2:] / 2,
            np.array(box.xywh[0])[2:]
        ]) for box in result[0].boxes if box.cls == 0 and box.conf >= conf_threshold
    ])
    conf_out = np.array([box.conf[0] for box in result[0].boxes if box.cls == 0 and box.conf >= conf_threshold])

    boxes = np.array([
        [box.xyxyn[0][1], box.xyxyn[0][0], box.xyxyn[0][3], box.xyxyn[0][2]]
        for box in result[0].boxes if box.cls == 0 and box.conf >= conf_threshold
    ])

    _img = tf.repeat(img[None], len(boxes), axis=0)

    cropped_image = tf.image.crop_and_resize(_img, boxes, range(len(boxes)), (128, 64))

    if ret_coord:
        return cropped_image, boxes_out, conf_out
    else:
        return cropped_image



# objects={} # id 별 이미지 저장 
# 딕셔너리(각 타겟 이미지) 안에 딕셔너리(id)
objects_sim={} # id 별 유사도 저장 
max_cosine_distance = 0.7
nn_budget = None

model_filename = 'mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

def ensemble_similar_id(candidates_dicts):
    """
    여러 target 후보(candidates) 딕셔너리를 받아
    1) 다수결(투표) 방식으로 가장 많이 등장한 ID 선정
    2) 만약 최다 득표가 동률이면, 각 target과의 평균 유사도(더 낮을수록 비슷함)가 가장 낮은 ID 선정
    :param candidates_dicts: 예) [ {id1: sim1, id2: sim2}, {id1: sim3, id3: sim4}, ... ]
    :return: 최종 picked_id
    """    
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


###################################
# "추가 후보" (교집합) 처리 함수
###################################
def get_additional_candidates(candidates_list, final_id):
    """
    1) 각 candidates_dict에서 final_id 제거
    2) 남은 ID들의 교집합(intersection)을 구함
       - 교집합이 비어있지 않으면 그 교집합을 반환
       - 교집합이 비어있으면 "남아 있는 모든 ID들의 합집합(union)"을 반환
    :param candidates_list: [ {id1: sim1, id2: sim2}, {id2: sim4, id3: sim5}, ... ]
    :param final_id: ensemble_similar_id로 결정된 최종 ID
    :return: 추가 후보 ID들의 set (예: {5, 7, 10} 등), 없으면 빈 set()
    """
    all_remaining_sets = []

    for cdict in candidates_list:
        # picked_id(final_id) 제외
        filtered_keys = [k for k in cdict.keys() if k != final_id]
        if len(filtered_keys) > 0:
            all_remaining_sets.append(set(filtered_keys))

    if not all_remaining_sets:
        # 남은 후보가 없다면 빈 set
        return set()

    # 교집합(intersection) 계산
    intersect_set = all_remaining_sets[0]
    for s in all_remaining_sets[1:]:
        intersect_set = intersect_set & s

    if len(intersect_set) > 0:
        return intersect_set
    else:
        # 교집합이 비어있으면 union 반환
        union_set = set()
        for s in all_remaining_sets:
            union_set = union_set | s
        return union_set

def make_target_list():
        
    target_list = []
    for i in sorted(os.listdir("./uploaded_images")):
        print(i)
        try:
            target = cv2.imread("uploaded_images/" + i)[...,:3]
            print("sibal")
            target_img=detect_person(target)[0]
            print(target_img.shape)
            target_list.append(target_img)
        except:
            print("except sibal ")
            continue
    return target_list

target_images = []
 
# ==== AI 모델로 프레임 처리 ====
def process_frame_with_ai(frame):
    global target_images

    processed_frame = frame.copy()

    sim_threshold = 0.085
    tracked=[]
    objects={} #프레임마다 찍히는 객체들의 이미지 저장
               #encoder로 객체 feature 뽑아내서 같은애인지 아닌지 확인하기 위해서
    
    crops,boxes,scores = detect_person(frame,ret_coord=True)

    names = ['person' for _ in range(len(boxes))]

    features=encoder(frame,boxes)
    
    # id별 feature 저장
    
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
    
    tracker.predict()
    tracker.update(detections)
    
    
    tracked_bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 5:
            continue 
        bbox = track.to_tlbr() # Get the corrected/predicted bounding box
        class_name = track.get_class() #Get the class name of particular object
        tracking_id = track.track_id # Get the ID for the particular track
        tracked_bboxes.append(bbox.tolist() + [tracking_id]) # Structure data, that we could use it with our draw_bbox function

    for t in tracked_bboxes:
        _t=np.array(t[:4])
        _t=np.concatenate([_t[:2][::-1],_t[2:][::-1]])
        cropped=tf.image.crop_and_resize(frame[None], np.array([_t[:4]])/(frame.shape[:2]+frame.shape[:2]), [0], (128,64))[0].numpy()
        
        if not t[4] in objects.keys(): objects[t[4]]=[]
        
        # objects.append(cropped)
        objects[t[4]].append(cropped)
        
    
    # 원래는 주어진 영상의 모든 프레임 정보를 저장하는 리스트였음
    # 이제는 실시간 영상의 모든 프레임 정보를 저장하는 역할을 함
    # 이전 프레임의 정보를 통해 계속 tracking를 해야하니까
    tracked.append(tracked_bboxes)
    
    if target_images is None:
        ret_jpg, jpg_buffer = cv2.imencode('.jpg', processed_frame) 
        return jpg_buffer.tobytes() if ret_jpg else None 

#------------------------------------------------------------------------------------------------------------------------------
    candidates_list = []
    for target in target_images:
        feature1=encoder(np.array(target),[[0,0,64,128]])
        for i in tuple(objects.keys()):
            obs=objects[i]
            ret=[]
            for o in obs:
                feature2=encoder(np.array(o),[[0,0,64,128]])
                ret.append(np.mean((feature1-feature2)**2)**0.5)
            if i not in objects_sim:
                objects_sim[i] = []
            objects_sim[i].extend(ret)
            
        candidates={}
        for k in objects_sim.keys():
            perc=np.percentile(objects_sim[k],10) 
            sim=np.mean(np.array(objects_sim[k])[objects_sim[k]<=perc])

            if sim<=sim_threshold:
                candidates[k]=sim
        candidates_list.append(candidates)
    
    # print(candidates_list)
    
    picked_id = ensemble_similar_id(candidates_list)
    
    # print(picked_id)
    
#------------------------------------------------------------------------------------------------
   # 추가 후보(교집합 혹은 전체) 계산
    additional_ids = get_additional_candidates(candidates_list, picked_id)
    # print("addition : ", additional_ids)
    # --- BBox 색상 표시 ---
    #   - picked_id(초록)
    #   - additional_ids(노랑)
    #   - 그 외(빨강)
    for t in tracked_bboxes:
        x1, y1, x2, y2, pid = t

        if pid == picked_id:
            bbox_color = (0, 255, 0)   # Green
        elif pid in additional_ids:
            bbox_color = (0, 255, 255) # Yellow
        else:
            bbox_color = (0, 0, 255)   # Red

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)
    ret_jpg, jpg_buffer = cv2.imencode('.jpg', frame)
    return jpg_buffer.tobytes() if ret_jpg else None 
            
#------------------------------------------------------------------------------------------------------------------------------

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

def reset_deepsort():
    global objects_sim, tracker
    objects_sim = {}
    tracker = Tracker(metric)
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch')
def switch():
    """원본 또는 AI 처리 영상으로 스위치"""
    global switch_to_ai
    mode = request.args.get('to', 'original')
    switch_to_ai = (mode == 'ai')
    reset_deepsort()
    return f"Switched to {'AI Processed' if switch_to_ai else 'Original'} Video"

@app.route('/reset')
def reset_frame_index():
    """현재 프레임 인덱스를 0으로 리셋"""
    global current_frame_idx
    current_frame_idx = 0
    return "Frame index reset to 0."
    
@app.route('/upload', methods=['POST'])
def process_images():
    global target_images
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
            image_path = os.path.join(save_dir, f"image_{idx + 1}.png")
            with open(image_path, "wb") as f:
                f.write(image_data)
                
        target_images = make_target_list()
        return jsonify({"success": True, "message": "Images processed successfully."})
    except Exception as e:
        print("Error processing images:", str(e))
        return jsonify({"success": False, "message": "Error processing images."}), 500



# ==== 초기화 단계 ====
if __name__ == '__main__':
    framesA = load_video_frames('test_cam4.mp4')  # 원본 영상 로드
    app.run(host='0.0.0.0', port=4004, debug=True)