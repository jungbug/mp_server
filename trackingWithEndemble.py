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

def similar_id(dictionary, value, threshold=1e-5):
    r=None
    last=np.inf
    for k in dictionary.keys():
        if np.abs(value-dictionary[k])<last and np.abs(value-dictionary[k])<threshold:
            last=np.abs(value-dictionary[k])
            r=k
            
    return r

def detect_person(img, conf_threshold=0.5, ret_coord=False):
    result=det_model.cpu()(img, verbose=False)
    
    boxes_out=np.array([np.concatenate([np.array(box.xywh[0])[:2]-np.array(box.xywh[0])[2:]/2,np.array(box.xywh[0])[2:]]) for box in result[0].boxes if box.cls==0 and box.conf>=conf_threshold])
    conf_out=np.array([np.array(box.conf) for box in result[0].boxes if box.cls==0 and box.conf>=conf_threshold])
    
    boxes=np.array([np.array([box.xyxyn[0][1],box.xyxyn[0][0],box.xyxyn[0][3],box.xyxyn[0][2]]) for box in result[0].boxes if box.cls==0 and box.conf>=conf_threshold])
    
    _img=tf.repeat(img[None], len(boxes), axis=0)

    cropped_image=tf.image.crop_and_resize(_img, boxes, range(len(boxes)), (128,64))
    
    if ret_coord:
        return tf.image.crop_and_resize(_img, boxes, range(len(boxes)), (128,64)), boxes_out, conf_out
    else:
        return tf.image.crop_and_resize(_img, boxes, range(len(boxes)), (128,64))

def ensemble_similar_id(candidates_list, method="voting"):
    """
    앙상블 기법으로 최종 picked_id 결정
    :param candidates_dicts: 각 target별 candidates 딕셔너리 리스트
    :param method: 'voting' 또는 'mean' 지원
    :return: 최종 picked_id
    """
    import numpy as np

    all_ids = []
    if candidates_list:
        for candidates in candidates_list:
            print(candidates)
            if candidates:
                picked_id = similar_id(candidates, np.sort(list(candidates.values()))[0])
                print("in Ensemble : ", picked_id)
                all_ids.append(picked_id)

        if method == "voting":
            # 다수결로 ID 결정 (원핫 인코딩 + argmax 사용)
            unique_ids = list(set(all_ids))  # 고유 ID 추출
            id_to_index = {uid: idx for idx, uid in enumerate(unique_ids)}  # ID를 인덱스로 매핑

            # 원핫 벡터 초기화
            one_hot_counts = np.zeros(len(unique_ids), dtype=int)

            # 원핫 벡터에 등장 횟수 누적
            for pid in all_ids:
                one_hot_counts[id_to_index[pid]] += 1

            # argmax로 가장 많이 등장한 ID 선택
            final_id = unique_ids[np.argmax(one_hot_counts)]
        elif method == "mean":
            # 평균 유사도가 가장 낮은 ID
            mean_scores = {pid: np.mean([cand.get(pid, float("inf")) for cand in candidates_list]) for pid in all_ids}
            final_id = min(mean_scores, key=mean_scores.get)
        else:
            raise ValueError("지원하지 않는 method: 'voting' 또는 'mean'을 선택하세요.")

        return final_id
    return False

target_image_name='test.png'
img=cv2.imread(target_image_name)[...,:3]
target=detect_person(img)[0]


# objects={} # id 별 이미지 저장 
objects_sim={} # id 별 유사도 저장 
# idx = 0 # 몇번째 프레임인지 확인하기 위한 변수

max_cosine_distance = 0.7
nn_budget = None

model_filename = 'mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


# 타겟 이미지의 feature
feature1=encoder(np.array(target),[[0,0,64,128]])



# ==== AI 모델로 프레임 처리 ====
def process_frame_with_ai(frame):
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

#------------------------------------------------------------------------------------------------------------------------------
    # 업로드된 이미지 파일 경로 가져오기
    target_image_dir = "uploaded_images"  # 디렉터리 이름
    target_image_paths = [os.path.join(target_image_dir, file) for file in os.listdir(target_image_dir) if file.endswith(('.jpg', '.png'))]

# 이미지 로드
    try:
        target_images = [cv2.imread(image_path)[..., :3] for image_path in target_image_paths if cv2.imread(image_path) is not None]
        if not target_images:
            raise ValueError("No valid images found in the directory.")
    except Exception as e:
        print(f"Error loading images: {e}")
    candidates_list = []
# objects_sim에 각 프레임마다 찍힌 객체들과 타겟 객체의 유사도를 추가 저장 
    for target in target_images:
    # 각 target에 대해 유사도 계산
        objects_sim = {}
        for i in tuple(objects.keys()):
            obs = objects[i]
            ret = []
            for o in obs:
                feature1 = encoder(np.array(target), [[0, 0, 64, 128]])
                feature2 = encoder(np.array(o), [[0, 0, 64, 128]])
                ret.append(np.mean((feature1 - feature2) ** 2) ** 0.5)
            objects_sim[i] = ret

        candidates = {}
        for k in objects_sim.keys():
            perc = np.percentile(objects_sim[k], 10)
            sim = np.mean(np.array(objects_sim[k])[objects_sim[k] <= perc])
    
            #threshold수정(실험)
            if sim <= sim_threshold:
                candidates[k] = sim

        candidates_list.append(candidates)
#------------------------------------------------------------------------------------------------------------------------------
    print(candidates)
    print(candidates_list)
    # candidates에서 제일 비슷한 애의 id 저장
    if candidates_list is not None:
        print("in candidates")
        picked_id = ensemble_similar_id(candidates_list, method="voting")
        print(picked_id)
        if picked_id is None:
            ret_jpg, jpg_buffer = cv2.imencode('.jpg', processed_frame)
            return jpg_buffer.tobytes() if ret_jpg else None
    # candidates에서 비슷하다고 판단된 애들(노란박스)
        candidates_id=[similar_id(candidates,i) for i in np.sort(list(candidates.values()))[1:]] # 두번째 값부터는 비슷한 걸로 분류

    #------------------------------------------------------------------------------------------------------------------------------

    # 마지막 타겟 인식하면 초록색, 비슷하면 노란색, 아니면 빨간색 치는 부분
    # tracked는 이전 프레임의 정보가 계속 저장된 리스트
        for t in tracked[0]:
            x1, y1, x2, y2, pid = t

            bbox_color=(0,0,255)
            
            if pid==picked_id: # 
                bbox_color=(0,255,0)
                print(pid)
            elif pid in candidates_id:
                bbox_color=(0,255,255)
                print(pid)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)
        
        #idx+=1
        
        ret_jpg, jpg_buffer = cv2.imencode('.jpg', frame)
        return jpg_buffer.tobytes() if ret_jpg else None
    
    ret_jpg, jpg_buffer = cv2.imencode('.jpg', processed_frame)
    return jpg_buffer.tobytes() if ret_jpg else None

# ==== 실제 스트리밍(제너레이터) ====
def generate_mjpeg():
    global current_frame_idx, switch_to_ai

    while True:
        if not switch_to_ai:
            # 원본 영상 프레임 송출
            frame_data = framesA[current_frame_idx]
            tracker = Tracker(metric)
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
