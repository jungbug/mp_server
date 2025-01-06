from flask import Flask, request, jsonify, make_response, Response
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import load_model

#from keras.models import load_model
import os
import base64, ssl, json

import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

import matplotlib.pyplot as plt
from ultralytics import YOLO
import imageio.v3 as iio
from pathlib import Path

app = Flask(__name__)

sim_threshold=0.09

det_model = YOLO("yolov8n.pt")

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
    

def similar_id(dictionary, value, threshold=1e-5):
    r=None
    last=np.inf
    for k in dictionary.keys():
        if np.abs(value-dictionary[k])<last and np.abs(value-dictionary[k])<threshold:
            last=np.abs(value-dictionary[k])
            r=k
            
    return r

@app.route('/')
def index():
    return jsonify({'message': 'Flask server is running!'})


def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def build_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response




exp4coord = lambda x:tf.math.exp(-x**2/0.05).numpy()
exp4sim = lambda x:tf.math.exp(-(1.-x)**2/0.08).numpy()
exp4cnt = lambda x:tf.math.exp(-x**2/5000.).numpy()
act4alpha = lambda x:tf.math.tanh(x/100.).numpy()
target_image_alpha=0.2
from flask import Flask, Response, jsonify
import numpy as np
import cv2
import base64
import tensorflow as tf
from pathlib import Path

app = Flask(__name__)
@app.route('/stream', methods=['GET'])
def stream_video():
    _bytes = Path("./teest.mp4").read_bytes()

    frames = iio.imread(_bytes, index=None)
    img = iio.imread('./test.png')[..., :3]
    target = detect_person(img)[0]

    max_cosine_distance = 0.7
    nn_budget = None

    # Initialize deep sort object
    model_filename = 'mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    def generate_frames():
        tracked = []
        objects = {}
        max_object_size = 100  # Limit number of tracked objects
        fps = 10  # Target FPS for streaming

        for frame, i in zip(frames, range(len(frames))):
            if i % 5 != 0:  # Skip frames to reduce processing load
                continue

            start_time = time.time()

            crops, boxes, scores = detect_person(frame, ret_coord=True)
            names = ['person' for _ in range(len(boxes))]
            features = encoder(frame, boxes)

            detections = [Detection(bbox, score, class_name, feature)
                          for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

            tracker.predict()
            tracker.update(detections)

            # Obtain info from the tracks
            tracked_bboxes = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 5:
                    continue
                bbox = track.to_tlbr()
                tracking_id = track.track_id
                tracked_bboxes.append(bbox.tolist() + [tracking_id])

            # Update tracked objects
            for t in tracked_bboxes:
                _t = np.array(t[:4])
                _t = np.concatenate([_t[:2][::-1], _t[2:][::-1]])
                cropped = tf.image.crop_and_resize(frame[None], np.array([_t[:4]]) / (frame.shape[:2] + frame.shape[:2]), [0], (128, 64))[0].numpy()

                if t[4] not in objects:
                    objects[t[4]] = []
                objects[t[4]].append(cropped)

            # Limit the size of the objects dictionary
            if len(objects) > max_object_size:
                objects = {k: v for k, v in list(objects.items())[-max_object_size:]}

            # Track objects similarity
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

                if sim <= sim_threshold:
                    candidates[k] = sim

            if len(candidates) == 0:
                continue

            picked_id = similar_id(candidates, np.sort(list(candidates.values()))[0])
            candidates_id = [similar_id(candidates, i) for i in np.sort(list(candidates.values()))[1:]]

            # Draw bounding boxes on the frame
            _frame = frame.copy()
            for t in tracked_bboxes:
                x1, y1, x2, y2, pid = t

                bbox_color = (255, 0, 0)
                if pid == picked_id:
                    bbox_color = (0, 255, 0)
                elif pid in candidates_id:
                    bbox_color = (255, 255, 0)

                cv2.rectangle(_frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)

            # Convert BGR to RGB
            _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)

            # Convert frame to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', _frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # Enforce FPS
            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / fps:
                time.sleep((1 / fps) - elapsed_time)

            # Clear unused variables to prevent memory leaks
            del boxes, scores, names, features, detections, tracked_bboxes, objects_sim, candidates, picked_id, candidates_id

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
