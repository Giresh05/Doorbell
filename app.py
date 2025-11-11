from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
import pickle
import os
import time

app = Flask(__name__)

# Disable eager execution for TF1 compatibility
tf.disable_eager_execution()

# Global model variables
models_tuple = None

# --- Load Models Once ---
def load_all_models(modeldir, classifier_path, npy_dir):
    from detect_face import create_mtcnn

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        pnet, rnet, onet = create_mtcnn(sess, npy_dir)
        with tf.gfile.FastGFile(modeldir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        with open(classifier_path, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')
    return sess, pnet, rnet, onet, images_placeholder, embeddings, phase_train_placeholder, model, class_names

# --- Image Processing Function ---
def process_image(frame, models_tuple):
    (sess, pnet, rnet, onet, images_placeholder, embeddings, phase_train_placeholder, model, class_names) = models_tuple
    from detect_face import detect_face

    minsize = 30
    threshold = [0.7, 0.8, 0.8]
    factor = 0.709
    input_image_size = 160

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bounding_boxes, _ = detect_face(frame_rgb, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]

    results = []
    for i in range(nrof_faces):
        xmin, ymin, xmax, ymax = map(int, bounding_boxes[i, 0:4])
        cropped = frame_rgb[ymin:ymax, xmin:xmax, :]
        pil_image = Image.fromarray(cropped).resize((input_image_size, input_image_size))
        scaled = np.array(pil_image)
        scaled = (scaled - np.mean(scaled)) / np.std(scaled)
        scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        predictions = model.predict_proba(emb_array)
        best_idx = np.argmax(predictions, axis=1)
        best_prob = predictions[np.arange(len(best_idx)), best_idx][0]
        name = class_names[best_idx[0]] if best_prob > 0.95 else 'Unknown'
        results.append({"name": name, "probability": float(best_prob)})
    return results

# --- Flask Endpoints ---
@app.route('/detect', methods=['POST'])
def detect():
    global models_tuple
    if models_tuple is None:
        return jsonify({"error": "Models not loaded"}), 500

    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image provided"}), 400

    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image format"}), 400

    result = process_image(frame, models_tuple)
    return jsonify({"results": result})

@app.route('/init', methods=['POST'])
def init_models():
    global models_tuple
    data = request.json
    modeldir = data.get('modeldir', './model/20180402-114759.pb')
    classifier_path = data.get('classifier', './class/classifier.pkl')
    npy_dir = data.get('npy', './npy')

    if not os.path.exists(modeldir) or not os.path.exists(classifier_path):
        return jsonify({"error": "Model or classifier not found"}), 404

    models_tuple = load_all_models(modeldir, classifier_path, npy_dir)
    return jsonify({"status": "Models loaded successfully"})

@app.route('/')
def home():
    return jsonify({"message": "Face recognition API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
