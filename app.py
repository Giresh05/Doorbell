from __future__ import absolute_import, division, print_function
import os
import pickle
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from flask import Flask, request, jsonify
from six import string_types, iteritems

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Model Paths ---
MODEL_DIR = "./model/20180402-114759.pb"
CLASSIFIER_PATH = "./class/classifier.pkl"
NPY_DIR = "./npy"

# --- Global Variables for Models ---
sess = None
pnet = None
rnet = None
onet = None
images_placeholder = None
embeddings = None
phase_train_placeholder = None
classifier_model = None
class_names = None
embedding_size = None

# ----------------- facenet functions ----------------- #
# (These are identical to your provided code)
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def to_rgb(img):
    if len(img.shape) == 2:
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
    return img

def load_model_graph(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        raise ValueError("Model file not found: %s" % model_exp)

# ----------------- detect_face.py essentials ----------------- #
# (Using a minimal import, assuming detect_face.py is available)
# If detect_face.py is not in the same directory, 
# you'll need to copy its functions (create_mtcnn, detect_face) here.
try:
    from detect_face import create_mtcnn, detect_face
except ImportError:
    print("Warning: 'detect_face' module not found.")
    print("Please ensure detect_face.py is in the same directory.")
    # You would paste the full detect_face.py content here as a fallback
    def create_mtcnn(sess, model_path):
        print("Error: detect_face.py not found. MTCNN cannot be loaded.")
        return None, None, None
    def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
        print("Error: detect_face.py not found. Face detection will not work.")
        return np.empty((0,5)), np.empty((0,10))


# --- Initialization Function ---
def initialize_models():
    """
    Loads all models into global variables ONCE at startup.
    """
    global sess, pnet, rnet, onet, images_placeholder, embeddings
    global phase_train_placeholder, classifier_model, class_names, embedding_size
    
    print("Initializing TensorFlow session and loading models...")
    tf.disable_v2_behavior()
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        with sess.as_default():
            # Load MTCNN
            print("Loading MTCNN...")
            pnet, rnet, onet = create_mtcnn(sess, NPY_DIR)

            # Load FaceNet
            print("Loading FaceNet model...")
            load_model_graph(MODEL_DIR)

            # Get FaceNet tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Load Classifier
            print("Loading classifier...")
            with open(CLASSIFIER_PATH, 'rb') as infile:
                (classifier_model, class_names) = pickle.load(infile, encoding='latin1')
            print("Loaded classifier with classes:", class_names)
    print("Models initialized successfully.")

# --- Image Processing Function ---
def process_image(frame):
    """
    Detects and recognizes faces in a single image frame.
    Uses the pre-loaded global models.
    """
    global sess, pnet, rnet, onet, images_placeholder, embeddings
    global phase_train_placeholder, classifier_model, class_names, embedding_size

    if frame is None:
        return []

    if frame.ndim == 2:
        frame = to_rgb(frame)

    # MTCNN parameters
    minsize = 30
    threshold = [0.7, 0.8, 0.8]
    factor = 0.709
    
    # FaceNet parameters
    input_image_size = 160

    print("Detecting faces...")
    bounding_boxes, _ = detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    print("Detected faces:", nrof_faces)

    results = []

    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        
        for i in range(nrof_faces):
            xmin, ymin, xmax, ymax = map(int, det[i])
            
            # Basic boundary check
            if xmin <= 0 or ymin <= 0 or xmax >= frame.shape[1] or ymax >= frame.shape[0]:
                print(f"Skipping face {i} due to boundary issue.")
                continue

            try:
                # Crop and resize
                cropped = frame[ymin:ymax, xmin:xmax, :]
                scaled_img = Image.fromarray(cropped).resize((input_image_size, input_image_size), Image.LANCZOS)
                scaled = np.array(scaled_img)
                scaled = prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)

                # Get embedding
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                
                # Run classifier
                predictions = classifier_model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                name = class_names[best_class_indices[0]]
                prob = float(best_class_probabilities[0]) # Convert to standard float for JSON
                
                # --- This is your "Decision Logic" ---
                recognition_threshold = 0.97 # You can adjust this
                if prob > recognition_threshold:
                    label = name
                else:
                    label = "Unknown"

                results.append({
                    "name": label,
                    "probability": prob,
                    "bounding_box": {
                        "xmin": int(xmin),
                        "ymin": int(ymin),
                        "xmax": int(xmax),
                        "ymax": int(ymax)
                    }
                })

            except Exception as e:
                print(f"Error processing face {i}: {e}")

    return results

# --- Flask API Endpoint ---
@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    API endpoint to receive an image and return recognition results.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file part in request."}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Read image from request
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
             return jsonify({"error": "Could not decode image."}), 400

        # Process the image using our function
        recognition_results = process_image(frame)
        
        # *** LIVENESS DETECTION WOULD GO HERE ***
        # We would loop through recognition_results.
        # For each "Unknown" or known person, we'd run a liveness check.
        # e.g., liveness = check_liveness(frame, result['bounding_box'])
        # Then add 'is_live': liveness to the result dictionary.
        
        print(f"Returning results: {recognition_results}")
        
        return jsonify({
            "status": "success",
            "face_count": len(recognition_results),
            "faces": recognition_results
        })

    except Exception as e:
        print(f"Error in /api/recognize: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

# --- Run the App ---
if __name__ == "__main__":
    initialize_models() # Load models once
    # Run Flask app
    # Use host='0.0.0.0' to make it accessible on your network
    # (Render will set its own host and port)
    app.run(host="0.0.0.0", port=5000, debug=False)
