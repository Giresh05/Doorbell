import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
import argparse
import os
import pickle
import time
from six import string_types, iteritems
from flask import Flask, request, jsonify # --- NEW: For web server ---
import logging # --- NEW: For logging in Render ---

# --- FIX ---
# Disable Eager Execution to make TF 1.x code (like placeholders) compatible
tf.disable_eager_execution()
# --- END FIX ---

# --- Setup Flask App ---
app = Flask(__name__)
# Set up logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# --- Global variable to hold models ---
models_tuple = None
save_dir = "./received_images" # We can still save images on the server

# ----------------- facenet functions ----------------- #
def prewhiten(x):
# ... existing code ...
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def flip(image, random_flip):
# ... existing code ...
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
# ... existing code ...
    if len(img.shape) == 2:
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
    return img

def load_model(model):
# ... existing code ...
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        app.logger.info('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        app.logger.error("Model file not found: %s" % model_exp)
        raise ValueError("Model file not found: %s" % model_exp)

# ----------------- detect_face.py essentials ----------------- #
# ... existing code ...
def imresample(img, sz):
# ... existing code ...
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data

def nms(boxes, threshold, method):
# ... existing code ...
    if boxes.size==0:
        return np.empty((0,3))
    x1, y1, x2, y2, s = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
# ... existing code ...
    return pick[:counter]

def pad(total_boxes, w, h):
# ... existing code ...
    tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)
    tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
# ... existing code ...
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

def rerec(bboxA):
# ... existing code ...
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
# ... existing code ...
    bboxA[:,2:4] = bboxA[:,0:2]+np.transpose(np.tile(l,(2,1)))
    return bboxA

try:
    from detect_face import create_mtcnn, detect_face
except ImportError:
    print("Error: Could not import 'create_mtcnn' and 'detect_face'.")
    print("Please ensure 'detect_face.py' from the facenet repository is in the same directory.")
    # On a server, we must exit if this fails
    import sys
    sys.exit(1)


# ----------------- Combined detection pipeline ----------------- #

def load_all_models(modeldir, classifier_path, npy_dir):
# ... existing code ...
    """
    Loads all models (MTCNN, FaceNet, Classifier) once at the start.
    """
    app.logger.info("Loading models...")
    # Setup TF session
# ... existing code ...
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    
    with sess.as_default():
        # Load MTCNN
        app.logger.info("Loading MTCNN...")
        pnet, rnet, onet = create_mtcnn(sess, npy_dir)

        # Load FaceNet
        app.logger.info("Loading FaceNet model...")
        load_model(modeldir)
# ... existing code ...
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Load Classifier
        app.logger.info(f"Loading classifier from {classifier_path}...")
        with open(classifier_path, 'rb') as infile:
# ... existing code ...
            (model, class_names) = pickle.load(infile, encoding='latin1')
        app.logger.info(f"Loaded classifier with classes: {class_names}")

    app.logger.info("All models loaded successfully.")
    
# ... existing code ...
    return (sess, pnet, rnet, onet, 
            images_placeholder, embeddings, phase_train_placeholder, 
            model, class_names)


def process_image_frame(frame, models_tuple):
# ... existing code ...
    """
    Runs face detection on a single, already-loaded image frame.
    Returns a list of detected faces and their probabilities.
    """
    
    # --- NEW: List to store results ---
    detection_results = []
    
    try:
# ... existing code ...
        # Unpack all the models and TF assets
        (sess, pnet, rnet, onet, 
         images_placeholder, embeddings, phase_train_placeholder, 
         model, class_names) = models_tuple
# ... existing code ...
        # --- Constants from original function ---
        minsize = 30
# ... existing code ...
        threshold = [0.7, 0.8, 0.8]
        factor = 0.709
        image_size = 182
# ... existing code ...
        input_image_size = 160
        # --- End Constants ---

        if frame is None:
            app.logger.warning("Error: received empty image frame.")
            return [] # Return empty list

        if frame.ndim == 2:
# ... existing code ...
            frame = to_rgb(frame)
        
        # Ensure frame is in RGB format for facenet processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        app.logger.info("Detecting faces...")
        bounding_boxes, _ = detect_face(frame_rgb, minsize, pnet, rnet, onet, threshold, factor)
# ... existing code ...
        nrof_faces = bounding_boxes.shape[0]
        app.logger.info(f"Detected faces: {nrof_faces}")

        if nrof_faces > 0:
# ... existing code ...
            det = bounding_boxes[:, 0:4]
            for i in range(nrof_faces):
# ... existing code ...
                xmin, ymin, xmax, ymax = map(int, det[i])
                
                # Clamp bounding box to be within image dimensions
# ... existing code ...
                xmin = np.maximum(0, xmin)
                ymin = np.maximum(0, ymin)
                xmax = np.minimum(frame_rgb.shape[1], xmax)
                ymax = np.minimum(frame_rgb.shape[0], ymax)

                # Check if the bounding box is valid
                if xmin >= xmax or ymin >= ymax:
                    app.logger.warning("Skipping invalid bounding box.")
                    continue
                
                cropped = frame_rgb[ymin:ymax, xmin:xmax, :]
# ... existing code ...
                
                # Resize using PIL (as in original facenet align)
                pil_image = Image.fromarray(cropped)
# ... existing code ...
                pil_image_resized = pil_image.resize((input_image_size, input_image_size), Image.LANCZOS)
                scaled = np.array(pil_image_resized)

                # Prewhiten the image
# ... existing code ...
                scaled_prewhitened = prewhiten(scaled)
                scaled_reshape = scaled_prewhitened.reshape(-1, input_image_size, input_image_size, 3)
                
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
# ... existing code ...
                
                with sess.as_default():
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
# ... existing code ...
                    
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
# ... existing code ...
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                name = class_names[best_class_indices[0]]
                prob = best_class_probabilities[0]
                
                # Set label and color based on probability threshold
# ... existing code ...
                label = name if prob > 0.95 else "Unknown"
                
                # --- NEW: Add to results list instead of drawing ---
                app.logger.info(f"Detection: {label} ({prob:.2f})")
                detection_results.append({
                    "name": label,
                    "probability": float(prob), # Convert numpy float to python float for JSON
                    "box": [int(xmin), int(ymin), int(xmax), int(ymax)] # Convert numpy int
                })
                # --- END NEW ---

                # --- REMOVED: cv2.rectangle and cv2.putText ---
# ... existing code ...
        else:
            app.logger.info("No faces found.")

        # --- REMOVED: cv2.imshow and cv2.waitKey ---

    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        
    # --- NEW: Return the list of results ---
    return detection_results


# ----------------- NEW: Flask Web Server Endpoint ----------------- #
@app.route('/upload', methods=['POST'])
def handle_upload():
    app.logger.info("Received image upload request...")
    img_data = request.data
    
    if not img_data:
        app.logger.warning("No data received.")
        return jsonify({"error": "No data received"}), 400

    app.logger.info(f"Received {len(img_data)} bytes. Decoding image...")
    
    try:
        # Decode the image
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # --- SAVE IMAGE (optional) ---
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Get IP from request
                client_ip = request.remote_addr.replace('.', '_')
                filename = os.path.join(save_dir, f"img_{timestamp}_{client_ip}.jpg")
                cv2.imwrite(filename, frame)
                app.logger.info(f"Saved received image to {filename}")
            except Exception as e:
                app.logger.error(f"Error saving image: {e}")
            # --- END SAVE ---

            # Process the received image
            results = process_image_frame(frame, models_tuple)
            
            # Return results as JSON
            return jsonify(results)
            
        else:
            app.logger.error("Failed to decode image. Received data might not be a valid JPEG.")
            return jsonify({"error": "Failed to decode image"}), 400
            
    except Exception as e:
        app.logger.error(f"Error during image decode/process: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500


# ----------------- Run ----------------- #
def main():
    global models_tuple, save_dir # Make sure we're setting the global variables
    
    # Set up command-line argument parsing
# ... existing code ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, 
                        help='Path to the FaceNet model (.pb) file.', 
                        default="./model/20180402-114759.pb")
    parser.add_argument('--classifier_path', type=str, 
                        help='Path to the classifier (.pkl) file.', 
                        default="./class/classifier.pkl")
    parser.add_argument('--npy_dir', type=str, 
                        help='Path to the directory containing MTCNN .npy files.', 
                        default="./npy")
    # --- MODIFIED: Port is handled by Render, but we set a default ---
    parser.add_argument('--port', type=int,
                        help='Port to listen on.',
                        # Render sets the PORT env var, default to 10000 for local testing
                        default=int(os.environ.get("PORT", 10000))) 
    parser.add_argument('--save_dir', type=str, 
                        help='Directory to save received images.', 
                        default="./received_images")
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    save_dir = args.save_dir
# ... existing code ...
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        app.logger.info(f"Created directory: {save_dir}")

    # 1. Load models ONCE
    # This must be done *before* the first request
    app.logger.info("Starting server, loading models...")
    models_tuple = load_all_models(args.modeldir, args.classifier_path, args.npy_dir)
    app.logger.info("Models loaded, server is ready to accept connections.")

    # 2. Start the Flask server
    # Host '0.0.0.0' is crucial to be reachable in a container
    app.run(host='0.0.0.0', port=args.port)


if __name__ == "__main__":
    main()
