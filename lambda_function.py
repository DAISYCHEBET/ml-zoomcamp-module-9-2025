from PIL import Image, ImageOps
import numpy as np
import onnxruntime as ort

# Path to test image inside container (will mount or copy during docker build)
TEST_IMAGE_PATH = "test_image.jpeg"
MODEL_PATH = "hair_classifier_v1.onnx"

# Preprocessing function (same as Module 8)
def preprocess_image(img_path, target_size=(200, 200)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0

    # Module 8 normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2,0,1))  # CHW
    arr = np.expand_dims(arr, 0).astype(np.float32)  # batch dim
    return arr

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def lambda_handler():
    x = preprocess_image(TEST_IMAGE_PATH)
    pred = session.run([output_name], {input_name: x})
    logit = float(pred[0].ravel()[0])
    sigmoid = 1 / (1 + np.exp(-logit))
    print("Raw output (logit):", logit)
    print("Sigmoid probability:", sigmoid)

if __name__ == "__main__":
    lambda_handler()
