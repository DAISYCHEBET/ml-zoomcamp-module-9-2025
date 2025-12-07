# Straight vs Curly Hair Classifier – Module 9

This repository contains work for the ML Zoomcamp Module 9 homework, where we deployed a Straight vs Curly Hair Type classifier using a pre-trained ONNX model and Docker.

## What I Did

Downloaded Model Files

hair_classifier_v1.onnx and hair_classifier_v1.onnx.data from Alexey Grigorev’s dataset
.

Prepared Image for Inference

Wrote Python code to download, resize, and preprocess images:

from PIL import Image
from io import BytesIO
from urllib import request

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    img = Image.open(BytesIO(buffer))
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img.resize(target_size, Image.NEAREST)


Target image size used: 200x200.

Applied the Model Locally

Loaded the ONNX model.

Applied preprocessing to an example image:

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/16a55292-9c33-4c53-af5c-0651c03dac7b" />



Obtained model output for the example image.

Dockerization

Extended the AWS Lambda Python 3.13 Docker image.

Copied the model files into the container.

Installed required Python libraries in the image.

Added Lambda handler code for inference.

Successfully ran the Docker container locally:

docker run --rm -p 9000:8080 my-hairstyle-lambda lambda_function.lambda_handler


AWS Deployment (Optional)

Plan to push the Docker image to ECR.

Create a Lambda function using the ECR image.

Test the function and optionally expose it via API Gateway.

Technologies Used

Python 3.13

ONNX Runtime

Pillow (for image processing)

Docker

AWS Lambda (planned deployment)

How to Run Locally

Build Docker Image

docker build -t my-hairstyle-lambda .


Run Docker Container

docker run --rm -p 9000:8080 my-hairstyle-lambda lambda_function.lambda_handler


Test the Lambda Function Locally

Send a test request using curl (replace <image_url> with your image):

curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
     -d '{"url": "<image_url>"}'


Expected Output

The model will return a score indicating the likelihood of straight vs curly hair. Example:

{
  "logit": 0.0915,
  "probability": 0.523
}

Next Steps

Complete AWS deployment and test the Lambda function.

Optionally expose the Lambda function as an HTTP API using API Gateway.
