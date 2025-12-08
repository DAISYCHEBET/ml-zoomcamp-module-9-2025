FROM agrigorev/model-2025-hairstyle:v1
WORKDIR /app
COPY lambda_function.py .
COPY test_image.jpeg .
COPY hair_classifier_v1.onnx .
COPY hair_classifier_v1.onnx.data .
RUN pip install --no-cache-dir pillow onnxruntime numpy
CMD ["python", "lambda_function.py"]
