# MLflow tracking server + model registry. Also used by the optional
# `mlflow-serve` profile (mlflow models serve), hence onnx + onnxruntime.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir "mlflow==3.17.2" "onnx>=1.17" "onnxruntime>=1.20,<2"
WORKDIR /mlflow
EXPOSE 5000
