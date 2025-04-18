FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pillow opencv-python-headless numpy tqdm pyyaml torch torchvision torchaudio
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
