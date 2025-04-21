FROM python:3.10-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ARG PYTORCH_URL=https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
      fastapi "uvicorn[standard]" python-multipart pillow \
      opencv-python-headless numpy tqdm pyyaml \
      torch torchvision torchaudio \
      --extra-index-url $PYTORCH_URL && \
    rm -rf /root/.cache/pip


CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
