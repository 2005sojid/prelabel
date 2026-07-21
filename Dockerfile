# Prelabel — CPU image.
#
# The awkward part of installing this by hand is the native stack: OpenCV needs
# system libraries, and torch pulls a CUDA runtime by default even when you only
# want CPU. Both are pinned down here so `docker run` works the first time.
#
# For GPU, build with:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t prelabel:gpu .
# and run with `--gpus all`.

FROM python:3.12-slim AS base

# CPU wheels by default: a CUDA torch is ~2.5 GB and useless without a GPU.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Ultralytics otherwise tries to pip install things at inference time.
    YOLO_AUTOINSTALL=False \
    # It also wants a writable config dir; give it one inside the image.
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    PL_HOST=0.0.0.0 \
    PL_STORAGE_DIR=/data/storage \
    PL_DATA_ROOTS=/data/images

# libGL and libglib are what opencv-python links against; ffmpeg supplies the
# H.264 encoder, without which rendered video will not play in a browser.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source change does not re-resolve the whole stack.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple torch torchvision \
    && pip install -r requirements.txt

COPY prelabel/ ./prelabel/
COPY frontend/ ./frontend/
COPY run.py pyproject.toml README.md LICENSE ./

# Run unprivileged: this process reads user-supplied model files, and a `.pt` is
# a pickle. Root is the wrong identity for that.
RUN useradd --create-home --uid 10001 prelabel \
    && mkdir -p /data/storage /data/images \
    && chown -R prelabel:prelabel /app /data /tmp/ultralytics 2>/dev/null || true
USER prelabel

VOLUME ["/data/storage", "/data/images"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "run.py"]
