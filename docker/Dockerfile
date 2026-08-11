FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rag-adapt-lab
COPY . .

RUN python3 -m pip install --break-system-packages -U pip \
    && python3 -m pip install --break-system-packages -e '.[rag,train,wandb]'

CMD ["raglab", "--help"]
