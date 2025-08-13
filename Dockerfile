# ====================================================
# Base Image — CUDA Runtime + Miniconda (Python 3.10)
# ====================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ====================================================
# Working Directory & ENV
# ====================================================
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ====================================================
# Install System Dependencies
# ====================================================
RUN apt-get update && apt-get install -y \
    git curl wget build-essential libgl1 libglib2.0-0 \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ====================================================
# Install Miniconda
# ====================================================
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# ====================================================
# Copy Conda Environment File & Install Dependencies
# ====================================================
COPY environment.yml .

RUN conda env create -f environment.yml && conda clean -a

# Activate conda environment by default in subsequent RUN commands and container
SHELL ["conda", "run", "-n", "inference-env", "/bin/bash", "-c"]

# ====================================================
# Copy Application Code
# ====================================================
COPY . .

# ====================================================
# Expose FastAPI Port
# ====================================================
EXPOSE 8000

# ====================================================
# Start FastAPI via Uvicorn inside Conda Env
# ====================================================
CMD ["conda", "run", "--no-capture-output", "-n", "inference-env", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
