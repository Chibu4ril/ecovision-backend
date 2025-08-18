# ====================================================
# Base Image
# ====================================================
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:$PATH"


# ====================================================
# System dependencies
# ====================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ====================================================
# Copy dependency files
# ====================================================
COPY environment.yml .
COPY requirements.txt .

# ====================================================
# Build argument for conditional install
# ====================================================
ARG RENDER=false

# ====================================================
# Conditional install
# ====================================================
RUN if [ "$RENDER" = "true" ]; then \
        echo "Render build: Installing from requirements.txt" && \
        pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt ; \
    else \
        echo "Local build: Installing Miniconda & using environment.yml" && \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
        bash miniconda.sh -b -p /opt/conda && \
        rm miniconda.sh && \
        /opt/conda/bin/conda config --set always_yes yes && \
        /opt/conda/bin/conda config --set changeps1 no && \
        /opt/conda/bin/conda config --set channel_priority strict && \
        /opt/conda/bin/conda config --set auto_activate_base false && \
        # Accept TOS for channels explicitly \
        /opt/conda/bin/conda tos accept --channel https://repo.anaconda.com/pkgs/main --override-channels || true && \
        /opt/conda/bin/conda tos accept --channel https://repo.anaconda.com/pkgs/r --override-channels || true && \
        /opt/conda/bin/conda env create -f environment.yml && \
        echo "source activate inference-env" > ~/.bashrc ; \
    fi

# Make Conda environment available in subsequent RUN commands
SHELL ["conda", "run", "-n", "inference-env", "/bin/bash", "-c"]

# ====================================================
# Copy application code
# ====================================================
COPY . .

# ====================================================
# Expose FastAPI port
# ====================================================
EXPOSE 8000

# ====================================================
# Start FastAPI via Uvicorn (direct call, no conda run)
# ====================================================
CMD ["/opt/conda/envs/inference-env/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
