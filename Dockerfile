# ====================================================
# 1️⃣ Base Image — CPU+GPU Compatible
# ====================================================
FROM python:3.10-slim

# Optional: If building on GPU hosts, install NVIDIA runtime deps
# (Safe for CPU too — won't be used if no GPU present)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ====================================================
# 2️⃣ Workdir & Env Vars
# ====================================================
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ====================================================
# 3️⃣ Install Torch (CPU or GPU at runtime)
# We'll set up a script to choose the right wheel
# ====================================================
COPY requirements.txt requirements.txt
COPY trained_models/ /app/trained_models/


# Install pip-tools for clean dependency management
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU by default
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other deps
RUN pip install --no-cache-dir -r requirements.txt

# ====================================================
# 4️⃣ Copy App Code
# ====================================================
COPY . .

# ====================================================
# 5️⃣ Runtime Command
# Use Uvicorn to start FastAPI
# ====================================================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
