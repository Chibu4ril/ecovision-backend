# ecovision-backend

# On Local machine with a running docker

docker build --build-arg RENDER=false -t ecovision-backend .
docker run -p 8000:8000 ecovision-backend
Uses environment.yml and creates the conda environment.

# To enable GPU support with PyTorch, install it manually

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# On Render:

Render sets RENDER=true automatically in builds.

# No docker

conda create -n ecovision-backend python=3.10
conda activate ecovision-backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
