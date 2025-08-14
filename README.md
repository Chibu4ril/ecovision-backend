# ecovision-backend

On Local machine
docker build --build-arg RENDER=false -t ecovision-backend .
docker run -p 8000:8000 ecovision-backend
Uses environment.yml and creates the conda environment.

On Render:
Render sets RENDER=true automatically in builds.
