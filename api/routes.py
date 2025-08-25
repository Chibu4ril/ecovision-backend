import logging
import os
import numpy as np
import requests
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from .utils import delete_uploaded_files
from fastapi.responses import JSONResponse
from typing import Optional
from config.config import supabase



from .utils import fetch_unprocessed_files

from app.pipeline import run_inference
  # not anon key



router = APIRouter()

def to_python_type(val):
    if isinstance(val, (np.integer, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float64)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

def download_image_from_url(url, save_dir="temp"):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(url.split("?")[0])
    save_path = os.path.join(save_dir, filename)

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path


@router.get("/unprocessed_files")
async def get_uploaded_files():
    try:
        unprocessed = await fetch_unprocessed_files()
        return {"unprocessed": unprocessed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ImageDownloadRequest(BaseModel):
    image_url: str
@router.post("/process_image")
async def process_new_image(req: ImageDownloadRequest):
    try:
        final_dir = "./cached_inputs"
        os.makedirs(final_dir, exist_ok=True)

        # Download image
        local_path = download_image_from_url(req.image_url, final_dir)
        file_name = os.path.basename(local_path)
        result = {"status": "downloaded", "filename": file_name}
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictionRequest(BaseModel):
    filename: str
    

@router.post("/runModels")
async def run_prediction(request: PredictionRequest, authorization: str = Header(None) ):
    try:
        local_folder = "./cached_inputs"
        file_path = os.path.join(local_folder, request.filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing auth header")

        token = authorization.split(" ")[1]
        user_resp = supabase.auth.get_user(token)
        if not user_resp or not getattr(user_resp, "user", None):
            raise HTTPException(status_code=401, detail="Invalid token")

        user_id = user_resp.user.id


        result = run_inference(file_path, request.filename, user_id=user_id, supabase=supabase )

        cleaned_result = {k: to_python_type(v) for k, v in result.items()}

        return {
            "status": "success",
            "data": cleaned_result
        }

    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



class FileDeleteRequest(BaseModel):
    fileUrl : str

@router.delete("/delete")
async def match_the_file(request: FileDeleteRequest):
    try:
        delete_result = delete_uploaded_files(request.fileUrl)
        if not delete_result:
            raise HTTPException(status_code=404, detail="File not found or could not be deleted")

        return {"success": True, "message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    return {"status": "ok"}