import logging
import os
import tempfile
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .utils import download_image_from_supabase, delete_uploaded_files
from fastapi.responses import JSONResponse

from utils import run_inference, run_pipeline, fetch_unprocessed_files



router = APIRouter()


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

        return {"status": "downloaded", "filename": file_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictionRequest(BaseModel):
    filename: str

@router.post("/runModels")
async def run_prediction(request: PredictionRequest):
    try:
        local_folder = "./cached_inputs"
        file_path = os.path.join(local_folder, request.filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")

        result = run_pipeline(file_path)

        return {
            "status": "success",
            "data": result
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