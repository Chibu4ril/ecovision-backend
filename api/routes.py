import logging
import subprocess
from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel
import json
import sys
import os

from .methods import fetch_training_sets, fetch_unprocessed_files, delete_uploaded_files

from fastapi.responses import JSONResponse



router = APIRouter()

class FileDeleteRequest(BaseModel):
    fileUrl : str

@router.get("/unprocessed_files")
async def get_uploaded_files():
    try:
        unprocessed = await fetch_unprocessed_files()
        return {"unprocessed": unprocessed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
async def match_the_file(request: FileDeleteRequest):
    try:
        delete_result = delete_uploaded_files(request.fileUrl)
        if not delete_result:
            raise HTTPException(status_code=404, detail="File not found or could not be deleted")

        return {"success": True, "message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class PredictionRequest(BaseModel):
    selectedFileUrl: str
    

@router.post("/runModels")
async def run_prediction(request: PredictionRequest):
    try:
        selected_file = request.selectedFileUrl

        script_path = os.path.abspath("eco-pipeline/main.py")
     
        process = subprocess.run(
            [sys.executable, script_path, selected_file],
            capture_output=True,
            text=True
        )

    
        log_file = "/tmp/script_output.log"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_contents = f.read()
            logging.info(f"Script Log: \n{log_contents}")  # Print to Render logs


        # If the script runs successfully
        # if process.returncode == 0:
        #     try:
        #         output_json = json.loads(process.stdout.strip())  # Parse JSON safely
        #         return JSONResponse(content=output_json)  # Return JSON response
        #     except json.JSONDecodeError:
        #         raise HTTPException(status_code=500, detail="Invalid JSON output from script")

        # If the script fails, return an error response
        raise HTTPException(status_code=500, detail=f"Script execution failed: {process.stderr.strip()}")

    except Exception as e:
        logging.error("Exception in runPrediction:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running prediction: {str(e)}")
    


@router.get("/health")
def health_check():
    return {"status": "ok"}