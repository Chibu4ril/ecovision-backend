import logging
from typing import List, Dict, Optional
from config.config import supabase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_files_from_bucket(bucket_name: str, folder_path: str = "") -> List[Dict[str, str]]:
    try:
        response = supabase.storage.from_(bucket_name).list(folder_path)
        
        if response is None or (isinstance(response, dict) and "error" in response):
            logger.error(f"Error fetching files from bucket '{bucket_name}/{folder_path}': {response}")
            return []
        
        return [
            {
                "name": file["name"],
                "url": supabase.storage.from_(bucket_name).get_public_url(f"{folder_path}/{file['name']}")
            }
            for file in response
            if file["name"] != ".emptyFolderPlaceholder"
        ]
    except Exception as e:
        logger.error(f"Exception occurred while fetching files from bucket '{bucket_name}/{folder_path}': {e}")
        return []

async def fetch_unprocessed_files() -> List[Dict[str, str]]:
    return await fetch_files_from_bucket("tryit", "unprocessed")

async def fetch_training_sets() -> List[Dict[str, str]]:
    return await fetch_files_from_bucket("training_set")



   
async def delete_uploaded_files(file_url: str) -> Dict[str, str]:
    from urllib.parse import urlparse
    
    try:
        parsed_url = urlparse(file_url)
        file_path = parsed_url.path.lstrip("/")  

        response = await supabase.storage.from_("uploads").remove([file_path])

        logger.info(f"Supabase delete response for '{file_path}': {response}")
        
        # Check if there was an error in the response
        if response.get("error"):
            logger.error(f"Error deleting file '{file_path}': {response['error']}")
            return {"success": False, "message": f"Error deleting file: {response['error']}"}
        
        logger.info(f"File '{file_path}' deleted successfully")
        return {"success": True, "message": f"File '{file_path}' deleted successfully"}
        
    except Exception as e:
        logger.error(f"Exception occurred while deleting file '{file_url}': {e}")
        return {"success": False, "message": f"Exception occurred: {str(e)}"}
    

