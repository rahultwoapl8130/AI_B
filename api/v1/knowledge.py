from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import os
import tempfile
from rag.ingestion import process_single_file

router = APIRouter()

@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Receives a PDF/DOCX file from the frontend, saves it temporarily, 
    and processes it into the Vector Store in the background.
    """
    if not file.filename.endswith(('.pdf', '.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported currently.")
        
    try:
        # Create a temp file to save the upload
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Send to background task so the API responds quickly to the frontend
        background_tasks.add_task(process_single_file, file_path, file.filename)
        
        return {"message": "File uploaded successfully! Processing into AI Knowledge Base..."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
