# run with `uvicorn server:app --host 0.0.0.0 --port 8000`

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from extract_actions import parse_id

app = FastAPI()

@app.get("/download-result")
def download_result(id: str = "minecraft"):
    parse_id(id)
    return FileResponse(f"{id}.jsonl")

