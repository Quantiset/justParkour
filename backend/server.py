from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/download-result")
def download_result():
    return FileResponse("predicted_actions.jsonl")  # processed file

