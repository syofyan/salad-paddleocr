import os, re, base64, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from paddlex import create_pipeline

app = FastAPI(title="PaddleOCR-VL Scanner")
executor = ThreadPoolExecutor(max_workers=4)

pipeline = create_pipeline(pipeline="PaddleOCR-VL")

def predict_sync(image_path: str) -> dict:
    try:
        for res in pipeline.predict(image_path):
            md = getattr(res, "markdown", None) or ""
            if isinstance(md, dict):
                md = "\n".join(v for v in md.values() if isinstance(v, str))
            return {"markdown": md.strip(), "status": "ok"}
        return {"markdown": "", "status": "empty"}
    except Exception as e:
        return {"markdown": "", "status": "error", "error": str(e)}

async def predict_async(image_path: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, predict_sync, image_path)

@app.post("/scan")
async def scan(file: UploadFile | None = None, input: str | None = Form(None)):
    if file:
        suffix = Path(file.filename).suffix or ".png"
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, await file.read())
            os.close(fd)
            return await predict_async(path)
        finally:
            os.unlink(path)

    if input:
        if input.startswith("data:image"):
            m = re.match(r"data:image/\w+;base64,(.+)", input)
            if not m:
                raise HTTPException(400, "Invalid base64 image data")
            fd, path = tempfile.mkstemp(suffix=".png")
            try:
                os.write(fd, base64.b64decode(m.group(1)))
                os.close(fd)
                return await predict_async(path)
            finally:
                os.unlink(path)
        else:
            return await predict_async(input)

    raise HTTPException(400, "Provide 'file' (multipart) or 'input' (URL/base64 string)")

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="::", port=8080)
