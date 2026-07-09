import os, re, base64, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from paddlex import create_pipeline

app = FastAPI(title="PaddleOCR Classic Scanner")
executor = ThreadPoolExecutor(max_workers=4)

pipeline = create_pipeline(pipeline="OCR", layout_model="PP-DocLayoutV2")

LABEL_PRIORITY = {
    "document_title": 0,
    "title": 1,
    "section_title": 2,
    "subsection_title": 3,
    "plain_text": 4,
    "text": 4,
    "table": 5,
    "figure": 6,
    "equation": 7,
    "formula": 7,
    "header": 8,
    "footer": 8,
    "page_number": 9,
    "reference": 10,
    "abandon": 11,
}

SKIP_LABELS = {"header", "footer", "page_number", "abandon", "reference"}

def bbox_center(bbox):
    cx = sum(p[0] for p in bbox) / 4
    cy = sum(p[1] for p in bbox) / 4
    return cx, cy

def is_inside(inner_bbox, outer_bbox):
    cx, cy = bbox_center(inner_bbox)
    xs = [p[0] for p in outer_bbox]
    ys = [p[1] for p in outer_bbox]
    return min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys)

def make_markdown(ocr_items, layout_items):
    if not ocr_items:
        return ""

    if layout_items:
        layout_items.sort(key=lambda x: LABEL_PRIORITY.get(x["label"], 99))

        assigned = set()
        sections = []

        for layout in layout_items:
            label = layout["label"]
            if label in SKIP_LABELS:
                continue
            group = []
            for i, ocr in enumerate(ocr_items):
                if i not in assigned and is_inside(ocr["bbox"], layout["bbox"]):
                    group.append(ocr)
                    assigned.add(i)
            if not group:
                continue
            group.sort(key=lambda x: (round(min(p[1] for p in x["bbox"]) / 25), min(p[0] for p in x["bbox"])))
            text = " ".join(x["text"] for x in group)
            if label in ("document_title",):
                sections.append(f"# {text}")
            elif label in ("title", "section_title"):
                sections.append(f"## {text}")
            elif label in ("subsection_title",):
                sections.append(f"### {text}")
            elif label in ("table",):
                sections.append(f"\n[{text}]\n")
            elif label in ("figure",):
                sections.append(f"*[Image: {text}]*\n")
            else:
                sections.append(text)

        unassigned = [ocr_items[i] for i in range(len(ocr_items)) if i not in assigned]
    else:
        unassigned = ocr_items
        sections = []

    if unassigned:
        unassigned.sort(key=lambda x: (round(min(p[1] for p in x["bbox"]) / 25), min(p[0] for p in x["bbox"])))
        lines = []
        last_row = -1
        for item in unassigned:
            row = round(min(p[1] for p in item["bbox"]) / 25)
            if row != last_row:
                lines.append(item["text"])
            else:
                lines.append(item["text"] if not lines else " " + item["text"])
                if lines:
                    lines[-1] = ""
        sections.append(" ".join(lines))

    return "\n\n".join(s for s in sections if s).strip()


def predict_sync(image_path: str) -> dict:
    try:
        for res in pipeline.predict(image_path):
            ocr_items = []
            for ocr_res in getattr(res, "ocr_res", []) or []:
                ocr_items.append({
                    "text": ocr_res.text,
                    "bbox": ocr_res.bbox,
                    "score": ocr_res.score,
                })

            layout_items = []
            for layout_res in getattr(res, "layout_res", []) or []:
                layout_items.append({
                    "label": layout_res.label,
                    "bbox": layout_res.bbox,
                    "score": layout_res.score,
                })

            md = make_markdown(ocr_items, layout_items)
            return {"markdown": md, "status": "ok"}
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
