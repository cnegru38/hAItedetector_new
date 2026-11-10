# app/main.py

from fastapi import FastAPI

from app.schemas.request import DetectRequest
from app.schemas.response import DetectResponse
from app.services.detector_service import DetectorService
from app.schemas.request import FlagRequest

app = FastAPI(
    title="hAItedetector",
    description="API de detecție spam, hate speech și hărțuire în limba română și engleză.",
    version="0.1.0",
)

# instanță unică - modelul și taxonomia sunt încărcate o singură dată la startup
detector = DetectorService()

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    return detector.detect(
        text=req.text,
        lang=req.lang,
        censor_hate=req.censor_hate,
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/flag")
def flag_message(req: FlagRequest) -> dict:
    import csv
    from pathlib import Path
    from datetime import datetime

    fp_path = Path(__file__).parent.parent / "data" / "raw" / "false_positives.csv"
    fp_path.parent.mkdir(exist_ok=True)

    # scriem header doar dacă fișierul e nou
    write_header = not fp_path.exists()

    with open(fp_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "text",
                "predicted_label", "correct_label"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            req.text,
            req.predicted_label,
            req.correct_label,
        ])

    return {"status": "saved"}