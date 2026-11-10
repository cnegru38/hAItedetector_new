from pathlib import Path

from app.core.taxonomy_loader import load_taxonomy
from app.model.inference_distilbert import run_inference, InferenceResult
from app.model.preprocess import preprocess
from app.schemas.response import DetectResponse
from app.services.censorship import censor_hash

_TAXONOMY_PATH = Path(__file__).parent.parent.parent / "taxonomy.yaml"


class DetectorService:
    def __init__(self):
        
        # taxonomy.yaml e citit o singură dată per rulare a serverului.
        self.taxonomy = load_taxonomy(_TAXONOMY_PATH)

    def detect(self, text: str, lang: str | None = None, censor_hate: bool = False) -> DetectResponse:

        # etapa 1: preprocesare
        clean_text, detected_lang = preprocess(text, lang=lang)

        # etapa 2: inferență
        result: InferenceResult = run_inference(
            text          = clean_text,
            thresholds    = self.taxonomy.thresholds,
            thresholds_ro = self.taxonomy.thresholds_ro,
            lang          = detected_lang,
        )

        # etapa 3: cenzurare condiționată
        # cenzurăm textul ORIGINAL (nu cel preprocesаt)
        censored_text = None
        if censor_hate and result.decisions.get("hate_speech"):
            censored_text = censor_hash(text)

        return DetectResponse(
            scores=result.scores,
            decisions=result.decisions,
            detected_lang=detected_lang,
            spam_subtype=result.spam_subtype,
            hate_target=result.hate_target,
            hate_severity=result.hate_severity,
            censored_text=censored_text,
        )