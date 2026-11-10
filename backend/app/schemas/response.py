from pydantic import BaseModel

class DetectResponse(BaseModel):
    # scoruri brute sigmoid per intenție
    scores: dict[str, float]

    # decizii binare per intenție
    decisions: dict[str, bool]

    # limba detectată automat sau forțată din UI
    detected_lang: str

    # subtipul de spam
    spam_subtype: str | None = None

    # ținta hate speech
    hate_target: str | None = None

    # severitatea hate speech: 'low' / 'medium' / 'high'
    hate_severity: str | None = None

    # textul cu cenzură
    censored_text: str | None = None