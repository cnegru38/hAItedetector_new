from pydantic import BaseModel, Field

class DetectRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Textul de analizat.",
    )
    lang: str | None = Field(
        default=None,
        pattern="^(ro|en)$",
        description="Limba forțată din UI ('ro' sau 'en'). "
                    "Dacă None, limba e detectată automat.",
    )
    censor_hate: bool = Field(
        default=False,
        description="Dacă True și hate_speech=True, răspunsul include "
                    "textul cenzurat cu '#'.",
    )
    
class FlagRequest(BaseModel):
    text:            str
    predicted_label: str
    correct_label:   str