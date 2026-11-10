def censor_hash(text: str) -> str:
    return "".join("#" if not ch.isspace() else ch for ch in text)