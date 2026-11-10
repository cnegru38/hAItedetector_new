
LABEL_COLS = ["label_spam", "label_hate_speech", "label_harassment", "label_obscene"]

# etichete principale (multi-label — un mesaj poate avea mai multe)

INTENTS = ["spam", "hate_speech", "harassment", "obscene"]

# Mapare bidirecțională index - eticheta
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(INTENTS)}
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(INTENTS)}

# subetichete spam (clasificare separată, activată doar dacă spam=true)
SPAM_SUBTYPES = ["junk", "scam", "phishing"]

ID2SPAM: dict[int, str] = {i: label for i, label in enumerate(SPAM_SUBTYPES)}
SPAM2ID: dict[str, int] = {label: i for i, label in enumerate(SPAM_SUBTYPES)}

# ținte hate speech (ce grup e vizat)
HATE_TARGETS = ["race", "religion", "gender", "orientation", "ethnicity", "other"]

# severitate hate speech
HATE_SEVERITY = ["low", "medium", "high"]

# intervale de scor - severitate
SEVERITY_BANDS: dict[str, tuple[float, float]] = {
    "low":    (0.50, 0.65),
    "medium": (0.65, 0.80),
    "high":   (0.80, 1.00),
}

def score_to_severity(score: float) -> str | None:
    for severity, (low, high) in SEVERITY_BANDS.items():
        if low <= score < high:
            return severity
    return None