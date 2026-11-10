import re
import unicodedata
from pathlib import Path

import emoji
import yaml
from lingua import Language, LanguageDetectorBuilder

# încărcare configurație emoji
def load_emoji_config(config_path: str | Path) -> tuple[dict[str, str], set[str]]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    high_risk: dict[str, str] = data.get("high_risk", {}) or {}
    ambiguous_raw: dict = data.get("ambiguous", {}) or {}

    # cheile din secțiunea `ambiguous` sunt emoji-urile de ignorat
    ambiguous: set[str] = set(ambiguous_raw.keys())

    return high_risk, ambiguous

# preprocesare emoji

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticoane faciale
    "\U0001F300-\U0001F5FF"   # simboluri & pictograme
    "\U0001F680-\U0001F6FF"   # transport & hărți
    "\U0001F1E0-\U0001F1FF"   # steaguri
    "\U00002700-\U000027BF"   # Dingbats
    "\U0001F900-\U0001F9FF"   # simboluri suplimentare
    "\U00002600-\U000026FF"   # simboluri diverse
    "\U0001FA00-\U0001FA9F"   # simboluri extinse (topor, dinamită etc.)
    "]+",
    flags=re.UNICODE,
)

def replace_emoji_with_text(
    text: str,
    high_risk: dict[str, str],
    ambiguous: set[str],
    lang: str = "en",
) -> str:
    def replace_match(match: re.Match) -> str:
        char = match.group()

        # regula 1: risc clar - text explicit din config
        if char in high_risk:
            return f" {high_risk[char]} "

        # regula 2: ambiguu - ignorat complet
        if char in ambiguous:
            return " "

        # regula 3: orice alt emoji - denumire din baza de date Unicode
        name = (
            emoji.EMOJI_DATA.get(char, {}).get(lang)
            or emoji.EMOJI_DATA.get(char, {}).get("en")
            or ""
        )
        name = name.strip(":").replace("_", " ")
        return f" {name} " if name else " "

    return _EMOJI_RE.sub(replace_match, text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# detecție limbă
_DETECTOR = (
    LanguageDetectorBuilder
    .from_languages(Language.ROMANIAN, Language.ENGLISH)
    .with_preloaded_language_models()
    .build()
)

def detect_language(text: str) -> str:
    detected = _DETECTOR.detect_language_of(text)

    if detected == Language.ROMANIAN:
        return "ro"
    return "en"  # fallback implicit

# Normalizare diacritice
_RO_DIACRITICS: dict[str, str] = {
    "ș": "s", "Ș": "S",   # s cu virgulă dedesubt
    "ț": "t", "Ț": "T",   # t cu virgulă dedesubt
    "ş": "s", "Ş": "S",   # s cu cedilă (encoding frecvent)
    "ţ": "t", "Ţ": "T",   # t cu cedilă (encoding frecvent)
    "ă": "a", "Ă": "A",
    "â": "a", "Â": "A",
    "î": "i", "Î": "I",
}

_RO_DIACRITICS_TABLE = str.maketrans(_RO_DIACRITICS)

def strip_diacritics(text: str) -> str:
    return text.translate(_RO_DIACRITICS_TABLE)

def normalize_punctuation(text: str) -> str:

    # reducem punctuația repetată la un singur caracter
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    # eliminăm punctul final singur
    text = re.sub(r'\.$', '', text)
    return text

# pipeline principal
_DEFAULT_CONFIG = Path(__file__).parent.parent / "emoji_config.yaml"

def preprocess(
    text: str,
    lang: str | None = None,
    config_path: str | Path = _DEFAULT_CONFIG,
) -> tuple[str, str]:
    
    # etapa 1: detecție limbă
    detected_lang = lang if lang in ("ro", "en") else detect_language(text)

    # etapa 2: procesare emoji
    high_risk, ambiguous = load_emoji_config(config_path)
    text = replace_emoji_with_text(text, high_risk, ambiguous, lang=detected_lang)

    # etapa 3: strip diacritice doar pentru română
    if detected_lang == "ro":
        text = strip_diacritics(text)
        
    # etapa 4: lowercase
    text = text.lower()
    
    # etapa 5: normalizare punctuație
    text = normalize_punctuation(text)

    # etapa 6: normalizare spații
    text = normalize_whitespace(text)

    return text, detected_lang

# test manual
if __name__ == "__main__":
    config = Path(__file__).parent / "emoji_config.yaml"

    examples = [
        ("Te omor 🔫🔪",                    None,   "RO detectat — armă"),
        ("Du-te dracu 💀😈",                None,   "RO detectat — ambigue ignorate"),
        ("Oferta GRATIS 🔥 click acum!",    None,   "RO detectat — emoji neutru"),
        ("Știu că ești un prost",           None,   "RO cu diacritice — strip"),
        ("stiu ca esti un prost",           None,   "RO fără diacritice — nemodificat"),
        ("You are an idiot 🔪",             None,   "EN detectat — fără strip"),
        ("Ce zi frumoasă ☀️",              "ro",   "forțat RO din UI"),
        ("text fără emoji",                 None,   "RO simplu"),
    ]

    print(f"{'INPUT':<45} {'LANG':<6} {'OUTPUT'}")
    print("-" * 100)
    for text, lang, desc in examples:
        result, detected = preprocess(text, lang=lang, config_path=config)
        print(f"{text:<45} [{detected}]  → {result}   # {desc}")