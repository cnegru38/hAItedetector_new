from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Taxonomy:
    intents: list[str]
    thresholds: dict[str, float]
    thresholds_ro: dict[str, float]

def load_taxonomy(taxonomy_path: str | Path) -> Taxonomy:
    p = Path(taxonomy_path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))

    intents = data.get("intents", [])
    thresholds = data.get("thresholds", {})
    thresholds_ro = data.get("thresholds_ro", {})

    # fallback sigur
    if not intents:
        intents = ["spam", "hate_speech", "harassment", "obscene"]
    if not thresholds:
        thresholds = {k: 0.5 for k in intents}
    # dacă thresholds_ro lipsește din YAML folosim thresholds EN ca fallback
    if not thresholds_ro:
        thresholds_ro = dict(thresholds)

    return Taxonomy(intents=intents, thresholds=thresholds, thresholds_ro=thresholds_ro)