import json
import logging
from pathlib import Path

log = logging.getLogger("pipeline.translate_skip")


def run(ocr_json_path: Path, cfg: dict, force: bool = False) -> Path:
    """
    Legge ocr.json di una pagina e scrive translated.json copiando
    testo_originale in testo_tradotto senza chiamare nessun traduttore.
    Usato quando il fumetto e' gia' in italiano (es. si vuole solo
    rifare pulizia/render) e si vuole comunque eseguire la pipeline
    completa senza passare per lo stage di traduzione.
    """
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for det in data["detections"]:
        det["testo_tradotto"] = det.get("testo_originale", "").strip()

    translated_json_path = ocr_json_path.parent / "translated.json"
    with open(translated_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"Traduzione saltata (testo OCR copiato): {translated_json_path}")
    return translated_json_path
