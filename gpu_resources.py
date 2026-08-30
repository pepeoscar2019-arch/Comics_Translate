import logging

import requests

log = logging.getLogger("pipeline.gpu")


def free_comfyui(cfg: dict) -> None:
    """Libera la VRAM occupata da ComfyUI (modello di pulizia caricato),
    prima di uno stage che non lo usa (OCR, traduzione). Best-effort: se
    ComfyUI non e' in esecuzione, non fa nulla."""
    base_url = cfg.get("comfyui", {}).get("base_url", "http://127.0.0.1:8188").rstrip("/")
    try:
        requests.post(f"{base_url}/free", json={"unload_models": True, "free_memory": True}, timeout=5)
        log.info("ComfyUI: modelli scaricati dalla VRAM.")
    except requests.exceptions.RequestException:
        pass
