"""
Cache persistente per le traduzioni.
Chiave = hash SHA256 di (testo + lingua_src + lingua_tgt).
Evita chiamate API ripetute per testi già tradotti.
"""

import json
import hashlib
import logging
from pathlib import Path

log = logging.getLogger("pipeline.translate_cache")

_CACHE_FILE = Path("work/translation_cache.json")
_cache = None  # caricato lazy


def _ensure_loaded():
    """Carica la cache in memoria una sola volta."""
    global _cache
    if _cache is not None:
        return
    if _CACHE_FILE.exists():
        try:
            _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            log.info(f"Cache caricata: {len(_cache)} traduzioni")
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Cache corrotta o illeggibile ({e}), ricreo vuota.")
            _cache = {}
    else:
        _cache = {}

def clear():
    """Svuota la cache in memoria e su disco."""
    global _cache
    _cache = {}
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()


def all_keys() -> list[str]:
    """Ritorna le chiavi presenti in cache (per ispezione/debug)."""
    _ensure_loaded()
    return list(_cache.keys())


def _make_key(text: str, src_lang: str, tgt_lang: str) -> str:
    """Chiave univoca per (testo, src, tgt)."""
    payload = f"{src_lang}:{tgt_lang}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(text: str, src_lang: str, tgt_lang: str) -> str | None:
    """
    Cerca una traduzione in cache.
    Ritorna la traduzione o None se non trovata.
    """
    _ensure_loaded()
    key = _make_key(text, src_lang, tgt_lang)
    cached = _cache.get(key)
    if cached is not None:
        log.debug(f"Cache HIT: {text[:40]}...")
    return cached


def put(text: str, src_lang: str, tgt_lang: str, translation: str):
    """
    Salva una traduzione in cache e scrive su disco.
    """
    _ensure_loaded()
    key = _make_key(text, src_lang, tgt_lang)
    _cache[key] = translation
    _save()


def _save():
    """Persiste la cache su disco."""
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=2), encoding="utf-8")


def stats() -> dict:
    """Statistiche della cache."""
    _ensure_loaded()
    return {"entries": len(_cache), "file": str(_CACHE_FILE)}