"""
Glossario per fumetto: nomi, generi e termini ricorrenti da tenere coerenti.

Il traduttore lavora una pagina (anzi, un batch di 10 balloon) alla volta e
non ha memoria di quelle precedenti: e' il motivo per cui lo stesso
personaggio finisce "lui" a pagina 12 e "lei" a pagina 40, e per cui un
soprannome viene reso in tre modi diversi nello stesso volume. Un elenco
breve, iniettato nel prompt, risolve entrambe le cose senza toccare il resto
della traduzione.

Il file vive accanto alle pagine di lavoro del fumetto
(<work_dir>/<fumetto>/glossary.json) ed e' scritto a mano o dalla Revisione:

    {
      "note": "Eric si traveste da donna: da pagina 65 in poi parla di se' al femminile",
      "termini": {
        "Erica": "Erica (femminile)",
        "waifu": "waifu",
        "chicks": "tipe"
      }
    }

Nessun campo e' obbligatorio: un glossario vuoto o assente lascia il prompt
esattamente com'era.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger("pipeline.glossary")

FILENAME = "glossary.json"

# Oltre questo numero di voci il glossario occuperebbe piu' contesto del
# testo da tradurre, e i modelli piccoli iniziano a ignorare le istruzioni
# lunghe. Chi ne ha bisogno di piu' ha probabilmente un problema diverso.
MAX_TERMS = 60


def load(work_dir: Path) -> dict:
    """Legge il glossario del fumetto a cui appartiene la cartella di pagina.

    `work_dir` e' la cartella della pagina (…/<fumetto>/<pagina>): il
    glossario sta un livello sopra, perche' vale per tutto il volume.
    """
    path = Path(work_dir).parent / FILENAME
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"Glossario non leggibile ({path}): {e}, proseguo senza.")
        return {}
    if not isinstance(data, dict):
        log.warning(f"Glossario ignorato ({path}): il contenuto non e' un oggetto JSON.")
        return {}
    return data


def prompt_section(gloss: dict) -> str:
    """Il pezzo di prompt da aggiungere, o stringa vuota se non c'e' nulla."""
    if not gloss:
        return ""
    righe = []

    termini = gloss.get("termini") or {}
    if isinstance(termini, dict) and termini:
        voci = list(termini.items())[:MAX_TERMS]
        elenco = "; ".join(f"{k} -> {v}" for k, v in voci)
        righe.append(
            "Use these fixed renderings for recurring names and terms, exactly as given "
            f"(they keep the whole volume consistent): {elenco}."
        )

    nota = (gloss.get("note") or "").strip()
    if nota:
        righe.append(f"Context for this comic: {nota}")

    vietate = gloss.get("vietate") or []
    if isinstance(vietate, list) and vietate:
        righe.append(
            "Never use these words or expressions, they were wrong in earlier pages: "
            + ", ".join(str(v) for v in vietate[:MAX_TERMS]) + "."
        )

    if not righe:
        return ""
    return "\n" + "\n".join(righe) + "\n"
