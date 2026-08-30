import json
import logging
import re
from pathlib import Path

import requests
import fit_check
import glossary
import translate_cache as tc

log = logging.getLogger("pipeline.translate_local")


def _build_prompt(detections: list[dict], src_lang: str, tgt_lang: str, gloss: dict | None = None) -> str:
    lines = []
    for det in detections:
        testo = det.get("testo_originale", "").strip()
        if testo:
            lines.append(f"[balloon {det['balloon_id']}] {testo}")
    text_block = "\n".join(lines)
    prompt = (
        f"Translate the following {src_lang} comic book dialogue lines to {tgt_lang}. "
        f"Follow these rules strictly:\n"
        f"1. Preserve the exact meaning and grammatical roles from the source: keep who "
        f"is doing what to whom exactly as in the original — never swap subject and "
        f"object, never reverse who initiates an action.\n"
        f"2. Keep the original tone, register and profanity, do not soften anything.\n"
        f"3. Be careful with Italian slang intensifiers: constructions like \"un cazzo\", "
        f"\"un casso\", \"un tubo\" after a verb are NEGATIONS in colloquial Italian "
        f"(e.g. \"non capisco un cazzo\" = I don't understand anything at all). NEVER use "
        f"them to intensify a positive statement (do not translate \"I like it a lot\" as "
        f"\"mi piace un cazzo\", which means the opposite). Use other intensifiers instead "
        f"(e.g. \"da morire\", \"un casino\", \"alla grande\", \"cazzo se mi piace\").\n"
        f"4. Translate the actual meaning of idioms and exclamations; do not invent "
        f"unrelated expressions just because they sound colloquial — if unsure of a "
        f"natural equivalent, translate literally rather than substituting something "
        f"that changes the meaning.\n"
        f"5. Proofread your own output: no typos, no stray digits instead of letters, "
        f"correct spelling and grammar in {tgt_lang}.\n"
        f"Output only the translations, no explanations, one line per input line, "
        f"keeping the exact same [balloon N] tag format:"
        # Il glossario va dopo le regole e prima del testo: e' specifico di
        # questo volume e deve avere l'ultima parola sulle regole generali
        # (es. un soprannome che si lascia in inglese).
        + glossary.prompt_section(gloss or {})
        + f"\n\n{text_block}"
    )
    return prompt


def _call_local_llm(prompt: str, cfg: dict) -> str:
    """Chiamata al modello locale servito da llama-server (endpoint
    OpenAI-compatible). Il modello e' quello caricato all'avvio del server da
    main.start_llama_server: qui non si sceglie, se ne parla uno solo alla
    volta - per questo "model" e' un segnaposto, llama.cpp lo ignora."""
    lm_cfg = cfg["local_llm"]
    system_prompt = lm_cfg.get(
        "translate_system_prompt",
        "You are a professional comic book translator. "
        "Keep the original tone, register and profanity, do not soften anything."
    )
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    resp = requests.post(f"{lm_cfg['base_url']}/chat/completions", json=payload, timeout=240)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_response(response_text: str) -> dict[int, str]:
    pattern = re.compile(r"^\[balloon\s+(\d+)\]\s*(.*)$", re.MULTILINE)
    result = {}
    for match in pattern.finditer(response_text):
        balloon_id = int(match.group(1))
        testo = match.group(2).strip()
        result[balloon_id] = testo
    return result

def _chunk_detections(detections: list[dict], chunk_size: int = 10) -> list[list[dict]]:
    """Divide i balloon in gruppi più piccoli per evitare timeout su pagine con molti balloon."""
    return [detections[i:i + chunk_size] for i in range(0, len(detections), chunk_size)]


_SHORTEN_PROMPT = (
    "Sei un traduttore di fumetti. Questa battuta tradotta e' troppo lunga per "
    "il balloon in cui deve stare.\n"
    "Riscrivila piu' corta, al massimo {max_chars} caratteri, mantenendo senso, "
    "tono, registro ed eventuale turpiloquio. Puoi togliere parole di riempimento "
    "e usare sinonimi piu' brevi, ma non cambiare cosa viene detto e non "
    "aggiungere nulla.\n"
    "Rispondi SOLO con la battuta accorciata, senza virgolette ne' spiegazioni.\n\n"
    "Testo originale: {source}\n"
    "Traduzione attuale ({cur_chars} caratteri): {current}"
)


def _make_shorten_fn(cfg: dict):
    """Costruisce la funzione di accorciamento usata da fit_check: stessa
    connessione e stesso modello della traduzione, una battuta alla volta
    (sono pochi balloon per pagina, non vale la pena raggrupparli)."""
    def shorten(current: str, max_chars: int, source: str) -> str:
        prompt = _SHORTEN_PROMPT.format(
            max_chars=max_chars, source=source, current=current, cur_chars=len(current)
        )
        risposta = _call_local_llm(prompt, cfg)
        # Il modello puo' rispondere con virgolette o con un prefisso tipo
        # "Traduzione:": si tiene solo la prima riga non vuota, ripulita.
        for riga in (risposta or "").splitlines():
            riga = riga.strip().strip('"').strip()
            riga = re.sub(r"^(traduzione|versione corta|testo)\s*:\s*", "", riga, flags=re.I)
            if not riga:
                continue
            # I fumetti sono lettrati in maiuscolo e il resto della pagina lo
            # e' gia': il modello risponde spesso in minuscolo, e una battuta
            # in minuscolo in mezzo a tutte maiuscole si nota subito. Si
            # allinea al caso della traduzione che sta sostituendo, invece di
            # chiederlo nel prompt (piu' affidabile di una istruzione).
            lettere = [c for c in current if c.isalpha()]
            if lettere and sum(c.isupper() for c in lettere) / len(lettere) > 0.8:
                riga = riga.upper()
            return riga
        return ""

    return shorten


def lang_pair(cfg: dict) -> tuple[str, str]:
    """Lingue sorgente/destinazione nella forma usata dalle chiavi di cache
    ("English"/"Italian", non "en"/"it")."""
    lang_map = {"en": "English", "fr": "French", "it": "Italian"}
    sub = cfg["local_llm"]
    return (lang_map.get(sub["source_lang"], sub["source_lang"]),
            lang_map.get(sub["target_lang"], sub["target_lang"]))


def gloss_key_for(page_dir: Path) -> str:
    """Impronta breve e stabile del glossario del fumetto, da accodare al
    testo nella chiave di cache (stringa vuota quando non c'e' glossario,
    cosi' le traduzioni gia' in cache restano valide)."""
    gloss = glossary.load(page_dir)
    if not gloss:
        return ""
    import hashlib
    return "\x00" + hashlib.sha1(
        json.dumps(gloss, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]


def remember(page_dir: Path, cfg: dict, testo_originale: str, testo_tradotto: str) -> bool:
    """
    Registra in cache una traduzione decisa da una persona (import da TXT o
    modifica in Revisione), con la stessa chiave che usa `run`.

    Senza questo, una traduzione corretta a mano andava persa al primo
    rilancio dello stage di traduzione: la cache non la conosceva e il
    modello ritraduceva da zero, riproponendo la resa che era stata
    scartata. Vale anche tra pagine e capitoli diversi, dove la stessa
    battuta ricorre.
    """
    orig = (testo_originale or "").strip()
    trad = (testo_tradotto or "").strip()
    if not orig or not trad:
        return False
    src_lang, tgt_lang = lang_pair(cfg)
    tc.put(orig + gloss_key_for(page_dir), src_lang, tgt_lang, trad)
    return True


def run(ocr_json_path: Path, cfg: dict, force: bool = False) -> Path:
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    src_lang, tgt_lang = lang_pair(cfg)

    gloss = glossary.load(ocr_json_path.parent)
    gloss_key = gloss_key_for(ocr_json_path.parent)
    if gloss:
        n_termini = len((gloss.get("termini") or {}))
        log.info(f"{data['page_id']}: glossario del fumetto attivo ({n_termini} termini)")

    detections_with_text = [d for d in data["detections"] if d.get("testo_originale", "").strip()]

    if not detections_with_text:
        log.warning(f"{data['page_id']}: nessun testo da tradurre.")
    else:
        chunks = _chunk_detections(detections_with_text, chunk_size=10)
        log.info(f"{data['page_id']}: {len(detections_with_text)} balloon divisi in {len(chunks)} batch...")

        all_translations = {}
        for i, chunk in enumerate(chunks):
            # filtra balloon già in cache
            chunk_to_translate = []
            for det in chunk:
                # Il glossario entra nella chiave di cache: con un glossario
                # diverso la stessa battuta va tradotta diversamente, e senza
                # questo la cache restituirebbe la vecchia resa ignorandolo.
                cache_src = det["testo_originale"] + gloss_key
                cached = None if force else tc.get(cache_src, src_lang, tgt_lang)
                if cached is not None:
                    all_translations[det["balloon_id"]] = cached
                    log.info(f"{data['page_id']} balloon {det['balloon_id']}: cache HIT")
                else:
                    chunk_to_translate.append(det)

            if not chunk_to_translate:
                log.info(f"{data['page_id']} batch {i+1}/{len(chunks)}: tutti in cache, salto chiamata API")
                continue

            prompt = _build_prompt(chunk_to_translate, src_lang, tgt_lang, gloss)

            log.info(f"{data['page_id']} batch {i+1}/{len(chunks)}: "
                     f"invio {len(chunk_to_translate)} balloon al modello locale...")
            response_text = _call_local_llm(prompt, cfg)

            parsed = _parse_response(response_text)

            # salva in cache i nuovi risultati
            for det in chunk_to_translate:
                bid = det["balloon_id"]
                if bid in parsed:
                    tc.put(det["testo_originale"] + gloss_key, src_lang, tgt_lang, parsed[bid])

            all_translations.update(parsed)

        for det in data["detections"]:
            bid = det["balloon_id"]
            if bid in all_translations:
                det["testo_tradotto"] = all_translations[bid]
            else:
                if det.get("testo_originale", "").strip():
                    log.warning(f"{data['page_id']} balloon {bid}: nessuna traduzione ricevuta.")
                det["testo_tradotto"] = ""

        # Controllo di capienza: meglio accorciare adesso, con il traduttore
        # gia' connesso e il contesto in mano, che scoprire il testo troncato
        # nella tavola renderizzata (vedi fit_check).
        try:
            n = fit_check.shorten_page(data, cfg, _make_shorten_fn(cfg))
            if n:
                log.info(f"{data['page_id']}: {n} traduzioni accorciate per farle stare nel balloon")
        except Exception as e:
            log.warning(f"{data['page_id']}: controllo capienza saltato ({e})")

    translated_json_path = ocr_json_path.parent / "translated.json"
    with open(translated_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"Tradotto: {translated_json_path}")
    return translated_json_path