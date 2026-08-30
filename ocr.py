import json
import base64
import logging
import re
import zlib
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

import font_analyzer
import paths

log = logging.getLogger("pipeline.ocr")

ONOMATOPEE = {
    "BOOM", "BAM", "POW", "CRASH", "BANG", "WHAM", "ZAP", "ZING",
    "KABOOM", "KAPOW", "BLAM", "KA-BOOM", "KA-POW", "SMASH", "THUD",
    "THUMP", "CLUNK", "CLANK", "RATTLE", "SHAKE", "CLANG", "RING",
    "DING", "DONG", "BONG", "BONK", "SWOOSH", "WHOOSH", "VROOM",
    "SNAP", "CRACK", "POP", "SPLAT", "SPLASH", "DRIP", "DROP",
    "ROAR", "GROWL", "HISS", "BUZZ", "WHIR", "CRUNCH", "CHOMP",
    "MUNCH", "SLURP", "GULP", "FIZZ", "BOING", "AH", "OH", "EH",
    "UH", "HUH", "HA", "HEH", "HEE", "EEK", "GASP",
    "PANT", "WHEE", "ZZZ", "ZZ", "Z",
    "SCREAM", "SHOUT", "YELL", "WHISPER", "MUMBLE", "SIGH", "SNIFF",
    "COUGH", "SNEEZE", "GRUNT"
}


def _is_onomatopoeia(text: str) -> bool:
    """
    Riconosce onomatopee/effetti sonori puri, da NON tradurre. Non deve
    scartare parole di dialogo brevi ma con significato reale (es. "DAMN!",
    "YES!!!", "WOW!", "STOP!") solo perche' sono maiuscole e corte: sono
    dialogo a tutti gli effetti nei fumetti, non rumori.
    """
    if not text:
        return True
    text_clean = text.strip().upper()
    for ono in ONOMATOPEE:
        if text_clean == ono or text_clean.startswith(ono + "!") or text_clean.startswith(ono + "!!"):
            return True
    if text_clean.isupper() and len(text_clean) <= 8:
        # Lettere ripetute 3+ volte (AAAAH, NOOOO, ZZZZ, HAHAHA): segnale forte di rumore/verso.
        if re.search(r'([AEIOU])\1{2,}', text_clean):
            return True
        if re.search(r'([BCDFGHJKLMNPQRSTVWXYZ])\1{2,}', text_clean):
            return True
    return False


def _clean_text(text: str) -> str:
    if not text:
        return text
    url_pattern = re.compile(
        r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?\b|'
        r'https?://[^\s]+|'
        r'www\.[^\s]+',
        re.IGNORECASE
    )
    email_pattern = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
    phone_pattern = re.compile(r'\b\d{3,}[-\s]?\d{3,}[-\s]?\d{3,}\b')
    page_pattern = re.compile(r'\b(?:pag\.?\s*)?\d{1,4}\b', re.IGNORECASE)
    text = url_pattern.sub('', text)
    text = email_pattern.sub('', text)
    text = phone_pattern.sub('', text)
    text = page_pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _crop_to_b64(image: cv2.Mat, bbox: list[int], upscale: float = 1.0, pad: int = 0) -> str:
    """`upscale` e `pad` servono al secondo tentativo sui balloon che il
    modello ha letto vuoti (vedi run): un ritaglio piccolo, o con le lettere
    a filo del bordo, e' il caso in cui i VLM tendono a non vedere nulla -
    ingrandirlo e dargli un po' di margine intorno spesso basta a farlo
    leggere, senza cambiare modello ne' prompt."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    if pad:
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Crop vuoto per bbox {bbox} su immagine {w}x{h}")
    if upscale and upscale != 1.0:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _ocr_model_name(cfg: dict) -> str:
    """Nome del modello da mettere nel payload. llama-server serve un solo
    modello per volta (quello caricato da main.start_llama_server) e ignora
    questo campo: resta solo perche' l'API OpenAI lo prevede, e serve nei log
    per capire quale backend di visione ha risposto."""
    return "paddleocr-vl" if cfg.get("ocr_backend", "qwen") == "paddleocr_vl" else "qwen3-vl"


_QWEN_SYSTEM_PROMPT = (
    "Sei un OCR specializzato in fumetti. Il tuo UNICO compito è trascrivere "
    "il testo visibile nel balloon o nella didascalia: dialogo o pensiero dei "
    "personaggi, oppure testo narrativo/descrittivo (caption, es. su sfondo "
    "colorato) che introduce o commenta la scena.\n\n"
    "IMPORTANTE: IGNORA e NON trascrivere:\n"
    "- Onomatopee/effetti sonori puri, senza significato di dialogo "
    "(es. BOOM, BAM, CRASH, POW, ZZZZ, AAAAH, HAHAHA)\n"
    "- URL, indirizzi web, email, numeri di pagina, marchi\n\n"
    "Trascrivi invece SEMPRE le parole con un significato reale anche se "
    "brevi, maiuscole o esclamative (es. WOW!, DAMN!, YES!!!, STOP!, HEY!, "
    "OMG!): sono dialogo a tutti gli effetti, non rumori.\n\n"
    "RISPONDI SOLO con il testo trascritto, nient'altro. "
    "Se non vedi testo, rispondi con stringa vuota."
)


_REPEATABLE_WORDS = {
    # inglese
    "yes", "no", "yeah", "nope", "hey", "hi", "ok", "okay", "oh", "ah", "ha",
    "ho", "wow", "please", "stop", "wait", "help", "run", "come", "go", "now",
    "never", "more", "very", "so", "well", "hello", "bye", "sorry",
    # italiano
    "si", "no", "ehi", "ciao", "dai", "presto", "aiuto", "forza", "basta",
    "fermo", "ferma", "vai", "corri", "subito", "mai", "piu", "molto",
    "certo", "davvero", "scusa", "grazie",
}
"""Parole che nei fumetti si raddoppiano per enfasi anche senza pausa in
mezzo ("YES YES", "NO NO", "GO GO"). Sono interiezioni e imperativi brevi:
un loop del modello raddoppia invece parole qualsiasi ("going going",
"the the"), che restano cosi' riconoscibili. Vedi _looks_hallucinated."""

_ENDS_WITH_PUNCT_RE = re.compile(r"[,.;:!?\u2026\u2014-]$")
"""Un token che finisce con punteggiatura chiude una pausa: la parola uguale
che segue e' enfasi voluta, non un loop del modello (vedi _looks_hallucinated)."""


_CHARS_PER_PX = 1 / 600
"""Caratteri ammessi per pixel d'area del balloon nel controllo di lunghezza
di _looks_hallucinated. Volutamente generoso (una didascalia fitta sta
intorno a 1/900): serve a distinguere un loop del modello da un testo lungo
ma plausibile, non a stimare la capienza reale."""

_MIN_CHARS_LIMIT = 250
"""Soglia minima di caratteri sotto la quale non si scende, qualunque sia
l'area: sui balloon piccoli - dove i loop del modello sono piu' frequenti -
il comportamento resta quello di prima."""


def _looks_hallucinated(text: str, crop_area: int | None = None) -> bool:
    """Euristica per rilevare output degenerati di PaddleOCR-VL-1.6 (modello
    0.9B specializzato, non un chatbot general-purpose come Qwen3-VL): a
    volte va in loop di ripetizione o duplica porzioni di frase invece di
    limitarsi a trascrivere - vedi _call_vlm_ocr. Non serve/non scatta per
    Qwen, che non mostra questo comportamento nei test.

    Tre segnali, ognuno sufficiente da solo:
    - testo assurdamente lungo per un balloon di fumetto;
    - rapporto di compressione molto basso (testo altamente ripetitivo,
      es. "- - - - - - ..." per centinaia di caratteri);
    - due parole identiche adiacenti, senza punteggiatura in mezzo e non
      fra quelle che si raddoppiano per enfasi (_REPEATABLE_WORDS):
      "going going" e "the the" si', "YES YES" e "NO NO" no.

    La punteggiatura e' decisiva nell'ultimo punto: nei fumetti la
    ripetizione separata da virgola o punto interrogativo e' enfasi
    normalissima ("YES, YES!", "ERIC, ERIC, YOU ARE MY ONLY HOPE.",
    "OKAY, OKAY, I HAVE OTHER CLOTHES.", "...WANT ME TO DO? DO YOU WANT...").
    Ignorandola, questa regola scartava letture perfettamente corrette e il
    balloon restava vuoto - e quindi in lingua originale fino alla tavola
    finita: 9 balloon su 280 in un solo volume, tutti falsi allarmi."""
    # Limite proporzionale all'area quando la conosciamo: con un tetto fisso
    # a 250 caratteri, le didascalie narrative larghe (400+ caratteri, del
    # tutto normali) venivano scambiate per allucinazioni e buttate via.
    max_chars = _MIN_CHARS_LIMIT
    if crop_area:
        max_chars = max(_MIN_CHARS_LIMIT, int(crop_area * _CHARS_PER_PX))
    if len(text) > max_chars:
        return True
    if len(text) >= 20:
        compressed = zlib.compress(text.encode("utf-8"))
        if len(compressed) / len(text) < 0.35:
            return True
    raw = [w for w in text.split() if w]
    for first, second in zip(raw, raw[1:]):
        # Una ripetizione "vera" da loop del modello arriva senza segni in
        # mezzo; se il primo token si chiude con punteggiatura e' una figura
        # retorica, non un difetto di trascrizione.
        if _ENDS_WITH_PUNCT_RE.search(first):
            continue
        a = re.sub(r"[^\w]", "", first).lower()
        b = re.sub(r"[^\w]", "", second).lower()
        if a and a == b and a not in _REPEATABLE_WORDS:
            return True
    return False


def _call_vlm_ocr_once(image_b64: str, cfg: dict, elaborate_prompt: bool) -> str:
    """Una singola chiamata OCR al modello di visione caricato in LM Studio
    (Qwen3-VL o, con ocr_backend: paddleocr_vl, PaddleOCR-VL-1.6): stesso
    endpoint OpenAI-compatible, cambia solo quale modello LM Studio ha in
    memoria (llama-server carica un modello di visione alla volta).
    elaborate_prompt=False
    usa un prompt minimale (l'unico che PaddleOCR-VL segue in modo affidabile
    su alcuni balloon, vedi _looks_hallucinated); Qwen usa sempre il prompt
    elaborato, piu' robusto per lui."""
    lm_cfg = cfg["local_llm"]
    ocr_model = _ocr_model_name(cfg)
    is_paddleocr = cfg.get("ocr_backend", "qwen") == "paddleocr_vl"

    if is_paddleocr and not elaborate_prompt:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": "Transcribe the text in this image."},
                ],
            },
        ]
        max_tokens = 200
    else:
        messages = [
            {"role": "system", "content": _QWEN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": "Trascrivi il dialogo in questo balloon."},
                ],
            },
        ]
        max_tokens = 2300

    payload = {
        "model": ocr_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if not is_paddleocr:
        # disabilita il "thinking" di Qwen3: specifico al suo chat template,
        # senza effetto (e senza errori) su altri modelli/template.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    resp = requests.post(f"{lm_cfg['base_url']}/chat/completions", json=payload, timeout=160)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    content = content.strip('"\'` \n')
    content = content.replace('\n', ' ')
    if content.startswith("```"):
        content = content.split("```")[-2] if content.count("```") >= 2 else content

    return _clean_text(content).strip()


_RETRY_PAD_PX = 8
"""Margine (px) aggiunto attorno al bbox nel secondo tentativo di OCR: le
lettere a filo del bordo del ritaglio sono una delle cause per cui il modello
restituisce vuoto."""

_EMPTY_SUSPECT_MIN_GLYPHS = 8
"""Numero minimo di macchie d'inchiostro con geometria da glifo perche' un
balloon dato per vuoto dall'OCR venga segnalato come testo mancato."""


def _looks_like_text(crop) -> tuple[bool, tuple]:
    """
    Dice se un ritaglio contiene quasi certamente del testo, guardando solo
    i pixel (nessun modello). Serve a intercettare i balloon che il VLM
    restituisce vuoti pur avendo testo dentro: succede soprattutto su bolle
    a sfondo colorato o con testo molto fitto, e passa inosservato fino al
    render, dove quel balloon resta in lingua originale (la pulizia lo
    ripristina proprio perche' non ha traduzione).

    Criteri, tarati per distinguere il testo da un fondale disegnato o
    fotografico (il caso che genera falsi allarmi: fogliame, mattoni,
    scaffali di libri):
      - abbastanza componenti connesse con geometria da glifo (altezza fra
        il 3% e il 40% del ritaglio, larghezza plausibile, rapporto
        d'aspetto e riempimento sensati);
      - almeno 2 righe da 3+ glifi: il testo si allinea in righe, la texture
        di un fondale no;
      - altezze dei glifi omogenee (coefficiente di variazione <= 0.55): in
        una stringa di testo i caratteri hanno tutti la stessa taglia;
      - uno sfondo con un tono dominante (>= 20% dei pixel in un solo bin
        dell'istogramma): l'interno di un balloon e' piatto, un fondale no.

    Ritorna (sospetto, (glifi, righe, uniformita, cv_altezze)); le metriche
    servono al log.
    """
    if crop is None or crop.size == 0:
        return False, (0, 0, 0.0, 0.0)
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return False, (0, 0, 0.0, 0.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    uniformity = float(hist.max() / hist.sum()) if hist.sum() else 0.0

    best = (0, 0, uniformity, 9.9)
    # Il testo puo' essere scuro su chiaro o chiaro su scuro (didascalie
    # invertite): si prova la soglia di Otsu in entrambi i versi.
    for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, binary = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
        if (binary > 0).mean() > 0.45:
            # Piu' del 45% del ritaglio e' "inchiostro": campitura, non testo.
            continue
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(binary)
        glyphs = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if not (h * 0.03 <= ch <= h * 0.40):
                continue
            if not (w * 0.004 <= cw <= w * 0.30):
                continue
            if not (0.08 <= cw / max(1, ch) <= 3.0):
                continue
            if area < 0.20 * cw * ch or area > 0.15 * h * w:
                continue
            glyphs.append((x, y, cw, ch))
        if len(glyphs) < 5:
            continue
        heights = np.array([g[3] for g in glyphs], dtype=float)
        cv_h = float(heights.std() / max(1e-6, heights.mean()))
        median_h = float(np.median(heights))
        centers = sorted((g[1] + g[3] / 2) for g in glyphs)
        rows, count, prev = [], 1, centers[0]
        for c in centers[1:]:
            if c - prev > median_h * 0.8:
                rows.append(count)
                count = 1
            else:
                count += 1
            prev = c
        rows.append(count)
        dense_rows = sum(1 for r in rows if r >= 3)
        if len(glyphs) > best[0]:
            best = (len(glyphs), dense_rows, uniformity, cv_h)

    glyphs_n, dense_rows, uni, cv_h = best
    suspect = (
        glyphs_n >= _EMPTY_SUSPECT_MIN_GLYPHS
        and dense_rows >= 2
        and uni >= 0.20
        and cv_h <= 0.55
    )
    return suspect, best


def _call_vlm_ocr(image_b64: str, cfg: dict, crop_area: int | None = None) -> tuple[str, str]:
    """Ritorna (testo, motivo_del_vuoto). Il motivo distingue i modi in cui
    il testo puo' uscire vuoto - "" nessun vuoto, "hallucinated" output
    scartato, "onomatopea" scartata di proposito, "vuoto" il modello non ha
    letto nulla - perche' solo l'ultimo caso vale la pena di verificare a
    pixel (_looks_like_text): negli altri il vuoto e' una scelta."""
    is_paddleocr = cfg.get("ocr_backend", "qwen") == "paddleocr_vl"

    content = _call_vlm_ocr_once(image_b64, cfg, elaborate_prompt=False)
    if is_paddleocr and _looks_hallucinated(content, crop_area):
        log.warning(f"  [!] paddleocr_vl: output sospetto col prompt minimale ({content[:80]!r}), ritento col prompt esteso")
        content = _call_vlm_ocr_once(image_b64, cfg, elaborate_prompt=True)
        if _looks_hallucinated(content, crop_area):
            log.warning(f"  [!] paddleocr_vl: output ancora sospetto ({content[:80]!r}), balloon lasciato vuoto")
            return "", "hallucinated"

    if _is_onomatopoeia(content):
        log.info(f"  [🔇] Onomatopea ignorata: \"{content}\"")
        return "", "onomatopea"

    return content, ("" if content else "vuoto")


def run(detections_json_path: Path, page_path: Path, cfg: dict, work_dir: Path) -> Path:
    with open(detections_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    page_id = data["page_id"]
    page_work_dir = work_dir / page_id

    image = cv2.imread(str(page_path))
    if image is None:
        raise FileNotFoundError(f"Impossibile leggere immagine: {page_path}")

    ocr_backend = cfg.get("ocr_backend", "qwen")
    log.info(f"{page_id}: OCR su {len(data['detections'])} balloon (backend: {ocr_backend})...")

    render_cfg = cfg.get("rendering", {})
    fonts_dir = str(paths.resolve(render_cfg.get("fonts_dir", "fonts")))
    default_font = render_cfg.get("font_path")
    if default_font:
        default_font = str(paths.resolve(default_font))
    font_library = font_analyzer.FontLibrary(fonts_dir)

    # Collassa i gruppi bubble_group_id PRIMA dell'OCR: quando CTD ha spezzato
    # il testo di un unico balloon lobato in piu' box, _merge_with_bubble_shapes
    # (detect.py) ha gia' dato al primo membro del gruppo il bbox/maschera del
    # balloon INTERO (balloon_source == "yolov8seg"), mentre gli altri membri
    # hanno ancora il loro bbox stretto di CTD. OCRizzare ogni box separatamente
    # e poi ricucire i testi (vecchio _dedupe_shared_bubble_text) duplicava/
    # mescolava il testo quando le letture delle aree sovrapposte divergevano di
    # qualche carattere (es. "KNOW..." vs "KNOWN!"): lo strip del prefisso
    # comune falliva e concatenava i doppioni. Qui invece si tiene solo il
    # membro con la forma reale (che copre tutto il balloon) e si scartano gli
    # altri: una sola lettura OCR sull'intera area, niente ricucitura fragile.
    # render.py NON usa i box CTD spezzati per impaginare i lobi: ri-spezza da
    # solo il testo completo sulla forma della maschera (find_waist), quindi
    # perdere i box stretti non gli toglie nulla.
    data["detections"] = _collapse_bubble_groups(data["detections"])

    empty_suspects: list = []
    recovered: list = []
    for det in data["detections"]:
        bbox = det["bbox"]
        try:
            image_b64 = _crop_to_b64(image, bbox)
            crop_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            testo, empty_reason = _call_vlm_ocr(image_b64, cfg, crop_area)
        except Exception as e:
            log.warning(f"  balloon {det['balloon_id']}: errore crop/OCR ({e}), salto.")
            testo, empty_reason = "", "errore"
        det["testo_originale"] = testo
        det.pop("ocr_empty_suspect", None)
        if testo:
            log.info(f"  balloon {det['balloon_id']}: \"{testo[:60]}\"")
        else:
            log.info(f"  balloon {det['balloon_id']}: (vuoto/onomatopea)")
            # Verifica a pixel del vuoto: se nel ritaglio c'e' chiaramente
            # del testo, il modello l'ha mancato. Senza questa segnalazione
            # il balloon resta in lingua originale fino al render e ci si
            # accorge dell'errore solo guardando la tavola finita.
            if empty_reason in ("vuoto", "hallucinated"):
                x1, y1, x2, y2 = bbox
                suspect, metrics = _looks_like_text(image[max(0, y1):y2, max(0, x1):x2])
                # Il secondo tentativo ha senso solo se il modello non ha
                # letto nulla: sul ramo "hallucinated" il prompt esteso e'
                # gia' stato provato da _call_vlm_ocr, ripeterlo qui
                # rifarebbe la stessa chiamata per lo stesso esito.
                if suspect and empty_reason == "vuoto":
                    # Secondo tentativo prima di arrendersi: stesso modello,
                    # ma ritaglio ingrandito 2x e con un margine intorno, e
                    # prompt esteso. Sui balloon mancati dal primo passaggio
                    # recupera la maggior parte del testo senza intervento
                    # manuale; solo se fallisce anche questo si segnala.
                    log.info(f"  balloon {det['balloon_id']}: nel ritaglio c'e' testo, ritento ingrandito 2x...")
                    try:
                        big_b64 = _crop_to_b64(image, bbox, upscale=2.0, pad=_RETRY_PAD_PX)
                        testo = _clean_text(_call_vlm_ocr_once(big_b64, cfg, elaborate_prompt=True)).strip()
                        if _looks_hallucinated(testo, crop_area) or _is_onomatopoeia(testo):
                            testo = ""
                    except Exception as e:
                        log.warning(f"  balloon {det['balloon_id']}: secondo tentativo fallito ({e})")
                        testo = ""
                    det["testo_originale"] = testo

                if suspect and testo:
                    recovered.append(det["balloon_id"])
                    log.info(f"  [\u2713] balloon {det['balloon_id']} recuperato: \"{testo[:60]}\"")
                elif suspect:
                    det["ocr_empty_suspect"] = True
                    empty_suspects.append(det["balloon_id"])
                    log.warning(
                        f"  [!] balloon {det['balloon_id']} dato per vuoto ma il ritaglio "
                        f"contiene testo (glifi={metrics[0]}, righe={metrics[1]}): "
                        f"testo mancato dall'OCR anche al secondo tentativo, "
                        f"da trascrivere in Revisione"
                    )

        # Suggerisce automaticamente un font simile allo stile originale del
        # balloon (bold/corsivo/manoscritto). L'utente puo' sempre cambiarlo
        # a mano in revisione: quel gesto imposta font_auto=False cosi' le
        # riesecuzioni successive di questo stage non lo sovrascrivono.
        if testo and not det.get("font_path"):
            try:
                features = font_analyzer.analyze_font(image, bbox, testo)
                font_path = font_library.find_best_match(features) or default_font
                det["font_path"] = font_path
                det["font_auto"] = True
            except Exception as e:
                log.warning(f"  balloon {det['balloon_id']}: analisi font fallita ({e}), uso default.")

    # I gruppi sono gia' stati collassati prima dell'OCR
    # (_collapse_bubble_groups): non c'e' piu' testo condiviso da deduplicare.
    # Restano solo da ripulire eventuali campi temporanei di raggruppamento.
    for det in data["detections"]:
        det.pop("bubble_group_id", None)
        det.pop("_orig_y1", None)

    ocr_json_path = page_work_dir / "ocr.json"
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if recovered:
        log.info(
            f"{page_id}: {len(recovered)} balloon recuperati al secondo tentativo "
            f"(balloon_id {recovered})"
        )
    if empty_suspects:
        log.warning(
            f"{page_id}: {len(empty_suspects)} balloon dati per vuoti ma con testo "
            f"visibile (balloon_id {empty_suspects}) - da trascrivere a mano"
        )

    log.info(f"{page_id}: ocr.json salvato ({ocr_json_path})")
    return ocr_json_path


def _strip_common_prefix(first: str, second: str) -> str:
    """
    Se `second` inizia con lo stesso testo (a meno di punteggiatura/maiuscole)
    di `first`, rimuove quella parte iniziale e ritorna solo il resto.
    Altrimenti ritorna `second` invariato.
    """
    def norm_words(s: str) -> list[str]:
        return re.sub(r"[^\w\s]", "", s).lower().split()

    first_norm = norm_words(first)
    second_words = second.split()
    second_norm = norm_words(second)

    n = 0
    while n < len(first_norm) and n < len(second_norm) and first_norm[n] == second_norm[n]:
        n += 1
    if n == 0:
        return second

    remainder = " ".join(second_words[n:]).strip()
    return remainder if remainder else second


def _collapse_bubble_groups(detections: list[dict]) -> list[dict]:
    """
    Collassa ogni gruppo bubble_group_id in una sola detection PRIMA dell'OCR.

    Un gruppo nasce quando comic-text-detector spezza il dialogo di un unico
    balloon (spesso lobato) in piu' box di testo, e _merge_with_bubble_shapes
    (detect.py) li associa allo stesso balloon reale di bubble_seg. Uno solo
    dei membri — quello con balloon_source == "yolov8seg" — ha ricevuto il
    bbox e la maschera dell'INTERO balloon; gli altri conservano il loro bbox
    stretto di CTD. Per l'OCR ci serve leggere una volta sola l'intera area:
    teniamo il membro yolov8seg e scartiamo gli altri. Il testo verra' poi
    ri-impaginato da render.py sulla forma della maschera (find_waist), che
    non ha bisogno dei box CTD originali.

    Le detection SENZA bubble_group_id (balloon singoli, caption rettangolari,
    SFX fuori balloon: tutto cio' che non e' stato raggruppato) passano
    invariate. Se per qualche motivo un gruppo non contiene un membro
    yolov8seg (non dovrebbe accadere: il primo assegnato lo e' sempre), si
    ripiega sul membro piu' in alto (_orig_y1 minore) per non perdere il
    testo.
    """
    groups: dict[int, list[dict]] = {}
    singles: list[dict] = []
    order: list[object] = []  # preserva l'ordine: gid (prima volta visto) o la det stessa

    for det in detections:
        gid = det.get("bubble_group_id")
        if gid is None:
            singles_marker = det
            singles.append(det)
            order.append(singles_marker)
        else:
            if gid not in groups:
                order.append(gid)
            groups.setdefault(gid, []).append(det)

    result: list[dict] = []
    for item in order:
        if isinstance(item, dict):
            result.append(item)  # detection singola
            continue
        members = groups[item]
        if len(members) == 1:
            primary = members[0]
        else:
            primary = next(
                (m for m in members if m.get("balloon_source") == "yolov8seg"),
                min(members, key=lambda d: d.get("_orig_y1", 0)),
            )
            log.info(
                f"  gruppo balloon {item}: {len(members)} box CTD collassati in 1 "
                f"(OCR singolo sull'area del balloon reale)"
            )
        result.append(primary)

    return result


def _dedupe_shared_bubble_text(data: dict) -> None:
    """
    Quando comic-text-detector spezza il dialogo di un unico balloon reale in
    piu' box di testo (vedi detect.py: bubble_group_id), ognuno viene
    OCRizzato separatamente sulla propria area: da' un ritaglio piu' stretto
    e quindi un OCR piu' accurato per ciascuna porzione rispetto a leggere
    l'intero balloon in un colpo solo. Se il box piu' piccolo copre l'inizio
    del dialogo, l'OCR del box piu' grande (che in bubble_seg contiene
    l'intero balloon) finisce pero' per ripetere quella stessa parte
    iniziale: qui si raggruppano i testi per bubble_group_id, si ordinano
    per posizione verticale originale, e si toglie da ciascun testo la parte
    iniziale gia' catturata da quello precedente nello stesso gruppo.

    A questo punto il gruppo rappresenta UN solo balloon fisico: i box che
    lo compongono servivano solo a migliorare l'OCR, non sono balloon
    separati. Se restassero entrambi in data["detections"], clean.py e
    render.py li pulirebbero/renderizzerebbero come due balloon distinti che
    pero' occupano (in parte) la stessa area della pagina, sovrapponendo i
    due testi nel render finale. Il testo deduplicato di ogni membro viene
    quindi concatenato nel membro con la forma reale del balloon
    (balloon_source == "yolov8seg", l'unica il cui bbox/maschera copre
    l'intero balloon), e gli altri membri del gruppo vengono scartati.
    """
    groups: dict[int, list[dict]] = {}
    for det in data["detections"]:
        gid = det.get("bubble_group_id")
        if gid is not None:
            groups.setdefault(gid, []).append(det)

    redundant_ids: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda d: d.get("_orig_y1", 0))
        for i in range(1, len(members)):
            prev_text = members[i - 1].get("testo_originale", "")
            cur_text = members[i].get("testo_originale", "")
            if prev_text and cur_text:
                members[i]["testo_originale"] = _strip_common_prefix(prev_text, cur_text)

        primary = next((m for m in members if m.get("balloon_source") == "yolov8seg"), members[-1])
        primary["testo_originale"] = " ".join(
            t for m in members if (t := m.get("testo_originale", "").strip())
        )
        redundant_ids.update(id(m) for m in members if m is not primary)

    if redundant_ids:
        data["detections"] = [d for d in data["detections"] if id(d) not in redundant_ids]

    for det in data["detections"]:
        det.pop("bubble_group_id", None)
        det.pop("_orig_y1", None)