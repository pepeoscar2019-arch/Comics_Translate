"""
Controllo di capienza sulle traduzioni, prima del render.

Il problema che risolve: l'italiano e' quasi sempre piu' lungo dell'inglese,
e su un balloon stretto questo si traduce in un font piu' piccolo o - nei
casi peggiori - in testo troncato da render.py (contenuto perso). Finora ce
ne si accorgeva solo guardando la tavola finita, e la correzione era un giro
completo "accorcia a mano -> ri-render".

Come funziona: per ogni balloon si misura la dimensione di font a cui il
testo entrerebbe (lo stesso fitting che usa render.py, senza disegnare
nulla), sia per il testo ORIGINALE sia per la TRADUZIONE. L'originale e' il
metro di paragone naturale: quel testo, in quel balloon, ci stava per
costruzione - e' quello che il disegnatore ci aveva messo. Se la traduzione
scende molto sotto quella dimensione, e' troppo lunga per lo spazio, e si
chiede al traduttore una resa piu' corta indicando quanti caratteri servono.

Non serve la pagina pulita (che in questa fase non esiste ancora): il
fitting lavora sulla maschera del balloon (bubble_seg) o, in mancanza, sul
rettangolo del bbox.
"""

import logging

from PIL import Image, ImageDraw

import paths
import render

log = logging.getLogger("pipeline.fit_check")

# Sotto questa frazione della dimensione di font dell'originale, la
# traduzione e' considerata troppo lunga per il balloon. 0.75 lascia passare
# il normale allungamento dell'italiano (~10-15% di caratteri, che costa
# poche unita' di font size) e intercetta le rese prolisse, quelle che poi
# fanno rimpicciolire l'intera tavola o vengono troncate.
DEFAULT_MIN_RATIO = 0.75

# Un balloon minuscolo (bollicine di sottofondo, "?!") ha misure instabili:
# poche parole cambiano la dimensione di font in modo brusco e il confronto
# con l'originale diventa rumore. Sotto quest'area non si interviene.
MIN_AREA_PX = 20_000

_measure_image = Image.new("RGB", (1, 1))
_measure_draw = ImageDraw.Draw(_measure_image)


def natural_font_size(det: dict, text: str, cfg: dict) -> int | None:
    """
    Dimensione di font a cui `text` entra nel balloon di `det`, con lo stesso
    fitting di render.py ma senza immagine di pagina: maschera reale del
    balloon quando c'e' (bubble_seg), altrimenti il rettangolo del bbox.

    Ritorna None se non e' misurabile (testo vuoto, bbox degenere).
    """
    text = (text or "").strip()
    if not text or not det.get("bbox"):
        return None

    render_cfg = cfg["rendering"]
    font_path = str(paths.resolve(det.get("font_path") or render_cfg["font_path"]))
    min_size = render_cfg["min_font_size"]
    max_size = render_cfg["max_font_size"]
    line_spacing = render_cfg["line_spacing"]
    bbox = det["bbox"]

    mask = None
    if det.get("balloon_source") == "yolov8seg" and det.get("mask_path"):
        mask = render._load_mask_array(det["mask_path"])

    if mask is not None:
        eroded = render._erode_mask(
            render._smooth_mask(mask), bbox, render_cfg.get("mask_margin_ratio", 0.05)
        )
        fitted = render._fit_text_to_mask(
            _measure_draw, render._normalize_punctuation(text), eroded, bbox,
            font_path, min_size, max_size, line_spacing,
            trust_single_lobe_shape=True,
        )
        if fitted is not None:
            return fitted[1].size

    # Nessuna maschera (didascalie rettangolari) o fitting a maschera non
    # riuscito: stessa via di ripiego di render.py, il rettangolo sicuro.
    # Qui non si puo' usare _measure_balloon_extent, che legge i pixel della
    # pagina pulita: in fase di traduzione non esiste ancora.
    x1, y1, x2, y2 = render._safe_area(bbox, render_cfg.get("safe_area_ratio", 0.75))
    _lines, font, _overflow = render._fit_text_to_box(
        _measure_draw, render._normalize_punctuation(text),
        max(1, x2 - x1), max(1, y2 - y1), font_path, min_size, max_size, line_spacing,
    )
    return font.size


def too_long(det: dict, cfg: dict, min_ratio: float = DEFAULT_MIN_RATIO) -> tuple[bool, int]:
    """
    Dice se la traduzione di `det` sta troppo stretta rispetto all'originale,
    e con quanti caratteri ci starebbe comodamente.

    Ritorna (troppo_lunga, caratteri_consigliati). Il numero consigliato viene
    dal fatto che l'area occupata da un testo cresce col quadrato della
    dimensione del font: per riportare la traduzione alla dimensione
    dell'originale servono circa `caratteri * (size_tradotto / size_originale)^2`
    caratteri.
    """
    originale = (det.get("testo_originale") or "").strip()
    tradotto = (det.get("testo_tradotto") or "").strip()
    if not originale or not tradotto:
        return False, 0

    x1, y1, x2, y2 = det["bbox"]
    if (x2 - x1) * (y2 - y1) < MIN_AREA_PX:
        return False, 0

    size_orig = natural_font_size(det, originale, cfg)
    size_trad = natural_font_size(det, tradotto, cfg)
    if not size_orig or not size_trad:
        return False, 0

    ratio = size_trad / size_orig
    if ratio >= min_ratio:
        return False, 0

    # L'area occupata da un testo cresce col quadrato della dimensione del
    # font: per riportare la traduzione alla dimensione dell'originale
    # servono circa questi caratteri.
    target = int(len(tradotto) * (ratio ** 2))

    # Due pavimenti, perche' un obiettivo troppo aggressivo non produce una
    # resa piu' asciutta ma una battuta mutilata (in prova, "20 caratteri"
    # ha prodotto un'accozzaglia senza senso):
    #  - il testo di partenza: la traduzione non puo' dire meno di quello che
    #    dice l'originale, quindi non si scende sotto la sua lunghezza;
    #  - il 60% della traduzione attuale: oltre quella soglia non si tratta
    #    piu' di essere concisi ma di buttare via contenuto.
    target = max(target, int(len(originale) * 0.9), int(len(tradotto) * 0.6))
    return True, target


def shorten_page(data: dict, cfg: dict, shorten_fn) -> int:
    """
    Passa in rassegna i balloon della pagina e, per quelli la cui traduzione
    non ci sta, chiede una resa piu' corta a `shorten_fn(testo, max_caratteri,
    originale) -> str`. La versione accorciata viene accettata solo se
    migliora davvero la dimensione di font: se il modello risponde con
    qualcosa di piu' lungo, o che non entra comunque meglio, si tiene la
    traduzione di partenza (meglio un font piccolo che una battuta
    stravolta a vuoto).

    Ritorna il numero di balloon effettivamente accorciati.
    """
    fit_cfg = cfg.get("fit_check", {}) or {}
    if not fit_cfg.get("enabled", True):
        return 0
    min_ratio = float(fit_cfg.get("min_ratio", DEFAULT_MIN_RATIO))

    shortened = 0
    for det in data.get("detections", []):
        try:
            over, target = too_long(det, cfg, min_ratio)
        except Exception as e:
            log.debug(f"balloon {det.get('balloon_id')}: controllo capienza saltato ({e})")
            continue
        if not over:
            continue

        original_text = det["testo_tradotto"]
        bid = det.get("balloon_id")
        log.info(
            f"  balloon {bid}: traduzione troppo lunga per il balloon "
            f"({len(original_text)} caratteri, ne entrano ~{target}), chiedo una resa piu' corta"
        )
        try:
            shorter = (shorten_fn(original_text, target, det.get("testo_originale", "")) or "").strip()
        except Exception as e:
            log.warning(f"  balloon {bid}: richiesta di accorciamento fallita ({e}), tengo la traduzione lunga")
            continue

        if not shorter or shorter == original_text:
            continue

        # Rete di sicurezza sul contenuto: una resa che sta a meno della meta'
        # del testo di partenza non e' piu' concisa, ha buttato via qualcosa.
        # Meglio un font piccolo che una battuta che non dice piu' la stessa
        # cosa - questa e' una scorciatoia automatica, non una revisione.
        source_len = len((det.get("testo_originale") or "").strip())
        if source_len and len(shorter) < source_len * 0.5:
            log.warning(
                f"  balloon {bid}: la versione corta ({len(shorter)} caratteri) e' molto piu' breve "
                f"dell'originale ({source_len}), probabile perdita di contenuto: la scarto"
            )
            continue

        before = natural_font_size(det, original_text, cfg)
        after = natural_font_size(det, shorter, cfg)
        if not after or not before or after <= before:
            log.info(f"  balloon {bid}: la versione corta non migliora la resa, tengo l'originale")
            continue

        log.info(f"  balloon {bid}: accorciato a {len(shorter)} caratteri (font {before} -> {after}): \"{shorter[:60]}\"")
        det["testo_tradotto"] = shorter
        det["_shortened_for_fit"] = True
        shortened += 1

    return shortened
