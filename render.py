import functools
import json
import logging
import math
import re
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pyphen
from PIL import Image, ImageColor, ImageDraw, ImageFont

import balloon_shape
import clean
import font_analyzer
import paths

log = logging.getLogger("pipeline.render")

_CLEANED_IMAGE_CACHE_MAX = 6
_cleaned_image_cache: "OrderedDict[str, tuple[float, Image.Image]]" = OrderedDict()
_cleaned_image_cache_lock = threading.Lock()


def _load_cleaned_image_cached(path: Path) -> Image.Image:
    """
    Carica cleaned.png in RGB, tenendo in cache le ultime pagine aperte
    (chiave: percorso, invalidata sul mtime del file). L'anteprima live del
    singolo balloon in Revisione richiama questa funzione ad ogni tocco di
    tastiera/trascinamento (ogni 220ms): senza cache, ogni richiesta
    rileggerebbe e decodificherebbe da zero l'intera pagina da disco — il
    costo dominante, molto piu' di tutta la logica di fitting/posizionamento
    del testo — introducendo una latenza percepibile proprio nell'interazione
    che dovrebbe sembrare istantanea. Limitata alle ultime N pagine (LRU) per
    non far crescere la memoria illimitatamente in una sessione lunga.
    """
    key = str(path)
    mtime = path.stat().st_mtime
    with _cleaned_image_cache_lock:
        cached = _cleaned_image_cache.get(key)
        if cached is not None and cached[0] == mtime:
            _cleaned_image_cache.move_to_end(key)
            return cached[1].copy()
        image = Image.open(path).convert("RGB")
        _cleaned_image_cache[key] = (mtime, image)
        _cleaned_image_cache.move_to_end(key)
        while len(_cleaned_image_cache) > _CLEANED_IMAGE_CACHE_MAX:
            _cleaned_image_cache.popitem(last=False)
        return image.copy()

_hyphenator = pyphen.Pyphen(lang="it_IT")

_ELLIPSIS_RE = re.compile(r'\.{3,}')

_SHOUT_PUNCT_RE = re.compile(r'[!?]{2,}')


def _is_shout_text(text: str) -> bool:
    """Nei fumetti reali il corpo del testo mantiene una dimensione font
    uniforme su tutta la tavola/il fumetto; fanno eccezione solo i balloon
    di urlo/stupore (punteggiatura multipla tipo "!!", "?!", "!?"), che sono
    tipicamente resi piu' grandi. Questi balloon restano fuori dal calcolo
    della dimensione comune e mantengono il fitting automatico libero."""
    return bool(_SHOUT_PUNCT_RE.search(text))


def _normalize_punctuation(text: str) -> str:
    """Convenzioni tipografiche italiane: puntini di sospensione come
    carattere singolo "…", non tre punti "...". I punti esclamativi/
    interrogativi multipli non vengono toccati: nel fumetto sono
    espressivi (urla, enfasi), non un refuso da normalizzare."""
    return _ELLIPSIS_RE.sub('…', text)


@functools.lru_cache(maxsize=4096)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Cache dei font caricati: costruire una FreeTypeFont e' un'operazione
    non gratuita (parsing del file, creazione della face freetype a quella
    dimensione), e _fit_text_to_box la richiama per ogni dimensione provata
    nella ricerca (fino a un centinaio di misure per singolo balloon)."""
    font = ImageFont.truetype(font_path, size)
    # Chiave stabile per _measure_text: evita di mettere in cache le misure
    # per identita' dell'oggetto font (id() puo' essere riciclato da un altro
    # oggetto dopo il garbage collection, corrompendo silenziosamente la
    # cache), usando invece font_path+size, sempre validi e univoci.
    font._measure_key = (font_path, size)
    return font


# Draw "vuoto" (nessuna immagine reale dietro) dedicato alla sola misura del
# testo: textbbox calcola le metriche dal font, non legge/scrive pixel, quindi
# non serve un'immagine di dimensioni reali per misurare.
_measure_draw = ImageDraw.Draw(Image.new("L", (1, 1)))


# Campi manuali di formattazione impostabili in Revisione (vedi
# revOpenEditor in web_app.py): se presenti su un det, il rendering usa
# sempre il percorso "manuale" (_fit_text_to_box + _draw_lines_centered)
# invece del fitting a maschera complesso, per lo stesso motivo per cui
# manual_balloon_shape=="ellipse" forza gia' oggi manual_box=True.
_MANUAL_STYLE_KEYS = (
    "manual_font_size", "manual_line_spacing", "manual_align",
    "manual_bold", "manual_italic", "manual_underline", "manual_outline_enable",
)


def _has_manual_style(det: dict) -> bool:
    return any(det.get(k) for k in _MANUAL_STYLE_KEYS)


def _resolve_italic_font_path(font_path: str) -> str | None:
    """Cerca un file font corsivo dedicato accanto a quello scelto (stessa
    convenzione di naming gia' presente in fonts/, es. animeace2_reg.ttf ->
    animeace2_ital.ttf): un vero font corsivo ha metriche/glifi disegnati
    apposta, molto meglio di uno shear sintetico. Ritorna None se non trovato
    (il chiamante ricade sullo shear)."""
    p = Path(font_path)
    stem = p.stem
    candidates = []
    if "_reg" in stem:
        candidates.append(stem.replace("_reg", "_ital"))
        candidates.append(stem.replace("_reg", "_italic"))
    candidates.append(stem + "_ital")
    candidates.append(stem + "_italic")
    for cand in candidates:
        cand_path = p.with_name(cand + p.suffix)
        if cand_path.exists():
            return str(cand_path)
    return None


_ITALIC_SHEAR = 0.22


def _draw_line_italic_shear(image, x, y, line, font, fill_color, stroke_width, stroke_fill):
    """Inclina il testo via trasformazione affine quando non esiste un file
    font corsivo dedicato per il font scelto: PIL non supporta uno shear
    diretto nel rendering dei glifi, quindi la riga va prima disegnata su un
    layer separato e poi incollata deformata sull'immagine."""
    bbox = _textbbox(line, font)
    pad = stroke_width + 2
    w = int(bbox[2] - bbox[0]) + pad * 2
    h = int(bbox[3] - bbox[1]) + pad * 2
    if w <= 0 or h <= 0:
        return
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ldraw = ImageDraw.Draw(layer)
    ldraw.text((pad - bbox[0], pad - bbox[1]), line, font=font, fill=fill_color,
               stroke_width=stroke_width, stroke_fill=stroke_fill)
    xshift = _ITALIC_SHEAR * h
    new_w = w + int(round(xshift))
    sheared = layer.transform(
        (new_w, h), Image.AFFINE, (1, -_ITALIC_SHEAR, xshift, 0, 1, 0), resample=Image.BICUBIC,
    )
    image.paste(sheared, (int(x) - pad, int(y) - pad), sheared)


def _draw_line_styled(image, draw, x, y, line, font, fill_color, style: dict):
    """Disegna una riga applicando contorno/grassetto finto (stroke nativo
    PIL) e, se serve, lo shear sintetico del corsivo. Il contorno, se
    attivo, ha priorita' sul grassetto finto: userebbero comunque lo stesso
    meccanismo di stroke, e un contorno esplicito e' una scelta di colore
    dell'utente che non va sovrascritta dal grassetto."""
    stroke_width = 0
    stroke_fill = None
    if style.get("outline_enable") and style.get("outline_color"):
        stroke_width = max(1, round(style.get("outline_width") or 1))
        stroke_fill = ImageColor.getrgb(style["outline_color"])
    elif style.get("bold"):
        stroke_width = 1
        stroke_fill = fill_color

    if style.get("italic_shear"):
        _draw_line_italic_shear(image, x, y, line, font, fill_color, stroke_width, stroke_fill)
    else:
        draw.text((x, y), line, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_fill)


@functools.lru_cache(maxsize=200_000)
def _measure_text(font_key: tuple[str, int], text: str) -> tuple[int, int, int, int]:
    font = _load_font(*font_key)
    return _measure_draw.textbbox((0, 0), text, font=font)


def _textbbox(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    """Sostituto cacheable di draw.textbbox((0, 0), text, font=font): la
    ricerca della dimensione font piu' grande che entra nel box rimisura le
    stesse combinazioni (font, size, testo) decine di migliaia di volte (una
    volta per ogni dimensione di font provata, per ogni box che si adatta in
    piu' righe) — la vera causa della lentezza dell'anteprima live, non il
    caricamento della pagina. La misura dipende solo da font e testo, mai
    dall'immagine/Draw da cui viene chiamata, quindi e' sicura da mettere in
    cache: stesso identico risultato di draw.textbbox, calcolato una sola
    volta per combinazione."""
    key = getattr(font, "_measure_key", None)
    if key is None:
        return _measure_draw.textbbox((0, 0), text, font=font)
    return _measure_text(key, text)


def _split_word_to_fit(draw: ImageDraw.ImageDraw, word: str, font: ImageFont.FreeTypeFont, max_width: int) -> tuple[str, str] | None:
    """
    Prova a spezzare `word` su un punto di sillabazione in modo che la prima
    parte (con trattino) entri in max_width. Ritorna (parte1-, resto) o None
    se la parola non è sillabile o nessun punto di spezzatura entra nel box.
    """
    positions = _hyphenator.positions(word)
    if not positions:
        return None
    best = None
    for pos in positions:
        candidate = word[:pos] + "-"
        bbox = _textbbox(candidate, font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            best = pos
        else:
            break
    if best is None:
        return None
    return word[:best] + "-", word[best:]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    hyphenate: bool = True,
) -> list[str]:
    """Spezza il testo in righe che stanno dentro max_width, sillabando le parole troppo lunghe."""
    words = text.split()
    lines = []
    current_line = ""

    i = 0
    while i < len(words):
        word = words[i]
        candidate = f"{current_line} {word}".strip()
        bbox = _textbbox(candidate, font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current_line = candidate
            i += 1
            continue

        if current_line:
            lines.append(current_line)
            current_line = ""
            continue

        # La parola da sola non entra nemmeno a inizio riga: prova a sillabarla.
        split = _split_word_to_fit(draw, word, font, max_width) if hyphenate else None
        if split is None:
            # Non sillababile, sillabazione disattivata, o nessun pezzo entra:
            # la mettiamo comunque per intero, andra' in overflow (gestito
            # da _fit_text_to_box, che nel frattempo prova font piu' piccoli).
            lines.append(word)
            i += 1
            continue

        head, tail = split
        lines.append(head)
        words[i] = tail

    if current_line:
        lines.append(current_line)

    return lines


def _wrap_text_balanced(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str] | None:
    """
    Come _wrap_text, ma invece di riempire greedy ogni riga finche' entra
    (che spesso lascia una parola sola e corta sull'ultima riga, una
    "vedova"), sceglie i punti di a-capo che minimizzano lo squilibrio tra
    righe (somma degli scarti al quadrato dalla larghezza massima). Il
    risultato e' un blocco visivamente piu' bilanciato, senza righe molto
    piu' corte delle altre. Programmazione dinamica O(n^2) sulle parole,
    accettabile per il numero di parole tipico di un balloon.

    Si applica solo quando ogni parola entra da sola in max_width: se anche
    solo una parola e' piu' larga del box (servirebbe sillabarla), ritorna
    None e il chiamante ricade su _wrap_text (che gestisce la sillabazione).

    Il numero di righe usato e' vincolato a quello che userebbe un wrap
    greedy (provatamente minimo per il word-wrap classico): senza questo
    vincolo la DP potrebbe preferire piu' righe piu' corte pur di ridurre
    lo sbilanciamento, aumentando l'altezza del blocco e facendolo
    "scoppiare" fuori dal balloon anche quando il greedy ci starebbe.
    """
    words = text.split()
    if not words:
        return []

    n = len(words)
    # width_cache[i][j] = larghezza di "words[i:j]" unite da spazi
    width_cache: dict[tuple[int, int], int] = {}

    def line_width(i: int, j: int) -> int:
        key = (i, j)
        if key not in width_cache:
            line = " ".join(words[i:j])
            bbox = _textbbox(line, font)
            width_cache[key] = bbox[2] - bbox[0]
        return width_cache[key]

    for idx in range(n):
        if line_width(idx, idx + 1) > max_width:
            return None

    target_lines = len(_wrap_text(draw, text, font, max_width, hyphenate=False))

    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(target_lines + 1)]
    breaks = [[0] * (n + 1) for _ in range(target_lines + 1)]
    dp[0][0] = 0.0

    for k in range(1, target_lines + 1):
        for j in range(k, n + 1):
            for i in range(j - 1, k - 2, -1):
                w = line_width(i, j)
                if w > max_width:
                    break  # righe piu' lunghe (i minore) saranno anch'esse troppo larghe
                if dp[k - 1][i] == INF:
                    continue
                cost = dp[k - 1][i] + (max_width - w) ** 2
                if cost < dp[k][j]:
                    dp[k][j] = cost
                    breaks[k][j] = i

    if dp[target_lines][n] == INF:
        # Non dovrebbe succedere (target_lines viene da un wrap greedy gia'
        # valido), ma per sicurezza ricadi sul greedy invece di propagare
        # un errore.
        return _wrap_text(draw, text, font, max_width, hyphenate=False)

    lines = []
    j, k = n, target_lines
    while k > 0:
        i = breaks[k][j]
        lines.append(" ".join(words[i:j]))
        j = i
        k -= 1
    lines.reverse()
    return lines


def _fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
    font_path: str,
    min_size: int,
    max_size: int,
    line_spacing: float,
) -> tuple[list[str], ImageFont.FreeTypeFont, bool]:
    """
    Trova la dimensione font più grande (tra min e max) per cui il testo
    wrappato entra nel box. Se anche min_size overflow, ritorna comunque
    min_size (overflow gestito a valle, es. troncamento o log warning).

    A parita' di risultato preferisce non sillabare: prova prima tutte le
    dimensioni con il wrap "normale" (spezzato solo sugli spazi), e ricorre
    alla sillabazione delle singole parole solo se nessuna dimensione va
    bene senza — evita di spezzare inutilmente parole/esclamazioni brevi
    quando basterebbe un font leggermente piu' piccolo su una riga sola.
    Il wrap "normale" usa _wrap_text_balanced (righe bilanciate, niente
    vedove/orfane) quando possibile, con fallback al greedy _wrap_text.

    Ritorna (righe, font, overflowed): overflowed e' True se il testo e'
    stato troncato pur di non farlo uscire dal balloon — il chiamante puo'
    usarlo per segnalare che la traduzione andrebbe accorciata.
    """
    for hyphenate in (False, True):
        for size in range(max_size, min_size - 1, -1):
            font = _load_font(font_path, size)
            if hyphenate:
                lines = _wrap_text(draw, text, font, box_width, hyphenate=True)
            else:
                lines = _wrap_text_balanced(draw, text, font, box_width)
                if lines is None:
                    lines = _wrap_text(draw, text, font, box_width, hyphenate=False)

            line_height = size * line_spacing
            total_height = line_height * len(lines)
            max_line_width = max(
                (_textbbox(line, font)[2] for line in lines),
                default=0,
            )

            if total_height <= box_height and max_line_width <= box_width:
                return lines, font, False

    font = _load_font(font_path, min_size)
    lines = _wrap_text(draw, text, font, box_width)
    log.warning(f"Overflow testo anche a font {min_size}px: \"{text[:40]}...\"")

    # Tronca le righe finché entrano in altezza, per evitare che escano dal balloon
    line_height = min_size * line_spacing
    max_lines = max(1, int(box_height / line_height))
    if len(lines) > max_lines:
        log.warning(f"  -> tronco da {len(lines)} a {max_lines} righe")
        lines = lines[:max_lines]

    # Tronca anche in larghezza le righe (es. parole non spezzabili piu' larghe del box)
    for i, line in enumerate(lines):
        bbox_test = _textbbox(line, font)
        if bbox_test[2] - bbox_test[0] <= box_width:
            continue
        truncated = line
        while len(truncated) > 3:
            truncated = truncated[:-1]
            bbox_test = _textbbox(truncated + "…", font)
            if bbox_test[2] - bbox_test[0] <= box_width:
                lines[i] = truncated + "…"
                break
        else:
            lines[i] = truncated

    return lines, font, True


def _measure_balloon_extent(
    image: Image.Image, bbox: list[int], dark_threshold: int = 110, inset: int = 10,
    center: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """
    Misura l'estensione reale del balloon "sparando" 4 raggi dal centro del
    bbox verso i 4 lati sull'immagine gia' pulita, fermandosi al contorno
    scuro. Il balloon reale non riempie mai una percentuale fissa del bbox
    di detection (varia da balloon a balloon, per forma e stile), quindi
    questa misura e' molto piu' affidabile di un rapporto costante — evita
    sia il testo che sconfina (rapporto troppo generoso per un balloon
    stretto) sia il testo inutilmente piccolo (rapporto troppo prudente per
    un balloon che riempie quasi tutto il bbox).
    Ritorna None se la misura sembra degenere (es. il centro del bbox cade
    gia' su un pixel scuro, balloon con sfondo non chiaro).
    """
    x1, y1, x2, y2 = bbox
    w_img, h_img = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    # `center`, quando fornito dal chiamante, e' il centro del corpo del
    # balloon ricavato dalla maschera (_mask_body_center): piu' affidabile
    # del centro del bbox, che comprende anche la coda.
    ccx, ccy = center if center else ((x1 + x2) // 2, (y1 + y2) // 2)
    cx = max(x1, min(x2 - 1, ccx))
    cy = max(y1, min(y2 - 1, ccy))
    cx = max(0, min(w_img - 1, cx))
    cy = max(0, min(h_img - 1, cy))

    gray = image.convert("L")
    px = gray.load()
    if px[cx, cy] < dark_threshold:
        return None

    def is_dark(x: int, y: int) -> bool:
        return px[x, y] < dark_threshold

    def cast(sx1: int, sy1: int, sx2: int, sy2: int) -> tuple[int, int, int, int]:
        left = cx
        while left > sx1 and not is_dark(left, cy):
            left -= 1
        right = cx
        while right < sx2 - 1 and not is_dark(right, cy):
            right += 1
        top = cy
        while top > sy1 and not is_dark(cx, top):
            top -= 1
        bottom = cy
        while bottom < sy2 - 1 and not is_dark(cx, bottom):
            bottom += 1
        return left, top, right, bottom

    def apply_inset(left, top, right, bottom):
        # L'inset e' un margine di sicurezza dal bordo scuro del balloon, ma
        # per balloon piccoli (es. piccole bolle "coda") un inset fisso puo'
        # mangiarsi quasi tutto lo spazio misurato, forzando il testo a un
        # font minuscolo con overflow verticale. Scala l'inset in proporzione
        # all'estensione trovata su ciascun asse, non oltre il valore fisso.
        inset_x = min(inset, max(0, (right - left) * 0.2))
        inset_y = min(inset, max(0, (bottom - top) * 0.2))
        return left + inset_x, top + inset_y, right - inset_x, bottom - inset_y

    left, top, right, bottom = cast(x1, y1, x2, y2)
    fleft, ftop, fright, fbottom = apply_inset(left, top, right, bottom)

    # Nella stragrande maggioranza dei casi il bbox di detection e' gia' un
    # buon proxy dell'area utile (il raggio esce dal bbox senza incontrare
    # un bordo scuro affidabile — contorno sottile/antialiasing — anche per
    # balloon disegnati correttamente): usare quel risultato e' sicuro.
    # Espandiamo la ricerca oltre il bbox solo come ripiego, quando il
    # risultato e' palesemente degenere (bbox di detection molto piu'
    # piccolo del balloon reale, come una piccola bolla "coda"): altrimenti
    # rischiamo di far scappare il raggio nello sfondo, oltre il vero bordo
    # del balloon, gonfiando la misura e facendo sconfinare il testo.
    degenerate = (fright - fleft) < 20 or (fbottom - ftop) < 20
    if degenerate:
        pad_x = min(150, max((x2 - x1), 40))
        pad_y = min(150, max((y2 - y1), 40))
        sx1, sx2 = max(0, x1 - pad_x), min(w_img - 1, x2 + pad_x)
        sy1, sy2 = max(0, y1 - pad_y), min(h_img - 1, y2 + pad_y)
        eleft, etop, eright, ebottom = cast(sx1, sy1, sx2, sy2)
        efleft, eftop, efright, efbottom = apply_inset(eleft, etop, eright, ebottom)
        if (efright - efleft) >= 20 and (efbottom - eftop) >= 20:
            fleft, ftop, fright, fbottom = efleft, eftop, efright, efbottom

    if fright - fleft < 8 or fbottom - ftop < 8:
        return None
    return int(fleft), int(ftop), int(fright), int(fbottom)


def _safe_area(bbox: list[int], safe_area_ratio: float) -> tuple[int, int, int, int]:
    """
    Restringe il bbox rettangolare (dalla detection) verso un'area
    "sicura" più piccola, per tenere conto del fatto che il balloon reale
    e' ovale/tondo e non riempie tutto il rettangolo che lo contiene.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    new_w = w * safe_area_ratio
    new_h = h * safe_area_ratio

    cx = x1 + w / 2
    cy = y1 + h / 2

    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)

    return new_x1, new_y1, new_x2, new_y2


_MIN_MASK_PIXELS_FOR_COLOR = 200
"""Sotto questo numero di pixel la media dentro la maschera e' rumore: si
torna a misurare sul bbox intero (vedi _detect_text_color)."""


def _detect_text_color(
    image: Image.Image, bbox: list[int], mask: np.ndarray | None = None
) -> tuple[int, int, int]:
    """
    Sceglie nero o bianco per il testo in base a quanto e' chiaro lo sfondo
    su cui finira'. Serve per i balloon a fondo scuro (flashback, urla,
    riquadri neri), dove il nero di default sarebbe illeggibile.

    La misura va fatta DENTRO la maschera del balloon quando c'e': il bbox e'
    un rettangolo, e su un balloon tondo - o su un lobo di un balloon doppio -
    gli angoli contengono lo sfondo della vignetta. Se quello sfondo e' scuro
    la media del rettangolo scende sotto la soglia anche con l'interno del
    balloon chiarissimo, e il testo viene scritto in bianco su bianco: e'
    successo sul lobo inferiore del balloon doppio a pagina 005 di
    "An Unconventional Couple 2", dove la maschera copriva il 54% del bbox
    (media nel bbox 127, dentro la maschera 191).
    """
    x1, y1, x2, y2 = bbox
    gray = np.array(image.crop((x1, y1, x2, y2)).convert("L"))
    if gray.size == 0:
        return (0, 0, 0)

    dentro = None
    if mask is not None:
        finestra = mask[max(0, y1):y2, max(0, x1):x2]
        if finestra.shape == gray.shape:
            dentro = gray[finestra > 0]

    if dentro is not None and dentro.size >= _MIN_MASK_PIXELS_FOR_COLOR:
        media = float(dentro.mean())
    else:
        media = float(gray.mean())

    if media < 128:
        return (255, 255, 255)  # sfondo scuro -> testo bianco
    return (0, 0, 0)            # sfondo chiaro -> testo nero


def _load_mask_array(mask_path: str) -> np.ndarray | None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def _box_coverage(box: list[int], other: list[int]) -> float:
    """Quanta parte di `other` e' coperta da `box` (0..1)."""
    ix1, iy1 = max(box[0], other[0]), max(box[1], other[1])
    ix2, iy2 = min(box[2], other[2]), min(box[3], other[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1, (other[2] - other[0]) * (other[3] - other[1]))


def _swallows_neighbour(old_box: list[int], new_box: list[int], neighbours: list[list[int]]) -> bool:
    """Dice se l'espansione ha inghiottito un balloon vicino.

    Serve a riconoscere le fughe del flood fill che `exclude` non ferma: la
    maschera di un vicino copre solo cio' che bubble_seg ha segmentato di
    lui, e se quel vicino e' sotto-segmentato (gli manca la calotta) resta
    una fascia del suo interno, dello stesso bianco, da cui il riempimento
    passa e dilaga dentro tutto il balloon accanto.

    Il criterio guarda al risultato, non al percorso: se dopo l'espansione il
    box copre meta' o piu' di un ALTRO balloon, e prima non lo faceva, quel
    recupero e' andato dove non doveva. Un recupero legittimo (la calotta o
    la coda che mancavano) non arriva a coprire mezzo balloon vicino.
    """
    for other in neighbours:
        prima = _box_coverage(old_box, other)
        dopo = _box_coverage(new_box, other)
        if dopo >= 0.5 and dopo - prima >= 0.3:
            return True
    return False


def _expand_mask_to_drawn_balloon(
    image: Image.Image, mask: np.ndarray, bbox: list[int], color_tolerance: int = 30,
    exclude: np.ndarray | None = None, neighbours: list[list[int]] | None = None,
) -> np.ndarray:
    """
    La maschera di bubble_seg puo' sotto-segmentare un balloon (es. manca la
    calotta superiore di un'ellisse): centrare il testo su quella maschera
    lo sposta verso il basso rispetto al balloon REALMENTE disegnato nella
    pagina. Qui si ricostruisce la vera sagoma con un flood fill dei pixel
    simili per colore a quello campionato al centro del bbox di detection
    sulla pagina gia' pulita: se il risultato e' piu' ampio della maschera
    del modello (in altezza o larghezza), lo si usa al suo posto per il
    fitting.

    Il criterio e' la similitudine di colore al pixel centrale, non la sola
    luminosita' (era cosi' in precedenza, con soglia "chiaro" >=200): un
    balloon a tinta piena ma colorata (verde, marroncino, ecc.) ha spesso
    luminosita' ben sotto 200 pur essendo comunque un fondo piatto
    riempibile via flood fill. Con la sola soglia di luminosita' la
    recupero non scattava mai su questi balloon, lasciando visibile un
    riempimento piu' piccolo del contorno realmente disegnato ("balloon nel
    balloon").

    Il flood fill puo' "sfondare" oltre il vero contorno se il bordo del
    balloon e' troppo sottile/antialiasato in un punto (es. sconfina nello
    sfondo circostante): per questo il risultato viene accettato solo se
    resta entro un multiplo ragionevole delle dimensioni della maschera
    originale, altrimenti si ricade su quella del modello.

    Due balloon distinti disegnati a contatto (nessun bordo scuro tra loro,
    es. dialoghi consecutivi dello stesso personaggio) hanno lo stesso
    colore di fondo continuo: il flood fill di uno sconfina nell'altro. Se
    `exclude` e' passata (unione delle maschere RAW degli altri balloon
    della pagina, gia' dilatata di un margine), quell'area viene tolta dal
    risultato del flood fill prima di unirla alla maschera originale, cosi'
    il recupero non invade un balloon vicino.
    """
    x1, y1, x2, y2 = bbox
    w, h = image.size
    cx = max(0, min(w - 1, (x1 + x2) // 2))
    cy = max(0, min(h - 1, (y1 + y2) // 2))

    rgb = np.array(image.convert("RGB")).astype(np.int16)
    seed_color = rgb[cy, cx]
    color_diff = np.abs(rgb - seed_color).sum(axis=2)
    similar = (color_diff <= color_tolerance).astype(np.uint8) * 255
    ff_flags = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(similar, ff_flags, (cx, cy), 128)
    region = (similar == 128).astype(np.uint8) * 255
    if exclude is not None:
        region = cv2.bitwise_and(region, cv2.bitwise_not(exclude))

    mys, mxs = np.where(mask > 0)
    rys, rxs = np.where(region > 0)
    if len(rys) == 0 or len(mys) == 0:
        return mask

    mask_w, mask_h = mxs.max() - mxs.min(), mys.max() - mys.min()

    # Non si sceglie tra maschera e regione: si uniscono. Il flood fill puo'
    # recuperare la cima persa dalla segmentazione ma perdere un pezzo su un
    # altro lato (es. la coda, dove il contorno e' piu' sottile e un pixel
    # sotto soglia lo puo' interrompere) — unendole si tiene il meglio di
    # entrambe senza dover scegliere una sola direzione da correggere.
    union = cv2.bitwise_or(mask, region)
    uys, uxs = np.where(union > 0)
    union_w, union_h = uxs.max() - uxs.min(), uys.max() - uys.min()

    # L'unione non e' piu' ampia della maschera del modello: niente da
    # recuperare (il flood fill non ha trovato nulla fuori dalla maschera).
    if union_w <= mask_w * 1.02 and union_h <= mask_h * 1.02:
        return mask
    # Troppo piu' ampia per essere plausibile: probabile fuga nello sfondo,
    # meglio tenere la maschera originale che rischiare di sconfinare.
    if union_w > mask_w * 4 or union_h > mask_h * 4:
        return mask

    # Fuga dentro un balloon vicino: il limite qui sopra non la intercetta
    # (restare entro 4x e' facile quando il vicino e' attaccato), ma il box
    # risultante inghiotte l'altro balloon. Vedi _swallows_neighbour.
    if neighbours:
        new_box = [int(uxs.min()), int(uys.min()), int(uxs.max()) + 1, int(uys.max()) + 1]
        # Il confronto parte dal bbox della MASCHERA, non da `bbox`: quello
        # arriva da translated.json, dove un render precedente puo' aver gia'
        # salvato un'espansione sbagliata - e allora il vicino risulterebbe
        # "gia' dentro da prima" e la guardia non scatterebbe mai, lasciando
        # l'errore incollato alla pagina per sempre. Partendo dalla maschera
        # il controllo e' sempre lo stesso, e il bbox salvato si ricorregge
        # da solo al primo render.
        mask_box = [int(mxs.min()), int(mys.min()), int(mxs.max()) + 1, int(mys.max()) + 1]
        if _swallows_neighbour(mask_box, new_box, neighbours):
            return mask

    return union


_MASK_SMOOTH_PX = 9  # stessa scala di detection.mask_dilation_px: le maschere del
# text detector sono generate a bassa risoluzione e poi riscalate, quindi hanno
# bordi "a blocchi" con piccole insenature spurie. Una chiusura morfologica con
# kernel fisso (non proporzionale al bbox) le smussa senza cancellare vere
# strozzature strutturali (es. balloon a clessidra), che sono molto piu' estese.


def _smooth_mask(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((_MASK_SMOOTH_PX * 2 + 1, _MASK_SMOOTH_PX * 2 + 1), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


_BALLOON_TAIL_ANGLES_DEG = {
    "n": -90, "ne": -45, "e": 0, "se": 45,
    "s": 90, "sw": 135, "w": 180, "nw": -135,
}
"""Direzione (bussola) -> angolo in gradi, 0=destra, 90=in basso (asse y
verso il basso come nelle immagini)."""


def _add_balloon_tail(mask: np.ndarray, bbox: list[int], direction: str | None) -> None:
    """Aggiunge alla maschera un codino triangolare rivolto verso
    'direction', unito al bordo dell'ellisse: un'ellisse piena da sola
    somiglia a un fumetto di pensiero/didascalia, non a un vero speech
    balloon. Il triangolo tocca il bordo dell'ellisse cosi' il contorno
    (bordo nero) e il riempimento seguono entrambi la sagoma combinata."""
    angle_deg = _BALLOON_TAIL_ANGLES_DEG.get(direction or "")
    if angle_deg is None:
        return
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax, ay = max(1.0, (x2 - x1) / 2), max(1.0, (y2 - y1) / 2)

    theta = math.radians(angle_deg)
    spread = math.radians(14)

    def _boundary_point(t: float) -> tuple[float, float]:
        return (cx + ax * math.cos(t), cy + ay * math.sin(t))

    base1 = _boundary_point(theta - spread)
    base2 = _boundary_point(theta + spread)
    bx, by = _boundary_point(theta)
    tail_len = 0.6 * min(ax, ay)
    tip = (bx + tail_len * math.cos(theta), by + tail_len * math.sin(theta))

    pts = np.array([base1, base2, tip], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)


def _apply_manual_fill(image: Image.Image, det: dict) -> None:
    """Sovrascrive l'area del balloon con un colore scelto in Revisione e/o le
    disegna un bordo nero, per i casi in cui la pulizia automatica (ComfyUI)
    lascia un risultato inutilizzabile e serve una correzione manuale rapida."""
    fill_hex = det.get("manual_fill_color")
    border = bool(det.get("manual_border"))
    if not fill_hex and not border:
        return

    h, w = image.height, image.width
    mask_path = det.get("mask_path")
    manual_box = bool(det.get("manual_text_box"))
    # mask_path esiste anche quando arriva da comic-text-detector, che segue
    # solo il testo (bbox dilatato), non la vera sagoma del balloon: quella
    # reale (ovale, con coda, ecc.) e' disponibile solo se bubble_seg ha
    # trovato una corrispondenza (vedi detect.py:_merge_with_bubble_shapes).
    # Riusare la maschera "solo testo" per il riempimento manuale darebbe
    # una toppa squadrata invece che un balloon, quindi va trattata come
    # "nessuna sagoma reale disponibile" e finire nel fallback a ellisse.
    has_real_balloon_shape = mask_path and det.get("balloon_source") == "yolov8seg"
    x1, y1, x2, y2 = det["bbox"]
    if det.get("manual_balloon_shape") == "ellipse":
        # Richiesta esplicita dal pulsante "Ricrea balloon" in Revisione:
        # vince sempre su tutto il resto, incluso un bbox ridimensionato a
        # mano (manual_text_box), perche' allargare il box per farlo
        # combaciare con l'area del balloon sparito e' proprio il primo
        # passo tipico di quel flusso, e non deve far ripiegare sul
        # rettangolo pieno sotto.
        shape_mask = np.zeros((h, w), np.uint8)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))
        cv2.ellipse(shape_mask, center, axes, 0, 0, 360, 255, thickness=-1)
        _add_balloon_tail(shape_mask, det["bbox"], det.get("manual_balloon_tail"))
    elif has_real_balloon_shape and not manual_box:
        # Segue la sagoma reale del balloon solo quando il box NON e' stato
        # toccato a mano: extend_to_bbox di clean.py limita comunque
        # l'estensione a una fascia vicina alla vecchia maschera (pensata per
        # non far uscire l'inpainting dai bordi ovali di un balloon), quindi
        # su un box ridimensionato a mano non copre tutto il rettangolo
        # verde. Se l'utente ha ridimensionato il box, vuole esattamente
        # quel rettangolo riempito: si usa il fallback sotto.
        shape_mask = clean._resolve_balloon_mask(mask_path, (h, w), det["bbox"], extend_to_bbox=False)
        shape_mask = _smooth_mask(shape_mask)
    elif manual_box:
        shape_mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(shape_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)
    else:
        # Nessuna sagoma reale di balloon disponibile: ne' bubble_seg ne'
        # un box ridimensionato a mano danno una forma da riusare (solo il
        # box "a testo" di comic-text-detector, gia' escluso sopra).
        # Un'ellisse inscritta nel box somiglia alla forma reale di un
        # balloon molto piu' di un rettangolo pieno.
        shape_mask = np.zeros((h, w), np.uint8)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))
        cv2.ellipse(shape_mask, center, axes, 0, 0, 360, 255, thickness=-1)

    np_img = np.array(image)
    if fill_hex:
        np_img[shape_mask > 0] = ImageColor.getrgb(fill_hex)
    if border:
        contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(np_img, contours, -1, (0, 0, 0), thickness=3)
    image.paste(Image.fromarray(np_img))


_INV_SQRT2 = 0.7071067811865476  # 1/sqrt(2)


def _ellipse_inscribed_box(bbox: list[int]) -> list[int]:
    """Il piu' grande rettangolo assiale inscritto nell'ellisse del bbox
    (formula standard: meta' assi * 1/sqrt(2)), centrato come il bbox.
    Sta sempre interamente dentro la curva per costruzione, quindi il testo
    disegnato dentro non puo' mai finire a ridosso del bordo — a differenza
    di un fitting che segue riga per riga la maschera reale, resta un
    rettangolo semplice e prevedibile per chi poi modifica il testo a mano."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) / 2 * _INV_SQRT2
    half_h = (y2 - y1) / 2 * _INV_SQRT2
    return [int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)]


def _erode_mask(binary: np.ndarray, bbox: list[int], margin_ratio: float) -> np.ndarray:
    """Restringe la maschera di un margine proporzionale alle dimensioni del
    bbox, cosi' il testo non finisce a ridosso del contorno del balloon."""
    x1, y1, x2, y2 = bbox
    margin_px = max(2, int(margin_ratio * min(x2 - x1, y2 - y1)))
    kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), np.uint8)
    return cv2.erode(binary, kernel)


def _row_extent(mask: np.ndarray, y: int, cx: int) -> tuple[int, int] | None:
    """Segmento continuo di maschera sulla riga y che contiene la colonna cx.
    None se cx non e' dentro la maschera su quella riga."""
    h, w = mask.shape
    y = max(0, min(h - 1, y))
    row = mask[y]
    if cx < 0 or cx >= w or row[cx] == 0:
        return None
    left = cx
    while left > 0 and row[left - 1] != 0:
        left -= 1
    right = cx
    while right < w - 1 and row[right + 1] != 0:
        right += 1
    return left, right


def _row_widest_run(mask: np.ndarray, y: int) -> tuple[int, int] | None:
    """Segmento continuo piu' largo di maschera sulla riga y, indipendente
    da una colonna di riferimento fissa. A differenza di _row_extent (che
    richiede che un cx globale ricada dentro la maschera su ogni riga),
    questo serve per balloon fusi in modo asimmetrico dove i lobi non sono
    allineati sullo stesso asse verticale: un cx fisso puo' cadere fuori
    dalla maschera su alcune righe (es. il lobo superiore spostato a
    sinistra rispetto al centro del bbox complessivo) anche se quella riga
    fa parte del balloon ed e' utilizzabile per il testo."""
    h, w = mask.shape
    y = max(0, min(h - 1, y))
    row = mask[y]
    nz = np.where(row)[0]
    if len(nz) == 0:
        return None
    # Trova i run contigui e tiene il piu' largo.
    breaks = np.where(np.diff(nz) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(nz) - 1]))
    widths = nz[ends] - nz[starts]
    best = int(np.argmax(widths))
    return int(nz[starts[best]]), int(nz[ends[best]])


def _mask_body_rows(mask: np.ndarray, trim: bool = True) -> tuple[int, int, list[int], int] | None:
    """
    (y_top, y_bottom, profilo_larghezze, y0) del CORPO del balloon: la parte
    della maschera al netto delle estremita' affusolate (la coda verso chi
    parla, le punte di un ovale). Vedi balloon_shape.trim_tapered_ends.
    None se la maschera e' vuota.

    `trim=False` restituisce l'estensione piena. Va usato con le maschere di
    comic-text-detector, che seguono i tratti del testo e non la sagoma del
    balloon: li' non c'e' nessuna coda da togliere, e il profilo di
    larghezza e' quello delle righe di testo — il taglio si mangerebbe la
    prima o l'ultima riga.
    """
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return None
    y0 = int(ys.min())
    widths = []
    for y in range(y0, int(ys.max()) + 1):
        ext = _row_widest_run(mask, y)
        widths.append(0 if ext is None else ext[1] - ext[0])
    if max(widths, default=0) <= 0:
        return None
    if not trim:
        return y0, y0 + len(widths) - 1, widths, y0
    first, last = _trim_tapered_ends(_smooth_widths(widths))
    return y0 + first, y0 + last, widths, y0


def _mask_body_center(mask: np.ndarray, trim: bool = True) -> tuple[int, int] | None:
    """
    Centro della riga piu' larga del corpo del balloon. Serve come punto di
    partenza per _measure_balloon_extent al posto del centro del bbox: il
    bbox comprende anche la coda, e su un balloon piccolo il suo centro
    cade dentro la coda stessa, dove i raggi misurano una colonna larga
    pochi pixel (pagina 010 di Voluptuous Housewives 10: 47px invece di
    ~200, con il testo incolonnato e troncato).
    """
    body = _mask_body_rows(mask, trim=trim)
    if body is None:
        return None
    y_top, y_bottom, widths, y0 = body
    best_y, best_w = None, 0
    for y in range(y_top, y_bottom + 1):
        w = widths[y - y0]
        if w > best_w:
            best_w, best_y = w, y
    if best_y is None or best_w <= 0:
        return None
    ext = _row_widest_run(mask, best_y)
    if ext is None:
        return None
    return (ext[0] + ext[1]) // 2, best_y


def _wrap_text_mask(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width_for_line,
    hyphenate: bool = True,
) -> list[str] | None:
    """
    Come _wrap_text, ma la larghezza massima e' calcolata riga per riga
    (max_width_for_line(line_idx)) invece che fissa: serve per i balloon non
    convessi (es. a clessidra/doppia bolla), dove righe diverse hanno spazio
    orizzontale diverso. Ritorna None se una riga capita fuori dalla
    maschera (nessuno spazio disponibile) o se una parola non entra da sola
    nemmeno nella riga piu' larga.
    """
    words = text.split()
    lines: list[str] = []
    current_line = ""
    line_idx = 0

    i = 0
    while i < len(words):
        max_width = max_width_for_line(line_idx)
        if max_width <= 0:
            return None

        word = words[i]
        candidate = f"{current_line} {word}".strip()
        bbox = _textbbox(candidate, font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current_line = candidate
            i += 1
            continue

        if current_line:
            lines.append(current_line)
            line_idx += 1
            current_line = ""
            continue

        split = _split_word_to_fit(draw, word, font, max_width) if hyphenate else None
        if split is None:
            word_bbox = _textbbox(word, font)
            if word_bbox[2] - word_bbox[0] > max_width:
                return None
            lines.append(word)
            line_idx += 1
            i += 1
            continue

        head, tail = split
        lines.append(head)
        line_idx += 1
        words[i] = tail

    if current_line:
        lines.append(current_line)

    return lines


# _smooth_widths/_find_waist sono state spostate in balloon_shape.py
# (condivise con detect.py::_split_fused_bubbles, stesso identico
# algoritmo di rilevazione strozzatura riusato per separare due balloon
# fisici fusi in un'unica maschera bubble_seg). Alias locali per non
# toccare le chiamate esistenti piu' sotto.
_smooth_widths = balloon_shape.smooth_widths
_find_waist = balloon_shape.find_waist
_trim_tapered_ends = balloon_shape.trim_tapered_ends


def _fit_text_to_mask(
    draw: ImageDraw.ImageDraw,
    text: str,
    mask: np.ndarray,
    bbox: list[int],
    font_path: str,
    min_size: int,
    max_size: int,
    line_spacing: float,
    trust_single_lobe_shape: bool = False,
) -> tuple[list[tuple[str, float, float]], ImageFont.FreeTypeFont] | None:
    """
    Come _fit_text_to_box, ma vincola ogni riga alla forma reale del balloon
    (dalla maschera di detection) invece che al rettangolo del bbox.
    Necessario per i balloon non convessi (es. a clessidra/doppia bolla),
    dove il rettangolo e' molto piu' largo della forma vera in alcuni punti
    (es. la strozzatura tra le due bolle) e il testo rettangolare ci "esce
    fuori" dal contorno. Se il balloon ha due lobi distinti, distribuisce il
    testo su entrambe le zone invece di accumularlo tutto in una.
    Ritorna None se la maschera e' vuota o nessuna combinazione di
    font/righe entra: il chiamante ricade sul metodo a rettangolo.
    """
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return None

    x1, _, x2, _ = bbox
    cx = (x1 + x2) // 2

    def row_width(y: float) -> int:
        ext = _row_widest_run(mask, int(round(y)))
        return 0 if ext is None else ext[1] - ext[0]

    def row_center(y: float) -> float:
        ext = _row_widest_run(mask, int(round(y)))
        return cx if ext is None else (ext[0] + ext[1]) / 2

    # Le estremita' affusolate della maschera — la coda del balloon e le
    # punte di un ovale — non sono utilizzabili per il testo: sono larghe
    # pochi pixel e nemmeno una parola ci entra. Vanno escluse dall'altezza
    # su cui si distribuisce il testo, altrimenti il fitting fallisce del
    # tutto e si ricade sul rettangolo (vedi trim_tapered_ends). Le
    # strozzature legittime (balloon a clessidra) restano invece dentro:
    # il taglio agisce solo alle estremita' e si rapporta alla larghezza
    # del lobo che trova, non a quella del balloon intero.
    body = _mask_body_rows(mask, trim=trust_single_lobe_shape)
    if body is None:
        return None
    y_top, y_bottom, all_widths, y0 = body
    first, last = y_top - y0, y_bottom - y0
    extent_height = y_bottom - y_top
    if extent_height <= 0:
        return None

    def make_max_width_fn(line_height: float, y_start: float):
        def fn(line_idx: int) -> int:
            top = y_start + line_idx * line_height
            bottom = top + line_height
            samples = (top, (top + bottom) / 2, bottom)
            return max(0, min(row_width(y) for y in samples) - 4)
        return fn

    def line_centers_for(line_height: float, y_start: float, n_lines: int) -> list[float]:
        # Il centro orizzontale reale della maschera puo' differire dal centro
        # del bbox (es. balloon a doppia bolla non allineate): ogni riga va
        # centrata sul proprio segmento di maschera, non su un cx fisso,
        # altrimenti il testo sconfina dal lato dove la bolla e' piu' stretta.
        centers = []
        for line_idx in range(n_lines):
            top = y_start + line_idx * line_height
            bottom = top + line_height
            samples = (top, (top + bottom) / 2, bottom)
            centers.append(sum(row_center(y) for y in samples) / len(samples))
        return centers

    def fit_zone(text_chunk, zone_top, zone_extent, font, line_height, hyphenate):
        if not text_chunk.strip():
            return None
        # Stima del numero di righe: usa la larghezza al centro verticale
        # della zona come approssimazione costante, non quella al bordo
        # superiore. Il bordo di un balloon ovale e' spesso molto piu'
        # stretto del centro (e' li' che la forma si assottiglia verso la
        # punta), quindi ancorare la stima al bordo fa fallire lo split
        # anche quando il testo, una volta centrato, ci starebbe comodo.
        center_width = max(0, row_width(zone_top + zone_extent / 2) - 4)
        lines = _wrap_text_mask(draw, text_chunk, font, lambda i: center_width, hyphenate)
        if lines is None:
            return None
        block_height = len(lines) * line_height
        if block_height > zone_extent:
            return None
        y_start = zone_top + (zone_extent - block_height) / 2
        lines = _wrap_text_mask(draw, text_chunk, font, make_max_width_fn(line_height, y_start), hyphenate)
        if lines is None:
            return None
        # Le righe vicino al bordo di un balloon ovale sono piu' strette di
        # quella al centro (usata sopra per la sola stima del numero di
        # righe): il ri-wrap con la larghezza reale riga per riga puo' quindi
        # produrre piu' (o meno) righe di quelle stimate. Se non si ricentra
        # su questo conteggio reale, il blocco resta ancorato allo y_start
        # pensato per un blocco di altezza diversa, e il testo finisce
        # sbilanciato verso il basso (o l'alto) invece che centrato.
        block_height = len(lines) * line_height
        if block_height > zone_extent:
            return None
        # Centrare sull'altezza NOMINALE (righe * line_height) sposta il
        # blocco verso il basso: il font riserva sopra il cap-height un
        # margine diverso da quello che riserva sotto la baseline per i
        # discendenti (che un testo in maiuscolo come questo non usa quasi
        # mai), quindi il primo/ultimo pixel di inchiostro non coincidono
        # con i bordi nominali della riga. Si centra invece sul bbox reale
        # (inchiostro) di prima e ultima riga, come gia' fa correttamente
        # _draw_lines_centered per il percorso a rettangolo manuale.
        first_bb = _textbbox(lines[0], font)
        last_bb = _textbbox(lines[-1], font)
        ink_top = first_bb[1]
        ink_bottom = (len(lines) - 1) * line_height + last_bb[3]
        block_center = (ink_top + ink_bottom) / 2
        y_start = zone_top + zone_extent / 2 - block_center
        centers = line_centers_for(line_height, y_start, len(lines))
        return [(line, centers[i], y_start + i * line_height) for i, line in enumerate(lines)]

    waist_y = _find_waist(_smooth_widths(all_widths[first:last + 1]), y_top)
    words = text.split()

    if waist_y is not None and len(words) >= 2:
        zone_a = (y_top, waist_y)
        zone_b = (waist_y, y_bottom)
        cap_a = sum(row_width(y) for y in range(zone_a[0], zone_a[1] + 1)) or 1
        cap_b = sum(row_width(y) for y in range(zone_b[0], zone_b[1] + 1)) or 1
        target = round(len(words) * cap_a / (cap_a + cap_b))
        target = max(1, min(len(words) - 1, target))
        # Prova split vicini al target proporzionale, dal piu' vicino in poi.
        split_order = sorted(range(1, len(words)), key=lambda s: abs(s - target))

        for hyphenate in (False, True):
            for size in range(max_size, min_size - 1, -1):
                font = _load_font(font_path, size)
                line_height = size * line_spacing
                for split in split_order:
                    part_a = " ".join(words[:split])
                    part_b = " ".join(words[split:])
                    placed_a = fit_zone(part_a, zone_a[0], zone_a[1] - zone_a[0], font, line_height, hyphenate)
                    if placed_a is None:
                        continue
                    placed_b = fit_zone(part_b, zone_b[0], zone_b[1] - zone_b[0], font, line_height, hyphenate)
                    if placed_b is None:
                        continue
                    return placed_a + placed_b, font

    # Nessuna strozzatura a doppio lobo. Se la maschera viene da bubble_seg
    # (la vera sagoma del balloon, non solo del testo), e' affidabile anche
    # a lobo singolo: adattiamo il testo alla sua estensione reale, che
    # segue la curvatura ovale meglio di un rettangolo con safe_area_ratio
    # (altrimenti il testo sconfina agli angoli, fuori dal balloon).
    # Se invece la maschera e' quella del solo testo originale (da
    # comic-text-detector), puo' essere molto piu' stretta del balloon
    # reale: in quel caso non e' affidabile per il fitting, meglio
    # ricadere sul rettangolo con safe_area_ratio.
    if trust_single_lobe_shape:
        for hyphenate in (False, True):
            for size in range(max_size, min_size - 1, -1):
                font = _load_font(font_path, size)
                line_height = size * line_spacing
                placed = fit_zone(text, y_top, extent_height, font, line_height, hyphenate)
                if placed is not None:
                    return placed, font

    return None


def _draw_text_in_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: list[int],
    cfg: dict,
    image: Image.Image,
    font_path: str = None,
    mask_path: str = None,
    manual_box: bool = False,
    trust_mask_shape: bool = False,
    text_color: str = None,
    manual_style: dict | None = None,
    exclude_mask: np.ndarray | None = None,
    neighbour_boxes: list[list[int]] | None = None,
    forced_size: int | None = None,
    extra_shape: np.ndarray | None = None,
    dry_run: bool = False,
) -> tuple[bool, int]:
    """Ritorna (overflowed, font_size_usato). overflowed e' True se il testo
    e' andato in overflow (troncato per non uscire dal balloon) — segnale
    che la traduzione andrebbe accorciata. `manual_style` e' il det stesso
    (o un dict con gli stessi campi manual_font_size/manual_line_spacing/
    manual_align/manual_bold/manual_italic/manual_underline/
    manual_outline_*), passato dal chiamante quando presente; qui viene
    usato solo se manual_box e' True (il chiamante e' responsabile di
    forzare manual_box quando questi campi sono impostati, vedi
    _has_manual_style).

    `forced_size`, se impostato, forza sia min che max della ricerca
    automatica della dimensione a quel valore (usato per la dimensione
    font uniforme calcolata su tutta la pagina, vedi run()); ignorato sul
    percorso "box manuale" (li' vince manual_font_size dell'utente).
    `extra_shape`, se passata, e' l'area del balloon che bubble_seg ha perso
    (ricavata dai tratti del testo originale, vedi
    balloon_shape.text_ink_outside_balloons): viene unita alla maschera
    prima del fitting, altrimenti il testo resta centrato su una sagoma piu'
    piccola del balloon disegnato.
    `dry_run`, se True, calcola tutto (dimensione/righe) ma non scrive sui
    pixel dell'immagine: serve per stimare la dimensione naturale di un
    balloon senza disegnarlo (fase di calcolo della size uniforme)."""
    render_cfg = cfg["rendering"]
    safe_area_ratio = render_cfg.get("safe_area_ratio", 0.75)
    mask_margin_ratio = render_cfg.get("mask_margin_ratio", 0.05)

    text = _normalize_punctuation(text)
    # Colore scelto a mano in Revisione ha priorita' sull'auto-detect
    # (nero/bianco in base alla luminosita' dello sfondo).
    fill_color = text_color
    if not fill_color:
        # La maschera serve a misurare la luminosita' del solo interno del
        # balloon: vedi _detect_text_color. Si carica qui anche quando il
        # fitting non la usera' (percorso "box manuale"), perche' il colore
        # sbagliato e' un difetto ben piu' visibile del fitting.
        color_mask = _load_mask_array(mask_path) if mask_path else None
        fill_color = _detect_text_color(image, bbox, color_mask)

    # Usa il font specifico del balloon se disponibile, altrimenti il default
    effective_font_path = str(paths.resolve(font_path if font_path else render_cfg["font_path"]))

    style = None
    if manual_style and _has_manual_style(manual_style):
        italic_shear = False
        if manual_style.get("manual_italic"):
            italic_variant = _resolve_italic_font_path(effective_font_path)
            if italic_variant:
                effective_font_path = italic_variant
            else:
                italic_shear = True
        style = {
            "bold": bool(manual_style.get("manual_bold")),
            "underline": bool(manual_style.get("manual_underline")),
            "outline_enable": bool(manual_style.get("manual_outline_enable")),
            "outline_color": manual_style.get("manual_outline_color"),
            "outline_width": manual_style.get("manual_outline_width"),
            "italic_shear": italic_shear,
        }

    # Bbox ridimensionato a mano in Revisione: rispetta le dimensioni scelte
    # dall'utente invece di ri-misurare/usare la maschera, che altrimenti
    # ignorerebbero un bbox allargato manualmente. Salta dritto al
    # rettangolo, con solo un piccolo margine di sicurezza fisso.
    #
    # Eccezione: manual_respect_shape (flag in Revisione) tiene invece il
    # fitting a maschera anche a box ridimensionato a mano. Serve per i
    # balloon doppi/a clessidra (trust_mask_shape, cioe' balloon_source
    # yolov8seg con una vera sagoma non rettangolare): li' il rettangolo
    # pieno del bbox e' piu' largo della forma reale nel "collo" tra i due
    # lobi, quindi il testo adattato al solo rettangolo sconfina oltre il
    # contorno curvo del balloon proprio li'.
    #
    # balloon_split (impostato automaticamente da
    # detect.py::_split_fused_bubbles) implica la stessa eccezione senza
    # bisogno di un intervento manuale in Revisione: i bbox di due meta'
    # di uno split via erosione/watershed possono sovrapporsi anche quando
    # le maschere reali non si toccano (split non allineato agli assi,
    # es. balloon fusi in diagonale) — il rettangolo pieno del bbox
    # farebbe allora sconfinare il testo di una meta' dentro l'area
    # dell'altra, sovrapponendole visivamente.
    respect_shape = bool(
        manual_style.get("manual_respect_shape") or manual_style.get("balloon_split")
    ) if manual_style else False
    if manual_box and not (respect_shape and trust_mask_shape and mask_path):
        x1, y1, x2, y2 = bbox
        # Un balloon nato da uno split (detect.py::_split_fused_bubbles) ha
        # un bbox che e' gia' il bounding box stretto della sua meta' di
        # maschera, a ridosso del bordo curvo reale del balloon — diverso
        # da un box ridimensionato a mano in Revisione (che l'utente ha
        # gia' allargato a piacere). Il 4px standard basta per quel caso ma
        # lascia il testo troppo vicino al bordo qui: serve un margine
        # proporzionale alla dimensione del box, non fisso.
        if manual_style and manual_style.get("balloon_split"):
            inset = max(4, int(min(x2 - x1, y2 - y1) * 0.12))
        else:
            inset = 4
        x1, y1, x2, y2 = x1 + inset, y1 + inset, x2 - inset, y2 - inset
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        manual_size = manual_style.get("manual_font_size") if manual_style else None
        min_size = manual_size or render_cfg["min_font_size"]
        max_size = manual_size or render_cfg["max_font_size"]
        line_spacing = (manual_style.get("manual_line_spacing") if manual_style else None) or render_cfg["line_spacing"]
        align = (manual_style.get("manual_align") if manual_style else None) or "center"
        lines, font, overflowed = _fit_text_to_box(
            draw, text, box_width, box_height, effective_font_path,
            min_size, max_size, line_spacing,
        )
        if not dry_run:
            _draw_lines_centered(draw, lines, font, x1, y1, box_width, box_height, line_spacing, fill_color,
                                  align=align, style=style, image=image)
        return overflowed, font.size

    # Se disponibile una maschera di detection, usala per il fitting: e'
    # l'unico modo di gestire correttamente i balloon non rettangolari
    # (a clessidra, doppia bolla, ecc.) dove il bbox non rispecchia la
    # forma reale del balloon.
    #
    # Solo pero' se la maschera e' davvero la SAGOMA del balloon
    # (trust_mask_shape, cioe' balloon_source yolov8seg/bubble_seg). Le
    # detection rimaste a balloon_source "text_detector" hanno come maschera
    # quella di comic-text-detector, che contiene i soli pixel d'inchiostro
    # del testo originale: decine di blob staccati (una parola per blob) con
    # cui _fit_text_to_mask impagina riga per riga su estensioni e centri
    # arbitrari, sparpagliando le parole a caso nel riquadro e collassando
    # la dimensione del font (didascalie rettangolari, pagina 002). Per
    # quelle si va dritti al percorso rettangolare qui sotto
    # (_measure_balloon_extent / _safe_area + _fit_text_to_box), che e'
    # esattamente cio' che serve a un riquadro di didascalia.
    auto_min_size = forced_size or render_cfg["min_font_size"]
    auto_max_size = forced_size or render_cfg["max_font_size"]

    mask = _load_mask_array(mask_path) if (mask_path and trust_mask_shape) else None
    if mask is not None and extra_shape is not None:
        # Dove un balloon disegnato sopra taglia la sagoma segmentata, i
        # tratti del testo ORIGINALE cadono fuori dalla maschera: sono la
        # prova che li' c'e' interno di balloon utilizzabile. Senza questo
        # recupero il testo tradotto viene centrato (e dimensionato) su una
        # sagoma piu' piccola del balloon che il lettore vede.
        mask = cv2.bitwise_or(mask, extra_shape)
    fitted = None
    body_center = None
    body_rows = None
    if mask is not None:
        # Un balloon nato da uno split ha gia' esattamente la maschera della
        # sua meta': il recupero dell'estensione piena del balloon farebbe
        # risalire il flood-fill fino a coprire di nuovo l'intera area fusa
        # (vedi il commento su "split" in detect.py::_split_fused_bubbles).
        is_balloon_split = bool(manual_style and manual_style.get("balloon_split"))
        if trust_mask_shape and not is_balloon_split:
            mask = _expand_mask_to_drawn_balloon(
                image, mask, bbox, exclude=exclude_mask, neighbours=neighbour_boxes
            )
        eroded = _erode_mask(_smooth_mask(mask), bbox, mask_margin_ratio)
        if trust_mask_shape:
            body_center = _mask_body_center(eroded)
            body = _mask_body_rows(eroded)
            body_rows = (body[0], body[1]) if body else None
        fitted = _fit_text_to_mask(
            draw,
            text,
            eroded,
            bbox,
            effective_font_path,
            auto_min_size,
            auto_max_size,
            render_cfg["line_spacing"],
            trust_single_lobe_shape=trust_mask_shape,
        )

    if fitted is not None:
        placed, font = fitted
        if not dry_run:
            for line, x_center, y in placed:
                line_bbox = _textbbox(line, font)
                line_width = line_bbox[2] - line_bbox[0]
                # Centro reale della maschera per questa riga, non il centro
                # del bbox: nei balloon a doppia bolla i due lobi possono non
                # essere allineati, e centrare su un cx fisso fa sconfinare
                # il testo dal lato dove la bolla e' piu' stretta.
                x = x_center - line_width / 2
                draw.text((x, y), line, font=font, fill=fill_color)
        return False, font.size  # _fit_text_to_mask non tronca mai: o entra, o fallback al rettangolo

    # Fallback: nessun lobo doppio rilevato dalla maschera. Prova prima a
    # misurare l'estensione reale del balloon dall'immagine pulita (piu'
    # affidabile di un rapporto fisso, che sovra- o sotto-stima a seconda
    # di quanto un balloon specifico riempie il suo bbox); se la misura
    # fallisce (es. sfondo del balloon non abbastanza chiaro), ricadi sul
    # rapporto fisso.
    measured = _measure_balloon_extent(image, bbox, center=body_center)
    if measured is not None:
        x1, y1, x2, y2 = measured
    else:
        x1, y1, x2, y2 = _safe_area(bbox, safe_area_ratio)
    if body_rows is not None:
        # Anche partendo dal centro del corpo, il raggio verso il basso puo'
        # infilarsi nella coda se questa parte proprio sotto il centro:
        # si limita comunque il rettangolo alle righe del corpo.
        y1 = max(y1, body_rows[0])
        y2 = min(y2, body_rows[1])
        if y2 - y1 < 8:
            y1, y2 = body_rows
    box_width = x2 - x1
    box_height = y2 - y1

    lines, font, overflowed = _fit_text_to_box(
        draw,
        text,
        box_width,
        box_height,
        effective_font_path,
        auto_min_size,
        auto_max_size,
        render_cfg["line_spacing"],
    )
    if not dry_run:
        _draw_lines_centered(draw, lines, font, x1, y1, box_width, box_height, render_cfg["line_spacing"], fill_color)
    return overflowed, font.size


def _draw_lines_centered(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    x1: float,
    y1: float,
    box_width: float,
    box_height: float,
    line_spacing: float,
    fill_color: tuple[int, int, int],
    align: str = "center",
    style: dict | None = None,
    image: Image.Image | None = None,
):
    """Centra (verticalmente, sempre) un blocco di righe gia' calcolato da
    _fit_text_to_box dentro il rettangolo (x1, y1, box_width, box_height) e
    lo disegna. `align` controlla solo l'allineamento orizzontale di ogni
    riga (default "center", comportamento invariato rispetto a prima).
    `style`/`image` sono usati solo per gli override manuali di formattazione
    (grassetto/corsivo/contorno/sottolineato impostati in Revisione)."""
    line_height = font.size * line_spacing

    bboxes = [_textbbox(line, font) for line in lines]
    if not bboxes:
        return

    # Centriamo verticalmente l'intero blocco di testo
    center_y = y1 + box_height / 2

    if len(lines) == 1:
        # Caso semplice: una sola riga
        line_bbox = bboxes[0]
        text_center = (line_bbox[1] + line_bbox[3]) / 2
        y_start = center_y - text_center
    else:
        # Più righe: calcola top e bottom del blocco
        first_bbox = bboxes[0]
        last_bbox = bboxes[-1]
        top = first_bbox[1]  # ascender della prima riga
        bottom = (len(lines) - 1) * line_height + last_bbox[3]  # descender dell'ultima
        block_center = (top + bottom) / 2
        y_start = center_y - block_center

    for i, line in enumerate(lines):
        line_bbox = _textbbox(line, font)
        line_width = line_bbox[2] - line_bbox[0]
        if align == "left":
            x = x1
        elif align == "right":
            x = x1 + box_width - line_width
        else:
            x = x1 + (box_width - line_width) / 2
        y = y_start + i * line_height
        if style:
            _draw_line_styled(image, draw, x, y, line, font, fill_color, style)
        else:
            draw.text((x, y), line, font=font, fill=fill_color)
        if style and style.get("underline"):
            underline_y = y + line_bbox[3] + max(1, round(font.size * 0.08))
            draw.line([(x, underline_y), (x + line_width, underline_y)],
                      fill=fill_color, width=max(1, round(font.size * 0.05)))


def render_balloon_preview(
    cleaned_image_path: Path, det: dict, cfg: dict, margin: int = 60,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Renderizza UN SOLO balloon (`det`, incluse modifiche non ancora salvate
    su disco: testo, font, riempimento manuale, forma, codino) su un ritaglio
    della pagina gia' pulita, per l'anteprima live in Revisione. Riusa la
    stessa `_apply_manual_fill`/`_draw_text_in_bbox` del rendering completo
    (`run`), quindi il risultato e' pixel-identico a quello che si otterrebbe
    rigenerando l'intera pagina — a differenza di un'approssimazione
    calcolata lato client, che dovrebbe reimplementare in JS autofit/
    word-wrap/sagoma del balloon e rischierebbe di disallinearsi nel tempo.

    Ritorna (ritaglio, bbox_ritaglio_nella_pagina_originale): il chiamante
    (l'endpoint web) usa il bbox per posizionare l'immagine sull'overlay.
    """
    image = _load_cleaned_image_cached(cleaned_image_path)
    draw = ImageDraw.Draw(image)
    _apply_manual_fill(image, det)

    text = (det.get("testo_tradotto") or "").strip()
    bbox = det["bbox"]
    if text and text != "-":
        font_path = det.get("font_path")
        mask_path = det.get("mask_path")
        manual_box = bool(det.get("manual_text_box")) or _has_manual_style(det)
        trust_mask_shape = det.get("balloon_source") == "yolov8seg"
        text_bbox = bbox
        if det.get("manual_balloon_shape") == "ellipse":
            text_bbox = _ellipse_inscribed_box(bbox)
            manual_box = True
        # L'anteprima live di Revisione mostra sempre il singolo balloon in
        # isolamento: qui non ha senso vincolarlo alla size uniforme
        # calcolata sull'intera pagina (quel calcolo vive in run()), quindi
        # usa il fitting naturale del balloon come nel resto dell'editor.
        _draw_text_in_bbox(
            draw, text, text_bbox, cfg, image, font_path, mask_path, manual_box, trust_mask_shape,
            det.get("manual_text_color"), manual_style=det,
        )

    x1, y1, x2, y2 = bbox
    w, h = image.size
    cx1 = max(0, int(x1) - margin)
    cy1 = max(0, int(y1) - margin)
    cx2 = min(w, int(x2) + margin)
    cy2 = min(h, int(y2) + margin)
    crop = image.crop((cx1, cy1, cx2, cy2))
    return crop, (cx1, cy1, cx2, cy2)


def run(cleaned_image_path: Path, translated_json_path: Path, cfg: dict, output_dir: Path) -> Path:
    with open(translated_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = _load_cleaned_image_cached(cleaned_image_path)
    draw = ImageDraw.Draw(image)

    # Maschere RAW (bubble_seg) di ogni balloon yolov8seg della pagina, per
    # impedire che il recupero via flood fill di un balloon (vedi
    # _expand_mask_to_drawn_balloon) sconfini in un balloon vicino quando i
    # due sono disegnati a contatto senza bordo scuro tra loro (es. dialoghi
    # consecutivi dello stesso personaggio, pagina 005 "CALM DOWN, LUCAS!" /
    # "WE WERE SO HORNY..."). Dilatata di qualche pixel per coprire anche
    # l'antialiasing del bordo.
    raw_masks: dict[int, np.ndarray] = {}
    for i, det in enumerate(data["detections"]):
        if det.get("balloon_source") == "yolov8seg" and det.get("mask_path"):
            m = _load_mask_array(det["mask_path"])
            if m is not None:
                raw_masks[i] = m
    _exclude_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Bbox delle maschere RAW, calcolati una volta sola: sono il riferimento
    # onesto della posizione di ogni balloon (il bbox nel json puo' essere
    # stato allargato da un render precedente, vedi _swallows_neighbour).
    _raw_boxes: dict[int, list[int]] = {}
    for i, m in raw_masks.items():
        ys, xs = np.where(m > 0)
        if len(ys):
            _raw_boxes[i] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    # Sagoma persa dalla segmentazione, per balloon (vedi extra_shape in
    # _draw_text_in_bbox). Si calcola una volta per pagina: l'assegnazione di
    # un testo al balloon giusto e' un confronto tra tutti i balloon.
    _extra_shapes = balloon_shape.text_ink_outside_balloons(
        balloon_shape.load_text_masks(
            [d["mask_path"] for d in data["detections"] if d.get("mask_path")],
            (image.size[1], image.size[0]),
        ),
        [raw_masks.get(i) for i in range(len(data["detections"]))],
    )

    def _neighbour_boxes_for(idx: int) -> list[list[int]]:
        """Bbox degli ALTRI balloon con sagoma reale: sono quelli in cui il
        recupero non deve sconfinare (vedi _swallows_neighbour)."""
        return [box for i, box in _raw_boxes.items() if i != idx]

    def _exclude_mask_for(idx: int) -> np.ndarray | None:
        others = [m for i, m in raw_masks.items() if i != idx]
        if not others:
            return None
        union = others[0]
        for m in others[1:]:
            union = cv2.bitwise_or(union, m)
        return cv2.dilate(union, _exclude_kernel)

    # Fase 1: correzione bbox sotto-segmentati (indipendente dal testo/fill).
    for det_idx, det in enumerate(data["detections"]):
        # Se bubble_seg ha sotto-segmentato il balloon (maschera piu' piccola
        # del balloon davvero disegnato), correggere solo il fitting del
        # testo non basta: il bbox salvato resta quello piccolo, e con lui
        # anche il box verde di selezione/trascinamento in Revisione, che
        # continua a mostrare (e a far editare) un'area piu' piccola del
        # balloon reale. Si aggiorna qui il bbox persistito, cosi' la
        # correzione vale anche per l'editor, non solo per il rendering.
        if (
            det.get("balloon_source") == "yolov8seg"
            and not det.get("manual_text_box")
            and not _has_manual_style(det)
            and det.get("manual_balloon_shape") != "ellipse"
            and det.get("mask_path")
        ):
            raw_mask = _load_mask_array(det["mask_path"])
            if raw_mask is not None:
                expanded_mask = _expand_mask_to_drawn_balloon(
                    image, raw_mask, det["bbox"], exclude=_exclude_mask_for(det_idx),
                    neighbours=_neighbour_boxes_for(det_idx),
                )
                eys, exs = np.where(expanded_mask > 0)
                if len(eys):
                    det["bbox"] = [int(exs.min()), int(eys.min()), int(exs.max()) + 1, int(eys.max()) + 1]

    # Fase 2: riempimenti manuali (Revisione), prima di leggere colori/
    # estensione dei balloon per il fitting, cosi' quelle letture (fase 3 e
    # 4) vedono gia' lo sfondo definitivo.
    for det in data["detections"]:
        _apply_manual_fill(image, det)

    def _text_render_params(det: dict) -> tuple[str, list[int], bool, bool] | None:
        text = det.get("testo_tradotto", "").strip()
        if not text or text == "-":
            return None
        # Un override manuale di formattazione (dimensione/interlinea/
        # allineamento/grassetto/corsivo/sottolineato/contorno impostati in
        # Revisione) forza lo stesso percorso "box manuale" gia' usato per
        # manual_text_box/l'ellisse ricreata: quei parametri non hanno senso
        # combinati con la ricerca automatica della dimensione a maschera.
        manual_box = bool(det.get("manual_text_box")) or _has_manual_style(det)
        trust_mask_shape = det.get("balloon_source") == "yolov8seg"
        text_bbox = det["bbox"]
        if det.get("manual_balloon_shape") == "ellipse":
            # Balloon ricreato a mano: il testo va nel rettangolo inscritto
            # nell'ellisse (garantito dentro la curva per costruzione),
            # invece che nel bbox intero (che e' il bounding box dell'ellisse
            # e la fa finire a ridosso del bordo agli angoli). Forza il
            # percorso "box manuale" (semplice, prevedibile) anche se
            # manual_text_box non era gia' impostato: il rettangolo inscritto
            # sostituisce comunque qualsiasi fitting piu' complesso.
            text_bbox = _ellipse_inscribed_box(det["bbox"])
            manual_box = True
        return text, text_bbox, manual_box, trust_mask_shape

    # Fase 3: stima (senza disegnare) la dimensione font naturale di ogni
    # balloon "normale" — non manuale, non di urlo/stupore — e ne prende il
    # minimo. Nei fumetti reali il testo mantiene una dimensione uniforme su
    # tutta la tavola/il fumetto (vedi _is_shout_text): calcolare il minimo
    # comune tra tutti i balloon "normali" della pagina garantisce che quella
    # size entri comunque in ognuno di essi, senza dover ridurre ulteriormente
    # a valle. I balloon manuali (l'utente ha gia' scelto una size) e quelli
    # di urlo (deliberatamente diversi, spesso piu' grandi) restano fuori dal
    # calcolo e mantengono il fitting automatico libero.
    #
    # rendering.fixed_font_size in config.yaml, se impostato, salta del tutto
    # questo calcolo per-pagina e forza quella size fissa su tutte le pagine
    # del fumetto (stessa config per ogni run()): serve a chi preferisce una
    # dimensione scelta a mano e costante ovunque invece del minimo comune
    # ricalcolato pagina per pagina.
    fixed_font_size = cfg["rendering"].get("fixed_font_size")
    free_fit: set[int] = set()
    if fixed_font_size:
        uniform_size = int(fixed_font_size)
    else:
        natural_sizes = []
        natural_by_idx: dict[int, int] = {}
        for det_idx, det in enumerate(data["detections"]):
            params = _text_render_params(det)
            if params is None:
                continue
            text, text_bbox, manual_box, trust_mask_shape = params
            # Le didascalie (nessuna sagoma di balloon da bubble_seg: sono i
            # riquadri narrativi rettangolari, spesso larghi e bassi) non
            # fanno parte del gruppo "balloon di dialogo" a cui si applica
            # la dimensione comune: nei fumetti reali hanno un corpo loro,
            # e un riquadro alto una riga vuole per forza un font piccolo
            # che, entrando nel minimo comune, rimpicciolirebbe tutti i
            # dialoghi della tavola. Restano a fitting libero.
            if manual_box or _is_shout_text(text) or not trust_mask_shape:
                continue
            _, size = _draw_text_in_bbox(
                draw, text, text_bbox, cfg, image, det.get("font_path"), det.get("mask_path"),
                manual_box, trust_mask_shape, det.get("manual_text_color"), manual_style=det,
                exclude_mask=_exclude_mask_for(det_idx),
                neighbour_boxes=_neighbour_boxes_for(det_idx),
                extra_shape=_extra_shapes[det_idx], dry_run=True,
            )
            natural_sizes.append(size)
            natural_by_idx[det_idx] = size
        # Il minimo puro non ha difese contro un singolo balloon mal
        # misurato: basta una detection anomala (maschera sbagliata, bbox
        # sotto-segmentato, testo lunghissimo in un box minuscolo) perche'
        # la sua size naturale trascini tutta la tavola verso il basso,
        # rimpicciolendo anche balloon che stavano comodi al doppio della
        # dimensione. Si scartano quindi dal calcolo le size palesemente
        # fuori scala rispetto alla pagina (sotto il 60% della mediana):
        # quei balloon mantengono comunque il proprio fitting, ma non
        # dettano piu' la size di tutti gli altri.
        if natural_sizes:
            floor = float(np.median(natural_sizes)) * 0.6
            kept = [s for s in natural_sizes if s >= floor]
            outliers = sorted(s for s in natural_sizes if s < floor)
            if outliers:
                log.info(
                    f"Font size: scartati {len(outliers)} balloon fuori scala dal calcolo "
                    f"della size uniforme {outliers} (soglia {floor:.0f})"
                )
            uniform_size = min(kept) if kept else min(natural_sizes)
            # Un balloon escluso dal CALCOLO va escluso anche dall'obbligo
            # di usare la size risultante: quella e' per definizione piu'
            # grande di quanto ci stia dentro, quindi imporgliela lo manda
            # in overflow e il testo viene troncato (perdita di contenuto).
            # Mantiene invece il proprio fitting libero, come i balloon
            # manuali e di urlo: risulta piu' piccolo degli altri, ma
            # completo.
            free_fit = {i for i, sz in natural_by_idx.items() if sz < floor}
        else:
            uniform_size = None

    # Fase 4: disegno vero e proprio, con la size uniforme per i balloon
    # "normali" e il fitting libero per quelli manuali/di urlo.
    overflow_count = 0
    for det_idx, det in enumerate(data["detections"]):
        params = _text_render_params(det)
        if params is None:
            det["_overflow"] = False
            det.pop("_rendered_font_size", None)
            continue
        text, text_bbox, manual_box, trust_mask_shape = params
        forced_size = None if (
            manual_box or _is_shout_text(text) or not trust_mask_shape or det_idx in free_fit
        ) else uniform_size
        overflowed, _size = _draw_text_in_bbox(
            draw, text, text_bbox, cfg, image, det.get("font_path"), det.get("mask_path"),
            manual_box, trust_mask_shape, det.get("manual_text_color"), manual_style=det,
            exclude_mask=_exclude_mask_for(det_idx),
            neighbour_boxes=_neighbour_boxes_for(det_idx), forced_size=forced_size,
            extra_shape=_extra_shapes[det_idx],
        )
        det["_overflow"] = overflowed
        det["_rendered_font_size"] = _size
        if overflowed:
            overflow_count += 1

    if uniform_size:
        log.info(f"Font size uniforme pagina: {uniform_size}")
    manual_sizes = sorted({
        det["_rendered_font_size"] for det in data["detections"]
        if det.get("_rendered_font_size") and (bool(det.get("manual_text_box")) or _has_manual_style(det) or _is_shout_text(det.get("testo_tradotto", "")))
    })
    if manual_sizes:
        log.info(f"Font size balloon manuali/urlo (fitting libero): {manual_sizes}")

    output_path = output_dir / f"{data['page_id']}.jpg"
    image.save(output_path, format="JPEG", quality=70)

    # Persiste il flag di overflow nel translated.json: la GUI di Revisione
    # lo usa per evidenziare i balloon dove il testo tradotto e' stato
    # troncato, cosi' l'utente sa quali accorciare invece di scoprirlo solo
    # nei log.
    with open(translated_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if overflow_count:
        log.warning(f"{data['page_id']}: {overflow_count} balloon in overflow (testo troncato)")
    log.info(f"Pagina renderizzata: {output_path}")
    return output_path