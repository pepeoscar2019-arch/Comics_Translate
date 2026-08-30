import json
import logging
from pathlib import Path

import cv2
import numpy as np

import balloon_shape

log = logging.getLogger("pipeline.clean")

_BORDER_INK_DARK_THRESH = 130
"""
Soglia di luminosita' (scala di grigi, 0-255) sotto la quale un pixel e'
considerato inchiostro, non sfondo. Usata da _protect_border_ink per
riconoscere il contorno del balloon indipendentemente dalla sua forma
esatta (ovale semplice, codina, svirgola decorativa, ecc.).
"""

_BORDER_INK_SEARCH_MARGIN_PX = 30
"""Margine di ricerca attorno al balloon per individuare inchiostro del
contorno che si trova appena fuori dalla sua sagoma grezza (non erosa)."""

_BORDER_INK_MAX_PROTECT_DEPTH_PX = 20
"""Profondita' massima (distanza dal perimetro vero del balloon) entro cui
un pixel scuro puo' essere protetto come "contorno". Su un balloon a
doppia gobba (due lobi fusi, vedi detect.py::_split_fused_bubbles) la
strozzatura tra i due lobi porta il contorno nero molto vicino al testo:
una lettera puo' toccare per un solo pixel l'esterno della sagoma grezza
ed essere quindi nella stessa componente connessa del contorno. Senza
questo limite di profondita', l'intera lettera verrebbe protetta insieme
al contorno (vedi _protect_border_ink). Codine e svirgole decorative
restano comunque protette perche' per definizione corrono a ridosso del
perimetro per tutta la loro estensione."""


def _protect_border_ink(
    eroded_mask: np.ndarray, raw_mask: np.ndarray, border_gray: np.ndarray | None
) -> np.ndarray:
    """
    Rifinisce la maschera erosa escludendo eventuale inchiostro del
    contorno rimasto dentro l'area da cancellare. Una singola erosione a
    margine fisso assume un bordo di spessore costante lungo tutto il
    contorno: sui balloon con dettagli che si insinuano piu' in profondita'
    (code, svirgole decorative), l'erosione fissa non basta e quel tratto
    di inchiostro finirebbe cancellato dall'inpainting o dal riempimento
    piatto.

    Per distinguere questo inchiostro dal testo (che va invece cancellato),
    usa la connettivita': si etichettano le componenti connesse di pixel
    scuri vicino al balloon, e si protegge (esclude dalla maschera) solo
    chi ha ALMENO un pixel fuori dalla sagoma grezza (non erosa) del
    balloon — il contorno del balloon per definizione si estende oltre la
    sua stessa sagoma (e' il perimetro), mentre le lettere del testo sono
    tratti isolati interamente contenuti all'interno del balloon e non
    toccano mai l'esterno. Questo evita di proteggere per errore lettere
    che capitano vicine al bordo (es. testo che riempie il balloon quasi
    fino al contorno).
    """
    if border_gray is None or not eroded_mask.any():
        return eroded_mask

    x, y, w, h = cv2.boundingRect(raw_mask)
    pad = _BORDER_INK_SEARCH_MARGIN_PX
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(raw_mask.shape[1], x + w + pad), min(raw_mask.shape[0], y + h + pad)

    sub_raw = raw_mask[y0:y1, x0:x1]
    sub_eroded = eroded_mask[y0:y1, x0:x1]
    sub_gray = border_gray[y0:y1, x0:x1]

    dark = (sub_gray < _BORDER_INK_DARK_THRESH).astype(np.uint8)
    if not dark.any():
        return eroded_mask

    # NB: deliberatamente NESSUNA chiusura morfologica qui. Richiudere le
    # interruzioni di un tratto di contorno sottile (anti-aliasing, rumore di
    # compressione) aiuterebbe a riconoscerlo come un'unica componente anche
    # quando e' spezzato, ma rischia di fondere nella stessa componente anche
    # una riga di testo tradotto vicina al bordo (comune nei balloon stretti,
    # dove il testo arriva quasi a ridosso del contorno): un solo pixel di
    # quella riga che tocca l'esterno protegge per errore l'intera riga di
    # testo. Meglio lasciare un contorno occasionalmente "a tratti" su
    # dettagli decorativi rari (svirgole, code) che rischiare di lasciare
    # testo originale non cancellato: il secondo caso e' un difetto ben piu'
    # grave del primo.
    num_labels, labels = cv2.connectedComponents(dark, connectivity=8)
    if num_labels <= 1:
        return eroded_mask

    to_erase_bool = sub_eroded > 0
    outside_raw_bool = sub_raw == 0

    # Distanza di ogni pixel interno dal perimetro vero (piu' vicino e' a
    # zero, piu' e' vicino all'esterno): limita la protezione a chi sta
    # davvero a ridosso del contorno, vedi _BORDER_INK_MAX_PROTECT_DEPTH_PX.
    dist_from_outside = cv2.distanceTransform(sub_raw, cv2.DIST_L2, 5)
    near_border_bool = dist_from_outside <= _BORDER_INK_MAX_PROTECT_DEPTH_PX

    protect = np.zeros_like(sub_eroded)
    for lbl in range(1, num_labels):
        comp = labels == lbl
        if not np.any(comp & to_erase_bool):
            continue  # non tocca l'area che stiamo per cancellare, irrilevante
        if np.any(comp & outside_raw_bool):
            protect[comp & near_border_bool] = 255  # si estende oltre la sagoma vera: e' il contorno

    protect = cv2.bitwise_and(protect, sub_eroded)
    if not protect.any():
        return eroded_mask

    result = eroded_mask.copy()
    result[y0:y1, x0:x1] = cv2.bitwise_and(sub_eroded, cv2.bitwise_not(protect))
    return result


def _should_extend_to_bbox(det: dict) -> bool:
    """
    True solo per un bbox allargato A MANO in Revisione (manual_text_box
    senza balloon_split): in quel caso l'area "guadagnata" va davvero unita
    alla maschera di pulizia, vedi extend_to_bbox in _resolve_balloon_mask.

    Un balloon nato da uno split di due bubble fusi (balloon_split) ha
    manual_text_box=True per un motivo diverso e non correlato: dice a
    render.py di non riespandere la maschera (vedi _split_fused_bubbles in
    detect.py). Il suo bbox e' gia' il rettangolo stretto attorno alla meta'
    di maschera, non un box allargato oltre la sagoma segmentata: unirlo
    (anche con il margine limitato di extend_to_bbox) spinge la maschera di
    pulizia oltre la curva reale del balloon nei punti dove il rettangolo
    "taglia gli angoli" fuori dall'ovale, e li' il flood-fill dipinge di
    colore di sfondo l'artwork sottostante invece del balloon.
    """
    return bool(det.get("manual_text_box")) and not det.get("balloon_split")


def _resolve_balloon_mask(
    mask_path: str, image_shape: tuple[int, int], bbox: list[int], extend_to_bbox: bool = False,
    erode_px: int = 0, border_gray: np.ndarray | None = None, extra_dilate_px: int = 0,
) -> np.ndarray:
    """
    Carica la maschera del balloon da disco. comic-text-detector la salva
    a piena pagina (con solo il balloon isolato al suo interno), quindi la
    dimensione attesa e' quella dell'immagine, non del bbox. Se il file manca
    o ha una dimensione inattesa (es. incompatibilita' con una versione
    diversa della pipeline), genera un fallback a piena pagina con un
    rettangolo pieno sul bbox corrente.

    Se extend_to_bbox e' True (bbox ridimensionato a mano in Revisione),
    unisce alla maschera anche il rettangolo del bbox: altrimenti l'area
    "guadagnata" allargando il box a mano non verrebbe mai pulita
    dall'inpainting, lasciando visibile l'artwork/sfondo originale sotto
    al testo quando il box supera i bordi reali del balloon.
    """
    h, w = image_shape[:2]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (h, w):
        if mask is not None:
            log.warning(
                f"Maschera {mask_path} con dimensioni inattese {mask.shape} "
                f"(attese {(h, w)}). Uso un rettangolo pieno sul bbox come fallback."
            )
        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    if extend_to_bbox:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        rect = np.zeros((h, w), dtype=np.uint8)
        rect[y1:y2, x1:x2] = 255
        # Limita l'estensione a una fascia vicina al vero contorno del
        # balloon (mask dilatata di un margine), invece di riempire tutto il
        # rettangolo del bbox: un box spostato/allargato a mano in Revisione
        # puo' finire in parte ben fuori dalla sagoma ovale reale, e
        # riempire quella zona produrrebbe un rettangolo bianco a spigoli
        # vivi che sporge visibilmente dal balloon.
        if mask.any():
            margin_px = max(20, int(0.06 * min(x2 - x1, y2 - y1)))
            kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), np.uint8)
            allowed = cv2.dilate(mask, kernel)
            rect = cv2.bitwise_and(rect, allowed)
        mask = cv2.bitwise_or(mask, rect)

    if erode_px > 0:
        raw_mask = mask
        kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        mask = cv2.erode(mask, kernel)
        mask = _protect_border_ink(mask, raw_mask, border_gray)

    if extra_dilate_px > 0:
        kernel = np.ones((extra_dilate_px * 2 + 1, extra_dilate_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)

    return mask


_COMFYUI_DRIFT_MARGIN_PX = 5
"""
Il workflow ComfyUI rigenera l'intera pagina (ridimensionamento interno a
megapixel fissi, poi ricampionamento alla risoluzione originale), a
differenza di IOPaint che modificava solo l'area mascherata lasciando il
resto invariato pixel per pixel. Questo introduce un possibile drift di
pochi pixel tra la maschera (calcolata sull'immagine originale, pre-pulizia)
e la posizione reale del testo nell'immagine pulita. Su balloon larghi non
si nota, ma su didascalie sottili (poche decine di pixel di altezza) basta
lo scarto per lasciare fuori dalla maschera un filo di testo residuo. Le
rifiniture post-pulizia allargano quindi la maschera di questo margine per
assorbirlo.
"""

_BALLOON_SHAPE_ERODE_PX = 6
"""
La maschera di bubble_seg segue il contorno esterno REALE del balloon,
bordo nero incluso. Erodendo la maschera si esclude il bordo dall'area che
consideriamo "interno del balloon" (per il ripristino dei balloon saltati e
per le rifiniture di colore), cosi' non si tocca mai il contorno nitido
originale. Le maschere di comic-text-detector non necessitano di questa
erosione: seguono solo il testo, gia' ben dentro il bianco del balloon.
"""

_DARK_BALLOON_BACKGROUND_THRESH = 150
"""Luminosita' sotto la quale un balloon/caption ha fondo scuro.

Su un caption nero il fondo e il bordo sono un'unica componente scura che
prosegue all'esterno della maschera. `_protect_border_ink` la scambierebbe
quindi per solo contorno e la escluderebbe interamente dalla pulizia,
lasciando intatte le lettere bianche originali.
"""


def _restore_skipped_balloons(
    cleaned: np.ndarray, original: np.ndarray, mask_paths: list[str], detections: list[dict],
    border_gray: np.ndarray | None = None,
) -> np.ndarray:
    """
    Il workflow ComfyUI pulisce l'intera pagina in un solo passaggio guidato
    da prompt testuale, senza maschere per-balloon: cancella indiscriminatamente
    il testo da OGNI balloon che riconosce, incluso quelli senza una traduzione
    valida (onomatopee, testo vuoto, '-'). Per quei balloon la pipeline non
    deve mostrare un interno ripulito senza nulla sopra: ripristina i pixel
    originali nella loro area, cosi' restano identici alla pagina di partenza.
    """
    if not mask_paths:
        return cleaned
    result = cleaned.copy()
    h, w = result.shape[:2]
    for mp, det in zip(mask_paths, detections):
        erode_px = _BALLOON_SHAPE_ERODE_PX if det.get("balloon_source") == "yolov8seg" else 0
        mask = _resolve_balloon_mask(
            mp, (h, w), det["bbox"], extend_to_bbox=_should_extend_to_bbox(det), erode_px=erode_px,
            border_gray=border_gray,
        )
        mask_bool = mask > 0
        result[mask_bool] = original[mask_bool]
    return result


def _dominant_color(pixels: np.ndarray) -> np.ndarray:
    """Colore di picco dell'istogramma quantizzato: a differenza di una media
    o di un KMeans a 2 gruppi, individua il colore piu' frequente nel
    campione senza doverlo separare nettamente dal resto. Robusto anche
    quando testo e sfondo hanno basso contrasto (es. testo grigio scuro su
    sfondo quasi nero), a patto che il campione passato sia gia' a
    maggioranza sfondo (vedi _call_floodfill per come viene costruito)."""
    bucket = 8
    q = pixels // bucket
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    peak_key = uniq[np.argmax(counts)]
    return pixels[keys == peak_key].mean(axis=0)


def _call_floodfill(
    image: np.ndarray, mask_paths: list[str], detections: list[dict], border_gray: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pulizia deterministica senza modelli generativi: per ogni balloon,
    riempie l'area di testo con il colore di sfondo dominante campionato
    dall'immagine originale. Nessun rischio di dettagli/testo inventati
    (visto con Flux su balloon a sfondo piatto), ma non ricostruisce
    sfumature/texture: adatta a balloon e caption box a tinta piena.

    Le maschere di comic-text-detector (balloon_source="text_detector", o
    assente: pagine elaborate prima che questo campo esistesse) seguono solo
    i tratti del testo, non tutto l'interno del box (vedi
    _BALLOON_SHAPE_ERODE_PX): campionare il colore DENTRO quella maschera
    prenderebbe quasi solo pixel di testo. Per quei balloon si campiona
    invece dal bbox ESCLUSA la maschera del testo, e si riempie l'intero
    bbox (i caption box sono di norma rettangoli piatti). Solo quando il
    campo dichiara esplicitamente "yolov8seg" (maschera sicuramente a piena
    forma del balloon) si campiona/riempie dentro la maschera stessa.

    I bounding box del detector sono spesso stretti al testo e possono
    perdere uno o due pixel di anti-aliasing, soprattutto sotto l'ultima
    riga. Per i box testuali aggiungiamo quindi una piccola fascia di
    sicurezza: senza di essa il flood-fill puo' lasciare il bordo inferiore
    delle lettere originali, poi visibile sotto alla traduzione appena
    renderizzata. Il padding e' limitato e si usa solo per i box che non
    rappresentano l'intera sagoma del balloon, quindi non invade il disegno.
    """
    result = image.copy()
    h, w = result.shape[:2]

    # Le maschere grezze servono tutte insieme prima del ciclo: l'assegnazione
    # di un testo al balloon giusto e' un confronto tra balloon, non una
    # decisione locale (vedi balloon_shape.text_ink_outside_balloons).
    raw_masks: list[np.ndarray | None] = [
        _resolve_balloon_mask(
            mp, (h, w), det["bbox"], extend_to_bbox=_should_extend_to_bbox(det),
        )
        if det.get("balloon_source") == "yolov8seg" else None
        for mp, det in zip(mask_paths, detections)
    ]
    extra_inks = balloon_shape.text_ink_outside_balloons(
        balloon_shape.load_text_masks(mask_paths, (h, w)), raw_masks
    )

    for idx, (mp, det) in enumerate(zip(mask_paths, detections)):
        x1, y1, x2, y2 = det["bbox"]
        is_full_balloon_mask = det.get("balloon_source") == "yolov8seg"

        if not is_full_balloon_mask:
            text_mask = _resolve_balloon_mask(
                mp, (h, w), det["bbox"], extend_to_bbox=_should_extend_to_bbox(det),
            )
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            box_w, box_h = x2 - x1, y2 - y1
            # 6 px coprono bene anti-aliasing e lievi errori OCR, mentre il
            # tetto evita di allargare in modo eccessivo didascalie minute.
            padding = max(4, min(8, round(0.08 * min(box_w, box_h))))
            xx1, yy1 = max(0, x1 - padding), max(0, y1 - padding)
            xx2, yy2 = min(w, x2 + padding), min(h, y2 + padding)
            bbox_mask[yy1:yy2, xx1:xx2] = 255
            bg_region = cv2.bitwise_and(bbox_mask, cv2.bitwise_not(text_mask))
            ys, xs = np.where(bg_region > 0)
            if len(ys) == 0:
                continue
            bg_color = _dominant_color(image[ys, xs].astype(np.int32))
            result[yy1:yy2, xx1:xx2] = bg_color.astype(np.uint8)
        else:
            # Prima rileva i balloon a fondo scuro usando la maschera grezza
            # (non erosa), SENZA l'euristica del contorno. Su un caption nero
            # quell'euristica protegge erroneamente tutto il rettangolo nero
            # (che e' connesso al bordo), quindi il flood-fill non arriva mai
            # alle lettere bianche.
            #
            # La polarita' va campionata sulla maschera grezza e non su quella
            # gia' erosa di _BALLOON_SHAPE_ERODE_PX: su un caption sottile
            # (poche decine di px di altezza) erodere 6px sopra e sotto lascia
            # una fascia cosi' stretta da essere occupata quasi per intero dal
            # corpo delle lettere maiuscole invece che dallo sfondo, facendo
            # risultare dominante il bianco del testo e classificando per
            # errore come "chiaro" un balloon in realta' scuro. Sulla maschera
            # grezza lo sfondo resta maggioranza anche in quel caso (il testo
            # occupa comunque meno area del riquadro).
            raw_for_polarity = raw_masks[idx]
            raw_bool = raw_for_polarity > 0
            if not raw_bool.any():
                continue
            bg_color_check = _dominant_color(image[raw_bool].astype(np.int32))
            if float(bg_color_check.mean()) < _DARK_BALLOON_BACKGROUND_THRESH:
                # L'erosione qui serve solo a non toccare il contorno nitido
                # del balloon, non a isolare lo sfondo dal testo (gia' fatto
                # sopra sulla maschera grezza).
                inner_mask = _resolve_balloon_mask(
                    mp, (h, w), det["bbox"], extend_to_bbox=_should_extend_to_bbox(det),
                    erode_px=_BALLOON_SHAPE_ERODE_PX, border_gray=None,
                )
                if extra_inks[idx] is not None:
                    inner_mask = cv2.bitwise_or(inner_mask, extra_inks[idx])
                inner_bool = inner_mask > 0
                if not inner_bool.any():
                    continue
                result[inner_bool] = bg_color_check.astype(np.uint8)
                continue

            mask = _resolve_balloon_mask(
                mp, (h, w), det["bbox"], extend_to_bbox=_should_extend_to_bbox(det),
                erode_px=_BALLOON_SHAPE_ERODE_PX, border_gray=border_gray,
            )
            mask_bool = mask > 0
            if not mask_bool.any():
                continue
            bg_color = _dominant_color(image[mask_bool].astype(np.int32))

            # Tratti di testo lasciati fuori dalla sagoma segmentata (vedi
            # balloon_shape.text_ink_outside_balloons). Rientrano nell'area da riempire,
            # devono ripassare da _protect_border_ink: dilatati, possono
            # sfiorare il contorno del balloon. La sagoma "vera" passata al
            # controllo comprende ora anche questi tratti, cosi' le lettere
            # non risultano inchiostro di contorno e restano da cancellare.
            extra_ink = extra_inks[idx]
            if extra_ink is not None:
                mask = _protect_border_ink(
                    cv2.bitwise_or(mask, extra_ink),
                    cv2.bitwise_or(raw_for_polarity, extra_ink),
                    border_gray,
                )
                mask_bool = mask > 0

            result[mask_bool] = bg_color.astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
def run(page_path: Path, translated_json_path: Path, cfg: dict, work_dir: Path) -> Path:
    with open(translated_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    image = cv2.imread(str(page_path))
    if image is None:
        raise FileNotFoundError(f"Impossibile leggere immagine: {page_path}")

    valid_detections, skipped_detections = [], []
    for det in data["detections"]:
        text = det.get("testo_tradotto", "").strip()
        (valid_detections if text and text != "-" else skipped_detections).append(det)

    # Un balloon nato dallo split di due bubble_seg fusi
    # (detect.py::_split_fused_bubbles) porta con se' `parent_mask_path`, la
    # maschera fusa da cui e' stato tagliato: e' QUELLA l'area da pulire.
    # Le maschere delle singole meta' escludono il margine di sicurezza
    # scavato tra loro (e assorbono ogni imprecisione del taglio),
    # quindi pulire meta' per meta' lasciava intatte le lettere originali che
    # cadono lungo il taglio — poi visibili sotto la traduzione renderizzata.
    # Il taglio serve solo a impaginare due testi in due aree distinte
    # (render.py), non a decidere cosa cancellare.
    #
    # Le meta' senza testo tradotto restano invece sulla propria maschera:
    # ripristinare la fusa "annullerebbe" anche la pulizia della meta'
    # gemella che il testo ce l'ha.
    def _clean_mask_path(det: dict) -> str:
        return det.get("parent_mask_path") or det["mask_path"]

    mask_detections, mask_paths, _seen_masks = [], [], set()
    for det in valid_detections:
        if "mask_path" not in det:
            continue
        mp = _clean_mask_path(det)
        if mp in _seen_masks:
            continue  # meta' gemelle dello stesso balloon fuso: una sola passata
        _seen_masks.add(mp)
        mask_detections.append(det)
        mask_paths.append(mp)
    skipped_mask_detections = [det for det in skipped_detections if "mask_path" in det]
    skipped_mask_paths = [det["mask_path"] for det in skipped_mask_detections]

    skipped = len(skipped_detections)
    if skipped > 0:
        log.info(f"{page_path.name}: {skipped} balloon senza testo tradotto, ripristinati dopo la pulizia")

    page_work_dir = work_dir / data["page_id"]

    if not mask_paths:
        log.info(f"{page_path.name}: nessun balloon da pulire, copio pagina originale")
        cleaned = image
    else:
        border_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        log.info(f"{page_path.name}: pulizia pagina ({len(mask_paths)} balloon)")
        cleaned = _call_floodfill(image, mask_paths, mask_detections, border_gray=border_gray)
        cleaned = _restore_skipped_balloons(
            cleaned, image, skipped_mask_paths, skipped_mask_detections, border_gray=border_gray
        )

    cleaned_path = page_work_dir / "cleaned.png"
    cv2.imwrite(str(cleaned_path), cleaned)
    return cleaned_path
