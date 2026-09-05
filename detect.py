import json
import shutil
import subprocess
import logging
from pathlib import Path

import cv2
import numpy as np

import balloon_shape
import paths

log = logging.getLogger("pipeline.detect")

CTD_PYTHON = paths.venv_bin(paths.VENV_CTD, "python")
CTD_SCRIPT = str(paths.CTD_SCRIPT)

BUBBLESEG_PYTHON = paths.venv_bin(paths.VENV_BUBBLESEG, "python")
BUBBLESEG_SCRIPT = str(paths.BUBBLESEG_SCRIPT)
BUBBLESEG_MODEL = str(paths.BUBBLESEG_MODEL_DEFAULT)


def _run_subprocess(cmd: list[str], timeout: int, label: str) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        log.error(f"Timeout ({timeout}s) durante {label}")
        if e.stdout:
            log.error(f"stdout parziale: {e.stdout}")
        if e.stderr:
            log.error(f"stderr parziale: {e.stderr}")
        raise RuntimeError(f"{label} in timeout") from e
    except OSError as e:
        log.error(f"Errore di sistema avviando il subprocess per {label}: {e}")
        raise RuntimeError(f"Impossibile avviare {label}") from e

    if result.stdout:
        log.info(f"[{label} stdout] {result.stdout.strip()}")
    if result.stderr:
        log.warning(f"[{label} stderr] {result.stderr.strip()}")

    return result


def _run_ctd(page_path: Path, cfg: dict, page_work_dir: Path) -> Path:
    """
    Esegue detection+segmentazione testo tramite comic-text-detector, in un
    processo Python separato (venv_ctd) per evitare conflitti di dipendenze
    con l'ambiente principale della pipeline. La maschera prodotta segue
    solo il testo (bbox testo + dilation fissa), non la forma reale del
    balloon: viene eventualmente rifinita da _merge_with_bubble_shapes.
    """
    det_cfg = cfg.get("detection", {})
    dilation_px = det_cfg.get("mask_dilation_px", 9)

    masks_dir = page_work_dir / "masks"
    detections_json_path = page_work_dir / "detections.json"

    cmd = [
        CTD_PYTHON,
        CTD_SCRIPT,
        "--image", str(page_path.resolve()),
        "--output-json", str(detections_json_path.resolve()),
        "--masks-dir", str(masks_dir.resolve()),
        "--dilation", str(dilation_px),
    ]

    log.info(f"Avvio comic-text-detector per {page_path.name}...")
    result = _run_subprocess(cmd, timeout=180, label="comic-text-detector")

    if result.returncode != 0:
        log.error(
            f"comic-text-detector fallito per {page_path.name} "
            f"(returncode={result.returncode})"
        )
        raise RuntimeError(f"Detection fallita per {page_path.name} (returncode={result.returncode})")

    if not detections_json_path.exists():
        log.error(
            f"comic-text-detector e' terminato con successo (returncode=0) "
            f"ma non ha prodotto il file atteso: {detections_json_path}"
        )
        raise RuntimeError(f"File detections.json mancante per {page_path.name}")

    return detections_json_path


def _run_bubbleseg(page_path: Path, cfg: dict, page_work_dir: Path) -> Path | None:
    """
    Esegue kitsumed/yolov8m_seg-speech-bubble (venv_bubbleseg) per segmentare
    la forma REALE dei balloon (bordo+interno), a differenza di
    comic-text-detector che segmenta solo il testo. Fallimenti qui non sono
    fatali per la pipeline: se il modello non e' disponibile o va in errore,
    si prosegue con le sole maschere di comic-text-detector (comportamento
    precedente all'integrazione).
    """
    bs_cfg = cfg.get("bubble_seg", {})
    if not bs_cfg.get("enabled", True):
        return None

    model_path = paths.resolve(bs_cfg.get("model_path", BUBBLESEG_MODEL))
    if not model_path.exists():
        log.warning(f"Modello bubble_seg non trovato ({model_path}), salto la segmentazione balloon reale.")
        return None

    conf = bs_cfg.get("conf", 0.25)
    masks_dir = page_work_dir / "bubble_masks"
    bubbles_json_path = page_work_dir / "bubbles.json"

    # Svuota le maschere di una run precedente: bubble_seg numera i file da zero
    # ad ogni run, quindi senza pulizia i vecchi PNG orfani (es. lobi pre-merge)
    # restano sul disco e possono essere referenziati da stage a valle stale
    # (translated.json) che non vengono rigenerati.
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        BUBBLESEG_PYTHON,
        BUBBLESEG_SCRIPT,
        "--image", str(page_path.resolve()),
        "--output-json", str(bubbles_json_path.resolve()),
        "--masks-dir", str(masks_dir.resolve()),
        "--model", str(model_path),
        "--conf", str(conf),
    ]

    log.info(f"Avvio bubble_seg (yolov8m-seg) per {page_path.name}...")
    try:
        result = _run_subprocess(cmd, timeout=120, label="bubble_seg")
    except RuntimeError as e:
        log.warning(f"bubble_seg non disponibile per {page_path.name} ({e}), proseguo senza.")
        return None

    if result.returncode != 0 or not bubbles_json_path.exists():
        log.warning(f"bubble_seg fallito per {page_path.name} (returncode={result.returncode}), proseguo senza.")
        return None

    return bubbles_json_path


def _dedupe_overlapping_bubbles(bubbles_json_path: Path, containment_ratio: float = 0.8) -> None:
    """
    bubble_seg a volte rileva lo stesso balloon fisico due volte: una
    maschera completa e una seconda, piu' piccola, quasi interamente
    contenuta nella prima (limite noto della NMS del modello su forme
    allungate/doppie, dove le due detection si sovrappongono ma non
    abbastanza da essere scartate come duplicate). Se lasciate cosi', in
    _merge_with_bubble_shapes il testo puo' finire associato alla maschera
    piu' piccola (parziale) invece di quella vera, e la piu' piccola
    fuorvia anche _split_fused_bubbles facendogli credere che ci sia una
    fusione dove in realta' c'e' solo un duplicato.

    Va eseguita PRIMA di _split_fused_bubbles: quella si aspetta che ogni
    maschera rappresenti un solo balloon fisico (fuso con altri o meno),
    non un duplicato parziale di un'altra maschera gia' presente.

    E va rieseguita anche DOPO: se bubble_seg ha rilevato lo stesso balloon
    sia da solo sia dentro una maschera fusa col vicino, prima dello split le
    due non si somigliano abbastanza (la fusa e' molto piu' grande), mentre
    dopo lo split una delle meta' e' di fatto una copia della maschera
    singola. La funzione e' idempotente, quindi rieseguirla non ha
    controindicazioni.
    """
    with open(bubbles_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bubbles = data.get("bubbles", [])
    if len(bubbles) < 2:
        return

    masks = [cv2.imread(b["mask_path"], cv2.IMREAD_GRAYSCALE) for b in bubbles]
    areas = [int((m > 0).sum()) if m is not None else 0 for m in masks]

    drop: set[int] = set()
    for i in range(len(bubbles)):
        if masks[i] is None or i in drop:
            continue
        for j in range(len(bubbles)):
            if i == j or masks[j] is None or j in drop or areas[i] == 0:
                continue
            # "i" e' il candidato piu' piccolo da eliminare se quasi tutta
            # la sua area ricade dentro "j": confrontiamo solo coppie dove
            # j non e' piu' piccolo di i, per non scartare entrambe le
            # maschere di una coppia genuinamente reciproca (raro, ma
            # eviterebbe di azzerare per errore un balloon vero).
            if areas[j] < areas[i]:
                continue
            inter = int(np.logical_and(masks[i] > 0, masks[j] > 0).sum())
            if inter / areas[i] >= containment_ratio:
                drop.add(i)
                break

    if not drop:
        return
    kept = [b for idx, b in enumerate(bubbles) if idx not in drop]
    for i, b in enumerate(kept):
        b["bubble_id"] = i
    data["bubbles"] = kept
    log.info(f"bubble_seg: scartate {len(drop)} maschera/e duplicate (quasi interamente contenute in un'altra)")
    with open(bubbles_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


_MIN_PARTIAL_OVERLAP = 0.25
"""Frazione minima dell'area della maschera piu' piccola che deve ricadere
dentro la piu' grande perche' le due siano considerate lo stesso balloon
visto due volte (e non due balloon che semplicemente si sfiorano)."""

_MIN_REMAINDER_RATIO = 0.15
"""Frazione minima dell'area originale che un frammento deve conservare per
sopravvivere alla sottrazione: sotto questa soglia e' bava del contorno, non
un balloon."""


def _subtract_overlapping_bubbles(
    bubbles_json_path: Path,
    min_overlap: float = _MIN_PARTIAL_OVERLAP,
    containment_ratio: float = 0.8,
) -> None:
    """
    Caso intermedio tra quelli gestiti da _dedupe_overlapping_bubbles e da
    _split_fused_bubbles: bubble_seg rileva UN balloon da solo e, con una
    seconda detection, lo stesso balloon fuso col vicino ma tagliato a
    meta' (la maschera fusa copre solo una parte del primo balloon, piu'
    tutto il secondo).

    Le due maschere si sovrappongono troppo poco perche' la deduplica le
    scarti (la piu' piccola non e' "quasi interamente contenuta" nella
    fusa) e la fusa arriva intatta a _split_fused_bubbles: li' contiene due
    box di testo, quindi viene divisa in due parti che non corrispondono ai
    due balloon fisici - il taglio passa dove la maschera e' monca. Il
    risultato sono tre detection sovrapposte sulla stessa coppia di
    balloon, con testo ripetuto e impaginato su aree sbagliate
    (4 . An Unconventional Couple, pagina 009, doppio balloon in alto a
    sinistra).

    La correzione e' sottrarre dalla maschera grande quella piccola: la
    parte gia' coperta dalla detection singola - piu' pulita, perche' segue
    il balloon intero - resta a lei, e alla maschera fusa rimane solo il
    balloon che nessun altro copre. Se dopo la sottrazione non resta nulla
    di significativo, la maschera grande era solo un duplicato mal
    ritagliato e viene scartata.

    Va eseguita DOPO _dedupe_overlapping_bubbles (i duplicati veri sono gia'
    spariti, qui restano solo le sovrapposizioni parziali) e PRIMA di
    _split_fused_bubbles, che si aspetta maschere non ridondanti.
    """
    with open(bubbles_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bubbles = data.get("bubbles", [])
    if len(bubbles) < 2:
        return

    masks: list[np.ndarray | None] = []
    for b in bubbles:
        m = cv2.imread(b["mask_path"], cv2.IMREAD_GRAYSCALE)
        masks.append(None if m is None else (m > 0))
    areas = [0 if m is None else int(m.sum()) for m in masks]
    orig_areas = list(areas)

    dropped: set[int] = set()
    changed: set[int] = set()
    # Dalla piu' piccola alla piu' grande: le maschere singole (piu' strette)
    # fanno da "sottraendo" e ritagliano quelle fuse, mai il contrario.
    order = sorted(range(len(bubbles)), key=lambda k: areas[k])
    for i in order:
        if masks[i] is None or areas[i] == 0 or i in dropped:
            continue
        for j in order:
            if j == i or masks[j] is None or j in dropped:
                continue
            if areas[j] <= areas[i]:
                continue
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            frac = inter / areas[i]
            if frac < min_overlap or frac >= containment_ratio:
                # Sotto soglia sono balloon distinti che si sfiorano; sopra
                # ci pensa gia' la deduplica per contenimento.
                continue
            remainder = _significant_parts(
                np.logical_and(masks[j], np.logical_not(masks[i])), orig_areas[j]
            )
            if remainder is None:
                dropped.add(j)
                continue
            masks[j] = remainder
            areas[j] = int(remainder.sum())
            changed.add(j)

    if not dropped and not changed:
        return

    kept: list[dict] = []
    for idx, b in enumerate(bubbles):
        if idx in dropped:
            continue
        if idx in changed:
            m = masks[idx]
            cv2.imwrite(b["mask_path"], (m.astype(np.uint8) * 255))
            ys, xs = np.where(m)
            b["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        kept.append(b)
    for i, b in enumerate(kept):
        b["bubble_id"] = i
    data["bubbles"] = kept
    log.info(
        f"bubble_seg: risolte sovrapposizioni parziali "
        f"({len(changed)} maschera/e ritagliata/e, {len(dropped)} scartata/e)"
    )
    with open(bubbles_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _significant_parts(remainder: np.ndarray, orig_area: int) -> np.ndarray | None:
    """
    Ripulisce il risultato di una sottrazione tra maschere dai frammenti di
    contorno: tiene solo le componenti connesse abbastanza grandi rispetto
    all'area originale. Restituisce None se non ne resta nessuna.
    """
    min_part = max(int(orig_area * _MIN_REMAINDER_RATIO), 300)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        remainder.astype(np.uint8), 8
    )
    out = np.zeros_like(remainder)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_part:
            out |= labels == k
    return out if out.any() else None


_MIN_CONCAVITY_DEPTH_PX = 8.0
"""Profondita' minima (px) perche' una rientranza del contorno conti come
punto di contatto tra due balloon fusi e non come frastagliatura della
segmentazione."""


def _deep_concavities(sub_bin: np.ndarray) -> list[tuple[float, tuple[int, int]]]:
    """
    Punti di massima rientranza del contorno della maschera (convexity
    defects), dal piu' profondo al meno profondo, scartati quelli sotto
    soglia. Due balloon disegnati sovrapposti formano una sagoma "a
    nocciolina": i due punti in cui i loro contorni si incrociano sono
    esattamente le due rientranze piu' profonde, e il segmento tra loro e'
    il confine reale tra i due. Un balloon singolo (ovale, convesso) non ha
    rientranze profonde, la coda esclusa.
    """
    cnts, _ = cv2.findContours(sub_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    contour = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 3:
        return []
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return []
    if defects is None:
        return []
    h, w = sub_bin.shape
    min_depth = max(_MIN_CONCAVITY_DEPTH_PX, 0.05 * max(h, w))
    out = []
    for row in defects.reshape(-1, 4):
        _s, _e, far_idx, depth_fp = (int(v) for v in row)
        depth = depth_fp / 256.0  # convexityDefects riporta in virgola fissa 8.8
        if depth >= min_depth:
            out.append((depth, tuple(int(v) for v in contour[far_idx][0])))
    out.sort(key=lambda t: -t[0])
    return out


def _label_at(labels: np.ndarray, point: tuple[float, float], valid: set[int]) -> int | None:
    """Etichetta della regione in cui cade `point`. Il centro di un box di
    testo puo' finire su una lettera o proprio sulla linea di taglio (non
    etichettate): si allarga allora la ricerca a un piccolo intorno."""
    h, w = labels.shape
    cx, cy = int(round(point[0])), int(round(point[1]))
    for r in range(0, 21, 2):
        window = labels[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
        found = [int(v) for v in np.unique(window) if int(v) in valid]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            return None  # intorno a cavallo del taglio: ambiguo
    return None


def _cut_by_concavity(
    sub_bin: np.ndarray, seeds_local: list[tuple[float, float]],
    min_area: float, cut_canvas: np.ndarray,
) -> list[tuple[np.ndarray, list[tuple[float, float]]]] | None:
    """
    Divide `sub_bin` in DUE lungo il segmento che unisce due rientranze
    profonde del contorno, scegliendo la coppia che separa davvero i semi
    (almeno uno per parte) e lascia due parti non degeneri. Disegna il
    taglio scelto su `cut_canvas` (serve poi a scavare il margine di
    sicurezza tra le parti). Ritorna le due parti con i rispettivi semi, o
    None se nessuna coppia di rientranze separa i semi.
    """
    candidates = _deep_concavities(sub_bin)[:6]
    best = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            depth_i, p1 = candidates[i]
            depth_j, p2 = candidates[j]
            cut = sub_bin.copy()
            cv2.line(cut, p1, p2, 0, 3)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=4)
            big = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_area]
            if len(big) != 2:
                continue
            valid = set(big)
            groups: dict[int, list[tuple[float, float]]] = {k: [] for k in big}
            for seed in seeds_local:
                lbl = _label_at(labels, seed, valid)
                if lbl is None:
                    groups = {}
                    break
                groups[lbl].append(seed)
            if not groups or any(not g for g in groups.values()):
                continue
            # A parita' di separazione valida, si tiene il taglio piu' netto:
            # quello tra le due rientranze piu' profonde.
            score = min(depth_i, depth_j)
            if best is None or score > best[0]:
                best = (score, labels, big, groups, (p1, p2))
    if best is None:
        return None
    _score, labels, big, groups, (p1, p2) = best
    cv2.line(cut_canvas, p1, p2, 255, 3)
    return [((labels == k).astype(np.uint8), groups[k]) for k in big]


def _split_by_concavity(
    sub_bin: np.ndarray, seeds_local: list[tuple[float, float]],
    min_area: float, cut_canvas: np.ndarray,
) -> list[np.ndarray] | None:
    """Applica _cut_by_concavity ricorsivamente finche' ogni parte non ha un
    solo seme: una fusione tripla si separa con due tagli successivi."""
    if len(seeds_local) <= 1:
        return [sub_bin]
    cut = _cut_by_concavity(sub_bin, seeds_local, min_area, cut_canvas)
    if cut is None:
        return None
    parts: list[np.ndarray] = []
    for region, region_seeds in cut:
        sub_parts = _split_by_concavity(region, region_seeds, min_area, cut_canvas)
        if sub_parts is None:
            return None
        parts.extend(sub_parts)
    return parts


def _assemble_parts(
    mask: np.ndarray, bbox: list[int], regions: list[np.ndarray], cut: np.ndarray,
    total_area: int, min_area_ratio: float,
) -> list[tuple[np.ndarray, list[int]]] | None:
    """
    Trasforma le regioni trovate da uno dei metodi di split (coordinate del
    ritaglio `bbox`) nelle maschere a piena pagina + bbox che il chiamante si
    aspetta, scavando tra loro un margine di sicurezza attorno a `cut` (la
    linea di taglio).

    Il margine serve perche' render.py applica alle maschere una chiusura
    morfologica di 9px (_MASK_SMOOTH_PX) per smussare i bordi "a blocchi"
    della segmentazione: due parti che si toccano con gap 0 tornerebbero a
    contatto dopo quella chiusura, e il testo adattato a ciascuna si
    sovrapporrebbe.

    La soglia di area minima si verifica sulla regione PRIMA di scavare il
    margine: altrimenti un lobo genuino ma piccolo scende sotto soglia solo
    per via del margine, che e' un accorgimento per il rendering a valle e
    non una misura della plausibilita' del lobo.
    """
    x1, y1, x2, y2 = bbox
    gap_margin = max(12, int(min(y2 - y1, x2 - x1) * 0.03))
    gap_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gap_margin * 2 + 1, gap_margin * 2 + 1)
    )
    gap_zone = cv2.dilate(cut, gap_kernel) > 0

    min_area = total_area * min_area_ratio
    parts = []
    for region_full in regions:
        if int(region_full.sum()) < min_area:
            return None
        region = region_full & ~gap_zone
        if not region.any():
            return None
        part_local = np.where(region, 255, 0).astype(np.uint8)
        full = np.zeros_like(mask)
        full[y1:y2, x1:x2] = part_local
        ys, xs = np.where(part_local > 0)
        parts.append((
            full,
            [x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1],
        ))
    return parts


def _split_bubble_by_seeds(
    mask: np.ndarray, bbox: list[int], seeds: list[tuple[float, float]],
    min_comp_area: int = 150, min_area_ratio: float = 0.05, max_erode_iter: int = 120,
) -> list[tuple[np.ndarray, list[int]]] | None:
    """
    Divide `mask` (ristretta a `bbox`) in tante regioni quanti `seeds`
    (centri dei box di testo CTD, coordinate immagine intera, uno per box —
    NON raggruppati in cluster: in molti fumetti due balloon distinti e
    vicini distano solo pochi pixel, meno di quanto separi normalmente due
    righe di testo dello stesso balloon, quindi un clustering per
    prossimita' finiva per fondere insieme due semi che dovevano restare
    separati).

    Due metodi, in ordine:

    1. Taglio tra rientranze profonde del contorno (_split_by_concavity).
       Due balloon disegnati sovrapposti formano una sagoma "a nocciolina":
       i punti dove i due contorni si incrociano sono le due rientranze piu'
       profonde, e il segmento tra loro e' il confine reale. Un balloon
       singolo (convesso) non ha rientranze profonde, quindi non si divide.
       E' il metodo primario perche' copre la fusione piu' comune, quella a
       ovali molto sovrapposti, dove non esiste nessun collo stretto.

    2. Test topologico per erosione: si erode progressivamente la maschera
       finche' non si spezza in almeno tanti componenti connessi quanti i
       semi, poi ogni componente si assegna al seme piu' vicino
       (assegnazione biunivoca greedy) e si ri-cresce fino alla sagoma
       originale via watershed sulla distance transform. Riconosce solo le
       fusioni con un collo davvero stretto, ma li' segue una linea di
       taglio curva invece che il segmento dritto del metodo 1.

    Ritorna None se nessuno dei due separa i semi, o se una regione finale
    e' troppo piccola (seme caduto vicino al bordo): il chiamante ricade sul
    metodo geometrico a strozzatura (_split_one_bubble).
    """
    x1, y1, x2, y2 = bbox
    sub = mask[y1:y2, x1:x2]
    n_seeds = len(seeds)
    if sub.size == 0 or n_seeds < 2:
        return None
    sub_bin = (sub > 0).astype(np.uint8)
    total_area = int(sub_bin.sum())
    if total_area == 0:
        return None

    seeds_local = [(sx - x1, sy - y1) for sx, sy in seeds]

    # Metodo primario: taglio tra le rientranze profonde del contorno (vedi
    # _split_by_concavity). Il test topologico per erosione qui sotto
    # riconosce solo le fusioni con un collo stretto, ma la forma di fusione
    # piu' comune e' due ovali che si sovrappongono in profondita': li' non
    # esiste nessun collo (la maschera non si spezza a nessuna profondita' di
    # erosione) e l'unico segnale sono le due rientranze dove i contorni dei
    # due balloon si incrociano.
    concavity_cut = np.zeros(sub_bin.shape, dtype=np.uint8)
    regions = _split_by_concavity(
        sub_bin, seeds_local, total_area * min_area_ratio, concavity_cut
    )
    if regions is not None and len(regions) == n_seeds:
        return _assemble_parts(
            mask, bbox, [r > 0 for r in regions], concavity_cut, total_area, min_area_ratio
        )

    found = None
    kernel = np.ones((3, 3), np.uint8)
    for it in range(1, max_erode_iter + 1):
        # borderValue=0: senza, cv2.erode tratta l'esterno del ritaglio come
        # foreground (per l'erosione il default e' il valore massimo), e la
        # maschera non viene erosa dove tocca il bordo del bbox — che e' per
        # costruzione sempre, essendo il bbox il bounding box della maschera.
        # Restavano cosi' "spuntoni" attaccati ai bordi che falsavano sia il
        # conteggio dei componenti sia i loro centroidi.
        eroded = cv2.erode(
            sub_bin, kernel, iterations=it,
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if eroded.sum() == 0:
            break
        n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
        big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > min_comp_area]
        if len(big) >= n_seeds:
            found = (labels, big)
            break
    if found is None:
        return None
    labels, big = found
    if len(big) > n_seeds:
        # Piu' componenti dei semi attesi (rumore residuo dell'erosione):
        # tiene solo i piu' grandi, presumibilmente i veri lobi.
        big = sorted(big, key=lambda i: -int((labels == i).sum()))[:n_seeds]

    comp_centroids = []
    for cid in big:
        ys, xs = np.where(labels == cid)
        comp_centroids.append((cid, xs.mean(), ys.mean()))

    # Assegnazione biunivoca componente<->seme: greedy sulla distanza
    # centro-centro crescente, cosi' anche se un centroide non e' il piu'
    # vicino in assoluto al "proprio" seme resta comunque una corrispondenza
    #1:1 coerente (mai due semi sullo stesso componente o viceversa).
    pairs = []
    for ci, (_cid, ccx, ccy) in enumerate(comp_centroids):
        # seeds_local, non seeds: i centroidi dei componenti sono in
        # coordinate del ritaglio (sub), non della pagina intera —
        # confrontarli con i semi assoluti dava distanze prive di senso e
        # quindi un'assegnazione componente<->seme casuale.
        for si, (sx, sy) in enumerate(seeds_local):
            d = (ccx - sx) ** 2 + (ccy - sy) ** 2
            pairs.append((d, ci, si))
    pairs.sort(key=lambda p: p[0])
    used_c, used_s = set(), set()
    assign: dict[int, int] = {}
    for _d, ci, si in pairs:
        if ci in used_c or si in used_s:
            continue
        assign[si] = comp_centroids[ci][0]
        used_c.add(ci)
        used_s.add(si)
    if len(assign) != n_seeds:
        return None

    bg_label = n_seeds + 1
    markers = np.zeros(sub_bin.shape, dtype=np.int32)
    markers[sub_bin == 0] = bg_label
    for si in range(n_seeds):
        markers[labels == assign[si]] = si + 1

    # Elevazione per il watershed di ri-crescita: alta ai bordi/fuori sagoma,
    # bassa al centro di ogni lobo (distance transform invertita). Lo
    # spartiacque finisce cosi' nella parte piu' stretta della maschera tra
    # i componenti erosi, che e' il vero punto di contatto tra i balloon.
    dist = cv2.distanceTransform(sub_bin, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elevation = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)
    cv2.watershed(elevation, markers)

    # cv2.watershed marca con -1 TUTTI i confini tra regioni diverse, e per
    # lui anche lo sfondo (bg_label) e' una regione: la linea -1 corre quindi
    # sia lungo il taglio interno tra due lobi sia lungo l'INTERO contorno
    # esterno del balloon. Si tiene solo il confine "interno" — i pixel di
    # bordo che hanno almeno due regioni di testo (1..n_seeds) nel proprio
    # intorno — altrimenti il margine di sicurezza scavato da
    # _assemble_parts diventa un anello lungo tutto il perimetro invece che
    # una fascia sul solo taglio.
    _nb = np.ones((3, 3), np.uint8)
    region_neighbors = np.zeros(sub_bin.shape, dtype=np.uint8)
    for i in range(1, n_seeds + 1):
        region_neighbors += cv2.dilate((markers == i).astype(np.uint8), _nb)
    inner_cut = ((markers == -1) & (region_neighbors >= 2)).astype(np.uint8)

    regions = [(markers == i) & (sub_bin > 0) for i in range(1, n_seeds + 1)]
    return _assemble_parts(mask, bbox, regions, inner_cut, total_area, min_area_ratio)


def _split_one_bubble(mask: np.ndarray, bbox: list[int]) -> tuple[tuple[np.ndarray, list[int]], tuple[np.ndarray, list[int]]] | None:
    """
    Fallback geometrico quando i box di testo CTD non danno un segnale
    utile (0/1 cluster nella maschera): cerca una strozzatura (stesso
    algoritmo di render.py/balloon_shape.py per i balloon a doppio lobo)
    nel profilo di larghezza della maschera, sia in verticale (righe,
    balloon fusi uno sopra l'altro) che in orizzontale (colonne, balloon
    fusi uno accanto all'altro). Se trovata, taglia la maschera in due a
    quel punto. A differenza di _split_bubble_by_seeds, gestisce solo 2
    parti: non e' in grado di rilevare fusioni triple da sola, per questo
    e' il metodo secondario e non il primario.
    Ritorna None se non c'e' una vera strozzatura, o se il taglio produce
    una meta' troppo piccola (rumore del bordo, non un secondo balloon
    vero).
    """
    x1, y1, x2, y2 = bbox
    sub = mask[y1:y2, x1:x2] > 0
    if sub.size == 0:
        return None

    row_widths = balloon_shape.smooth_widths(sub.sum(axis=1).tolist())
    col_widths = balloon_shape.smooth_widths(sub.sum(axis=0).tolist())

    # dip_ratio piu' permissivo del default (0.7, tarato su strozzature nette
    # tipo clessidra in render.py): due balloon distinti fusi da bubble_seg
    # al punto di contatto spesso mostrano un calo di larghezza piu' lieve
    # (~20-30%) di una vera clessidra, non quasi a zero. Vedi balloon_shape.
    # find_waist per i dettagli del parametro.
    _DIP_RATIO = 0.85

    # Una strozzatura vera e' piu' stretta della larghezza TIPICA della
    # sagoma, non solo dei due picchi che la circondano: e' questo che
    # distingue due lobi da un corpo uniforme con due protuberanze ai lati.
    # find_waist confronta la valle solo con i picchi, e sull'asse
    # ortogonale alla coda la coda e' proprio un picco — si somma in altezza
    # al corpo nelle sole colonne che attraversa e diventa il massimo
    # globale, mentre il min_lobe_ratio di find_waist la riconosce come coda
    # solo sull'asse parallelo, dove resta piu' stretta del corpo. Bug
    # osservato su pagina 003 di "An Unconventional Couple 3" ("DID YOU LIKE
    # IT?"): corpo alto 55px, coda 97px e un blob di segmentazione a
    # sinistra 82px facevano passare per strozzatura il corpo stesso, e il
    # balloon veniva tagliato a meta' frase — "TI E'" in un box, "PIACIUTO?"
    # nell'altro. Li' la valle vale il 93% della mediana; sulle fusioni vere
    # misurate resta tra il 29% e l'84%.
    _MAX_WAIST_OVER_MEDIAN = 0.90

    def _waist(widths: list[int]) -> int | None:
        waist = balloon_shape.find_waist(widths, 0, dip_ratio=_DIP_RATIO)
        if waist is None:
            return None
        body = [w for w in widths if w > 0]
        if not body:
            return None
        if widths[waist] > float(np.median(body)) * _MAX_WAIST_OVER_MEDIAN:
            return None
        return waist

    candidates = []
    row_waist = _waist(row_widths)
    if row_waist is not None and max(row_widths, default=0) > 0:
        candidates.append(("row", row_waist, row_widths[row_waist] / max(row_widths)))
    col_waist = _waist(col_widths)
    if col_waist is not None and max(col_widths, default=0) > 0:
        candidates.append(("col", col_waist, col_widths[col_waist] / max(col_widths)))
    if not candidates:
        return None
    # Se entrambi gli assi trovano una strozzatura, usa quella piu'
    # pronunciata (rapporto larghezza-minima/larghezza-massima piu' basso).
    axis, cut, _ratio = min(candidates, key=lambda c: c[2])

    # NB: non si puo' usare come veto il fatto che un box di testo CTD stia
    # a cavallo del taglio. Sembrerebbe la prova che il taglio spezza una
    # frase, ma CTD raggruppa regolarmente in UN SOLO box il testo di due
    # balloon adiacenti (verificato su 3 . An Unconventional Couple 3 pag.
    # 008 "AHHH, LOVE!" + "YOUR MOUTH ON MY BREASTS..." e su 1 . An
    # Unconventional Couple 1 pag. 012 "I HAD AN EROTIC DREAM..." + "AND I
    # WOKE UP..."): li' il box attraversa il confine vero tra i due
    # balloon, e il veto sopprimeva split legittimi.

    # Un taglio esatto sulla riga/colonna della strozzatura fa combaciare le
    # due meta' senza alcun margine: la strozzatura e' stretta ma non a
    # zero (dip_ratio permissivo), quindi entrambe le meta' restano larghe
    # abbastanza vicino al taglio da far collidere visivamente il testo
    # adattato a ciascuna forma. Si scava un piccolo margine vuoto (non
    # assegnato a nessuna delle due) centrato sul taglio, cosi' il fitting
    # del testo di render.py vede due sagome davvero separate.
    n = len(row_widths) if axis == "row" else len(col_widths)
    gap = max(6, int(n * 0.03))
    cut_lo = max(0, cut - gap // 2)
    cut_hi = min(n, cut + gap // 2)

    part_a = np.zeros_like(mask)
    part_b = np.zeros_like(mask)
    if axis == "row":
        part_a[y1:y1 + cut_lo, x1:x2] = np.where(sub[:cut_lo], 255, 0).astype(np.uint8)
        part_b[y1 + cut_hi:y2, x1:x2] = np.where(sub[cut_hi:], 255, 0).astype(np.uint8)
    else:
        part_a[y1:y2, x1:x1 + cut_lo] = np.where(sub[:, :cut_lo], 255, 0).astype(np.uint8)
        part_b[y1:y2, x1 + cut_hi:x2] = np.where(sub[:, cut_hi:], 255, 0).astype(np.uint8)

    total_area = sub.sum()
    area_a = int((part_a > 0).sum())
    area_b = int((part_b > 0).sum())
    if total_area == 0 or area_a < total_area * 0.05 or area_b < total_area * 0.05:
        return None

    def _bbox_of(part: np.ndarray) -> list[int] | None:
        ys, xs = np.where(part > 0)
        if len(xs) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    bbox_a, bbox_b = _bbox_of(part_a), _bbox_of(part_b)
    if bbox_a is None or bbox_b is None:
        return None
    return (part_a, bbox_a), (part_b, bbox_b)


def _split_fused_bubbles(bubbles_json_path: Path, detections_json_path: Path | None = None) -> None:
    """
    bubble_seg (instance segmentation) a volte fonde in un'unica maschera
    due o piu' balloon fisici vicini/che si toccano, invece di segmentarli
    come forme separate (limite noto della segmentazione su oggetti
    ravvicinati). Se lasciato cosi', in _merge_with_bubble_shapes i box di
    testo di comic-text-detector (uno per balloon) finiscono tutti associati
    alla stessa maschera fusa: solo il primo "vince" la forma reale, l'area
    risultante mostra un balloon solo dove ce ne sono piu' di uno, spesso
    con testo duplicato/concatenato/sovrapposto dei balloon originali.

    Il segnale primario per capire in quante parti dividere (e dove) sono i
    box di testo gia' rilevati da comic-text-detector: se una maschera
    bubble_seg contiene 2+ box di testo, si divide via watershed
    (_split_bubble_by_seeds) in tante parti quanti i box — generalizza a
    fusioni triple o piu', a differenza del vecchio metodo a sola
    strozzatura geometrica che tagliava sempre e solo in 2. Ogni box CTD e'
    usato direttamente come seme (non raggruppato per vicinanza: due
    balloon distinti possono distare pochi pixel, meno di quanto separi
    normalmente due righe di testo dello stesso balloon reale) —
    _split_bubble_by_seeds si difende da falsi positivi verificando che
    esista un vero "collo" nella maschera tra ogni coppia di regioni
    adiacenti prima di accettare lo split. Se il segnale di testo non basta
    (0/1 box nella maschera, es. CTD non ha rilevato quel testo), si ricade
    sul metodo geometrico (_split_one_bubble, solo 2 parti).

    Va eseguita PRIMA di _merge_with_bubble_shapes: quella funzione fa gia'
    un matching 1:1 corretto per centro-punto, quindi una volta che qui i
    bubble sono separati, ciascun box di testo si associera' correttamente
    al proprio balloon senza bisogno di altre modifiche a valle.
    """
    with open(bubbles_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bubbles = data.get("bubbles", [])
    if not bubbles:
        return

    text_centers: list[tuple[float, float]] = []
    if detections_json_path is not None:
        try:
            with open(detections_json_path, "r", encoding="utf-8") as f:
                detections = json.load(f).get("detections", [])
            text_centers = [
                ((d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2)
                for d in detections if d.get("bbox")
            ]
        except Exception as e:
            log.warning(f"Lettura box di testo per lo split fallita ({e}), uso solo il metodo geometrico.")

    masks_dir = Path(bubbles[0]["mask_path"]).parent
    # I nomi dei file maschera NON seguono i bubble_id: _dedupe_overlapping_bubbles
    # scarta delle maschere e rinumera i bubble_id da 0, ma i file su disco
    # conservano l'indice originale di bubble_seg. Numerare le nuove maschere a
    # partire da max(bubble_id)+1 sovrascriverebbe quindi il file di un balloon
    # ancora vivo: quel balloon si ritroverebbe con la maschera di un altro, il
    # suo box di testo CTD non lo intercetterebbe piu' per centro-punto e
    # _merge_with_bubble_shapes lo aggiungerebbe come balloon "non associato",
    # producendo due detection sullo stesso testo (doppio riconoscimento).
    # Si parte percio' dopo l'indice piu' alto realmente presente su disco.
    used_indices = set()
    for f in masks_dir.glob("bubble_*.png"):
        suffix = f.stem.split("_")[-1]
        if suffix.isdigit():
            used_indices.add(int(suffix))
    next_file_id = max(used_indices, default=-1) + 1
    next_id = max(next_file_id, max((b["bubble_id"] for b in bubbles), default=-1) + 1)
    result: list[dict] = []
    split_count = 0

    for b in bubbles:
        mask = cv2.imread(b["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            result.append(b)
            continue

        h, w = mask.shape
        seeds_in_mask = [
            (cx, cy) for cx, cy in text_centers
            if 0 <= int(cy) < h and 0 <= int(cx) < w and mask[int(cy), int(cx)] > 0
        ]

        split = None
        if len(seeds_in_mask) >= 2:
            split = _split_bubble_by_seeds(mask, b["bbox"], seeds_in_mask)
        if split is None:
            split = _split_one_bubble(mask, b["bbox"])

        if split is None:
            result.append(b)
            continue
        for mask_part, bbox_part in split:
            new_path = masks_dir / f"bubble_{next_id:03d}.png"
            cv2.imwrite(str(new_path), mask_part)
            result.append({
                "bubble_id": next_id,
                "bbox": bbox_part,
                "confidence": b.get("confidence", 1.0),
                "mask_path": str(new_path),
                # Segna la provenienza da uno split: _merge_with_bubble_shapes
                # la propaga come manual_text_box sul det, per disattivare in
                # render.py il recupero automatico dell'estensione piena del
                # balloon (_expand_mask_to_drawn_balloon, pensato per un
                # bubble_seg che ha sotto-segmentato UN balloon vero). Qui i
                # due balloon fisici hanno spesso l'interno bianco collegato
                # senza un bordo netto tra loro: quel recupero, ignorando lo
                # split, farebbe risalire il flood-fill fino a coprire di
                # nuovo l'intera area fusa per entrambe le meta'.
                "split": True,
                # Maschera fusa di provenienza. La pulizia (clean.py) deve
                # coprire TUTTA l'area fusa: usare le singole meta' lascia
                # scoperto il margine di sicurezza tra loro (gap_zone) e
                # qualunque imprecisione del taglio, e li' il testo
                # originale sopravvive alla pulizia per poi riaffiorare sotto
                # la traduzione. Il taglio serve solo a impaginare due testi
                # in due aree distinte (render.py), non a decidere cosa
                # cancellare.
                "parent_mask_path": b["mask_path"],
            })
            next_id += 1
        split_count += 1

    if split_count:
        for i, b in enumerate(result):
            b["bubble_id"] = i
        data["bubbles"] = result
        log.info(f"bubble_seg: separate {split_count} maschera/e con balloon fusi in maschere indipendenti")
        with open(bubbles_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _merge_with_bubble_shapes(detections_json_path: Path, bubbles_json_path: Path) -> None:
    """
    Per ogni testo rilevato da comic-text-detector, cerca il balloon di
    forma reale (da bubble_seg) che lo contiene: se trovato, sostituisce
    bbox e mask_path del testo con quelli del balloon reale, cosi'
    l'inpainting e il rendering successivi lavorano sulla forma vera del
    balloon invece che sul solo bbox del testo (spesso molto piu' piccolo).
    Se nessun balloon reale corrisponde (es. caption rettangolari, SFX
    fuori balloon), il testo mantiene bbox/maschera di comic-text-detector
    invariati: bubble_seg riconosce solo i veri speech bubble, non ogni
    contenitore di testo. Un balloon rilevato solo da bubble_seg viene
    comunque aggiunto senza testo, cosi' lo stage OCR puo' provare a
    leggerlo invece di lasciarlo inevitabilmente nella lingua originale.
    """
    with open(detections_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(bubbles_json_path, "r", encoding="utf-8") as f:
        bubbles_data = json.load(f)

    bubbles = bubbles_data.get("bubbles", [])
    if not bubbles:
        return

    # Cache delle maschere balloon caricate una volta sola.
    bubble_masks = []
    for b in bubbles:
        mask = cv2.imread(b["mask_path"], cv2.IMREAD_GRAYSCALE)
        bubble_masks.append(mask)

    matched = 0
    claimed_bubble_ids: dict[int, dict] = {}
    deduped: list[dict] = []
    for det in data["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        best = None
        for b, mask in zip(bubbles, bubble_masks):
            if mask is None:
                continue
            h, w = mask.shape
            if not (0 <= cy < h and 0 <= cx < w):
                continue
            if mask[cy, cx] > 0:
                best = b
                break

        # comic-text-detector puo' aver spezzato il testo di un unico balloon
        # in piu' box separati (es. righe distanti tra loro). Se un altro
        # testo e' gia' stato associato allo stesso balloon reale, non gli si
        # sovrascrivono bbox/mask con quelli gia' presi: verrebbero puliti e
        # renderizzati due volte nello stesso punto, sovrapposti. Si tratta
        # invece questo testo come un balloon a se' stante, mantenendo la
        # propria maschera/bbox di comic-text-detector (solo testo): ogni
        # detection viene ripulita e OCRizzata nella propria area, non in
        # quella (piu' ampia) del balloon reale condiviso. bubble_group_id
        # e _orig_y1 permettono a ocr.py di riconoscere questi gruppi dopo
        # l'OCR e togliere dal testo del balloon "pieno" la parte iniziale
        # gia' catturata dal balloon piu' piccolo (altrimenti la ripete).
        existing = claimed_bubble_ids.get(best["bubble_id"]) if best is not None else None

        if best is not None and existing is None:
            det["_orig_y1"] = y1
            det["bubble_group_id"] = best["bubble_id"]
            det["bbox"] = best["bbox"]
            det["mask_path"] = best["mask_path"]
            det["balloon_source"] = "yolov8seg"
            if best.get("split"):
                # bubble nato da _split_fused_bubbles: il bbox e' gia' quello
                # corretto (con margine dall'altra meta'), va rispettato cosi'
                # com'e' in render.py invece di lasciare che il recupero
                # dell'estensione piena del balloon
                # (_expand_mask_to_drawn_balloon) lo faccia risalire di nuovo
                # all'area fusa — vedi il commento su "split" in
                # _split_fused_bubbles.
                det["manual_text_box"] = True
                if best.get("parent_mask_path"):
                    det["parent_mask_path"] = best["parent_mask_path"]
                # Marcatore letto da render.py per applicare un inset piu'
                # ampio del 4px standard dei box manuali: il bbox di un
                # balloon splittato e' gia' il bounding box stretto della
                # sua meta' di maschera (a ridosso del bordo reale del
                # balloon, non un rettangolo con margine come un box
                # ridimensionato a mano in Revisione), quindi 4px lascia il
                # testo troppo vicino al bordo curvo.
                det["balloon_split"] = True
            claimed_bubble_ids[best["bubble_id"]] = det
            matched += 1
        else:
            if existing is not None:
                log.info(
                    f"bubble_seg: balloon reale gia' assegnato a {existing['bbox']}, "
                    f"testo a bbox {det['bbox']} trattato come balloon separato con la "
                    f"propria maschera di comic-text-detector"
                )
                det["_orig_y1"] = y1
                det["bubble_group_id"] = best["bubble_id"]
            det["balloon_source"] = "text_detector"

        deduped.append(det)

    # CTD puo' mancare un dialogo pur avendo bubble_seg individuato
    # correttamente la sua forma. Senza una detection, ocr.run non riceve
    # mai quel ritaglio; aggiungiamo quindi i balloon non ancora associati.
    # Gli eventuali balloon davvero vuoti vengono poi scartati dall'OCR.
    missing_bubbles = [b for b in bubbles if b["bubble_id"] not in claimed_bubble_ids]
    for b in missing_bubbles:
        extra = {
            "balloon_id": b["bubble_id"],
            "bbox": b["bbox"],
            "confidence": b.get("confidence", 1.0),
            "mask_path": b["mask_path"],
            "balloon_source": "yolov8seg",
        }
        if b.get("split"):
            extra["manual_text_box"] = True
            extra["balloon_split"] = True
            if b.get("parent_mask_path"):
                extra["parent_mask_path"] = b["parent_mask_path"]
        deduped.append(extra)

    data["detections"] = deduped

    if matched:
        log.info(f"bubble_seg: {matched}/{len(data['detections'])} testi associati a balloon di forma reale")
    if missing_bubbles:
        log.info(f"bubble_seg: aggiunti {len(missing_bubbles)} balloon senza testo CTD per OCR di recupero")

    with open(detections_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _sort_reading_order(detections_json_path: Path) -> None:
    """
    Riordina i balloon secondo l'ordine di lettura naturale italiano
    (alto->basso, sinistra->destra), al posto dell'ordine di detection
    (arbitrario: dipende dalla confidenza/NMS del modello, non ha
    relazione con la lettura). Serve sia per l'ordine con cui i testi
    arrivano al traduttore (coerenza di contesto tra balloon vicini) sia
    per la numerazione balloon_id mostrata in Revisione.

    Approssimazione a "bande orizzontali": due balloon il cui centro
    verticale cade nella stessa fascia (spessa quanto l'altezza mediana
    dei balloon della pagina) sono considerati sulla stessa riga di
    lettura e ordinati da sinistra a destra; fasce diverse sono lette
    dall'alto in basso. Non e' consapevole della struttura a pannelli
    (non rileva i riquadri): su layout con pannelli affiancati
    orizzontalmente puo' sbagliare l'ordine tra un pannello e l'altro.
    Resta comunque un miglioramento netto rispetto all'ordine di
    detection, che non ha alcuna relazione con la lettura.
    """
    with open(detections_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = data["detections"]
    if len(detections) <= 1:
        return

    heights = [d["bbox"][3] - d["bbox"][1] for d in detections]
    row_tol = max(20, int(np.median(heights) * 0.6))

    def sort_key(det):
        x1, y1, x2, y2 = det["bbox"]
        cy = (y1 + y2) / 2
        cx = (x1 + x2) / 2
        return (round(cy / row_tol), cx)

    detections.sort(key=sort_key)
    for i, det in enumerate(detections):
        det["balloon_id"] = i

    with open(detections_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run(page_path: Path, cfg: dict, work_dir: Path) -> Path:
    """
    Esegue detection+segmentazione testo tramite comic-text-detector, poi
    (se disponibile) rifinisce le maschere dei balloon con bubble_seg
    (kitsumed/yolov8m_seg-speech-bubble), che segmenta la forma reale del
    balloon invece del solo testo, e infine riordina i balloon per ordine
    di lettura.
    """
    page_work_dir = work_dir / page_path.stem
    detections_json_path = _run_ctd(page_path, cfg, page_work_dir)

    bubbles_json_path = _run_bubbleseg(page_path, cfg, page_work_dir)
    if bubbles_json_path is not None:
        try:
            _dedupe_overlapping_bubbles(bubbles_json_path)
        except Exception as e:
            log.warning(f"Deduplica balloon sovrapposti fallita per {page_path.name} ({e}), proseguo con le maschere originali.")
        try:
            _subtract_overlapping_bubbles(bubbles_json_path)
        except Exception as e:
            log.warning(f"Risoluzione sovrapposizioni parziali fallita per {page_path.name} ({e}), proseguo con le maschere originali.")
        try:
            _split_fused_bubbles(bubbles_json_path, detections_json_path)
        except Exception as e:
            log.warning(f"Split balloon fusi fallito per {page_path.name} ({e}), proseguo con le maschere originali.")
        try:
            # Seconda passata di deduplica, obbligatoria dopo lo split.
            # bubble_seg puo' rilevare lo stesso balloon due volte: una da
            # solo e una dentro una maschera fusa col vicino. La deduplica
            # iniziale non le riconosce come duplicate - la maschera fusa e'
            # molto piu' grande, quindi non e' "quasi interamente contenuta"
            # nell'altra - ma appena lo split la divide, una delle meta'
            # coincide con la maschera singola gia' presente. Senza questo
            # secondo controllo quel balloon arriva in fondo due volte: due
            # letture OCR, due traduzioni e il testo stampato due volte, uno
            # sopra l'altro (2 . An Unconventional Couple, pagine 009 e 012).
            _dedupe_overlapping_bubbles(bubbles_json_path)
        except Exception as e:
            log.warning(f"Deduplica dopo lo split fallita per {page_path.name} ({e}), proseguo.")
        try:
            _merge_with_bubble_shapes(detections_json_path, bubbles_json_path)
        except Exception as e:
            log.warning(f"Merge con bubble_seg fallito per {page_path.name} ({e}), uso solo comic-text-detector.")

    _sort_reading_order(detections_json_path)

    return detections_json_path
