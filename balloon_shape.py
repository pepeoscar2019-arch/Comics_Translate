"""Helper puri per analizzare il profilo di larghezza di una maschera di
balloon (nessuna dipendenza da PIL/Draw): individuano una "strozzatura"
(waist) che indica due lobi distinti, es. un balloon a clessidra o due
balloon vicini fusi in un'unica maschera da bubble_seg. Condiviso da
render.py (fitting del testo su balloon a doppio lobo) e detect.py
(_split_fused_bubbles: separa due balloon fisici fusi in una maschera)."""

from pathlib import Path

import cv2
import numpy as np


def smooth_widths(widths: list[int], window: int = 7) -> list[int]:
    """
    Media mobile sul profilo di larghezza. La maschera arriva da una
    segmentazione neurale (bubble_seg), il cui bordo non e' mai perfettamente
    liscio: senza smoothing, piccole irregolarita' di pochi pixel su una
    coda affusolata (singolo lobo) vengono scambiate da find_waist per una
    vera strozzatura a clessidra.
    """
    n = len(widths)
    if n == 0:
        return widths
    half = window // 2
    arr = np.array(widths, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out.round().astype(int).tolist()


def trim_tapered_ends(
    widths: list[int], ratio: float = 0.5, descent_ratio: float = 0.6,
) -> tuple[int, int]:
    """
    Indici (primo, ultimo) del profilo di larghezza da considerare "corpo"
    del balloon, escluse le estremita' affusolate: la CODA (il puntatore
    triangolare verso chi parla) e le punte di un ovale.

    Serve al fitting del testo: la maschera di bubble_seg comprende anche la
    coda, che su un balloon piccolo puo' valere meta' dell'altezza totale.
    Considerandola area utile, il testo viene distribuito su un'altezza che
    in gran parte e' larga 20-40px, le righe basse non riescono a contenere
    nemmeno una parola e l'intero fitting a maschera fallisce, ricadendo sul
    rettangolo (che a sua volta misura il balloon dal centro del bbox — che
    con la coda dentro cade nella coda stessa). Risultato osservato: testo
    incolonnato e troncato tipo "GUA / CHI / E' / ARR" (pagina 010 di
    Voluptuous Housewives 10, "GUARDA CHI E' ARRIVATO!").

    Regola: per ciascuna estremita' si cerca il massimo locale piu' esterno
    (si avanza dal bordo verso l'interno finche' la larghezza non cala), poi
    si taglia tutto cio' che sta oltre il punto in cui la larghezza scende
    sotto `ratio` di quel massimo.

    Distingue cosi' una coda da un secondo lobo, che e' il motivo per cui
    non basta una soglia fissa sulla larghezza massima assoluta:
    - coda: dalla punta verso l'interno la larghezza cresce senza sosta fino
      al corpo, quindi il massimo locale piu' esterno E' il corpo e la coda
      (molto piu' stretta) viene tagliata per intero;
    - secondo lobo (clessidra, balloon a doppia bolla): la larghezza cresce
      fino al centro del lobo e poi cala verso la strozzatura, quindi il
      massimo locale piu' esterno e' il lobo stesso e il taglio avviene
      rispetto alla SUA larghezza — il lobo resta utilizzabile anche se e'
      molto piu' stretto del lobo principale;
    - punta di un ovale: massimo locale = l'equatore, si taglia solo
      l'estremita' dove la forma e' meno di meta' della larghezza massima.
    """
    n = len(widths)
    if n < 3:
        return 0, max(0, n - 1)

    def cut_from_start(seq: list[int]) -> int:
        peak_i, peak = 0, seq[0]
        for i in range(1, len(seq)):
            if seq[i] >= peak:
                peak, peak_i = seq[i], i
            elif seq[i] < peak * descent_ratio:
                break  # vera discesa: peak e' il massimo locale piu' esterno
            # Un calo lieve non interrompe la salita: il bordo di una coda
            # non e' liscio e oscilla di qualche pixel (osservato 8->5->8 su
            # una coda larga 8px). Fermandosi al primo calo, il "massimo
            # locale" diventava quel dosso da 8px e non si tagliava nulla.
        limit = peak * ratio
        for i in range(peak_i + 1):
            if seq[i] >= limit:
                return i
        return peak_i

    first = cut_from_start(widths)
    last = n - 1 - cut_from_start(widths[::-1])
    if first >= last:
        return 0, n - 1
    return first, last


def find_waist(
    widths: list[int], y0: int, min_run_ratio: float = 0.08, dip_ratio: float = 0.7,
    min_lobe_ratio: float = 0.3,
) -> int | None:
    """
    Cerca una strozzatura pronunciata nel profilo di larghezza (indicizzato
    da y0), che indica una forma a due lobi (es. balloon "a doppia bolla",
    o due balloon vicini fusi in una maschera). Ignora il 15% agli estremi
    (spesso la punta di un lobo, larghezza quasi 0 ma non una vera
    strozzatura centrale). Ritorna la riga y della strozzatura solo se e'
    sotto `dip_ratio` (default 0.7 = 70%) del massimo su entrambi i lati, ED
    entrambi i lati sostengono quella larghezza per un tratto continuo
    (min_run_ratio della lunghezza totale) e non solo per un singolo picco
    isolato: una coda affusolata a lobo singolo puo' avere un massimo "dopo"
    il minimo che e' pero' solo un rigonfiamento di pochi pixel, non un vero
    secondo lobo. Altrimenti None (forma a lobo singolo, nessuno split).

    Un terzo controllo, indipendente dal precedente: il picco di ciascun
    lato deve raggiungere almeno `min_lobe_ratio` (default 30%) della
    larghezza massima assoluta del profilo. Una coda a punta che si allarga
    gradualmente verso il corpo del balloon (es. il puntatore triangolare
    verso il parlante) puo' restare sotto la soglia di `dip_ratio` per un
    tratto lungo quanto basta a superare `sustained_above` pur non essendo
    mai un vero secondo lobo — resta sempre molto piu' stretta del corpo
    reale del balloon. Un vero doppio lobo (clessidra, o due balloon fusi)
    ha invece entrambi i lati vicini alla larghezza massima. Bug osservato
    su una coda scambiata per una fusione (pagina 006, "HMM! YOU HAVE A
    COCK..."): la coda restava soggetta al 15%-agli-estremi ma si estendeva
    ben oltre, con larghezza fino a ~56px contro un corpo di ~350px (16%,
    sotto qualunque dip_ratio ragionevole ma senza questo controllo passava
    comunque perche' il tratto sostenuto esisteva).

    `dip_ratio` e' regolabile dal chiamante: il default 0.7 e' quello
    validato per il fitting del testo su balloon a doppio lobo in render.py
    (strozzature nette, tipo clessidra). Un chiamante con esigenze diverse
    (es. rilevare due balloon distinti fusi in una maschera, dove il calo
    di larghezza al punto di contatto puo' essere piu' lieve) puo' passare
    un valore piu' permissivo senza cambiare il comportamento di render.py.
    """
    n = len(widths)
    if n < 20:
        return None
    lo = int(n * 0.15)
    hi = n - int(n * 0.15)
    if hi <= lo:
        return None
    min_i = min(range(lo, hi), key=lambda i: widths[i])
    min_w = widths[min_i]
    if min_w <= 0:
        return None

    global_max = max(widths, default=0)
    if global_max <= 0:
        return None
    left_peak = max(widths[:min_i], default=0)
    right_peak = max(widths[min_i + 1:], default=0)
    if left_peak < min_lobe_ratio * global_max or right_peak < min_lobe_ratio * global_max:
        return None

    min_run = max(3, int(n * min_run_ratio))

    def sustained_above(values: list[int], threshold: float) -> bool:
        run = 0
        for w in values:
            if w >= threshold:
                run += 1
                if run >= min_run:
                    return True
            else:
                run = 0
        return False

    threshold = min_w / dip_ratio
    if not sustained_above(widths[:min_i], threshold):
        return None
    if not sustained_above(widths[min_i + 1:], threshold):
        return None
    return y0 + min_i


TEXT_INSIDE_RATIO = 0.8
"""Frazione minima della maschera di testo di comic-text-detector che deve
cadere dentro la sagoma del balloon perche' quel testo sia considerato suo.
Il testo di un balloon vicino cade a zero (maschere disgiunte), mentre il
testo proprio arriva a 0.95-1.0 anche quando la segmentazione perde un
pezzo di balloon: la soglia sta larga in mezzo."""

TEXT_DILATE_PX = 3
"""Margine attorno ai tratti recuperati dalla maschera di testo: copre
l'anti-aliasing delle lettere, che altrimenti resta come alone grigio."""


def load_text_masks(
    mask_paths: list[str], shape: tuple[int, int],
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """
    Carica le maschere di testo di comic-text-detector della pagina
    (work/<pagina>/masks/*.png), come coppie (bbox, sotto-maschera) per non
    tenere in memoria una maschera a piena pagina per ciascun testo.

    Servono a text_ink_outside_balloons: quando una detection viene promossa
    alla forma reale del balloon (detect.py::_merge_with_bubble_shapes) la
    sua maschera di testo viene sostituita da quella di bubble_seg, quindi
    l'unica copia dei tratti del testo resta quella su disco.
    """
    out: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    seen_dirs: set[Path] = set()
    for mp in mask_paths:
        mask_dir = Path(mp).parent.parent / "masks"
        if mask_dir in seen_dirs or not mask_dir.is_dir():
            continue
        seen_dirs.add(mask_dir)
        for f in sorted(mask_dir.iterdir()):
            if f.suffix.lower() != ".png":
                continue
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != shape:
                continue
            ys, xs = np.where(m > 0)
            if len(ys) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            out.append(((x1, y1, x2, y2), m[y1:y2, x1:x2] > 0))
    return out


def text_ink_outside_balloons(
    text_masks: list[tuple[tuple[int, int, int, int], np.ndarray]],
    raw_masks: list[np.ndarray | None],
) -> list[np.ndarray | None]:
    """
    Per ogni balloon, i tratti del suo testo che cadono FUORI dalla maschera
    di bubble_seg, da aggiungere all'area da riempire.

    bubble_seg sotto-segmenta il balloon dove un altro balloon disegnato
    sopra ne taglia la sagoma: il contorno del vicino separa un'insenatura
    di interno-balloon dal resto (non e' un buco chiuso, quindi non si
    recupera ne' riempiendo i buchi ne' con un flood fill di colore, che il
    contorno nero blocca). Le lettere che finiscono in quell'insenatura
    restano sull'originale e si vedono sotto la traduzione renderizzata.

    La maschera di testo di comic-text-detector invece copre tutti i tratti,
    insenatura compresa. Ogni testo va pero' assegnato a UN SOLO balloon,
    quello che lo contiene di piu': due balloon che si sovrappongono possono
    contenere entrambi buona parte dello stesso testo, e attribuirlo anche a
    quello sbagliato significa dipingerlo con il colore di fondo di
    quest'ultimo, fuori dalla sua area gia' pulita (macchia visibile).

    Si aggiungono solo i tratti del testo, mai l'area attorno: nessun
    rischio di dipingere artwork fuori dal balloon.
    """
    extras: list[np.ndarray | None] = [None] * len(raw_masks)
    if not text_masks:
        return extras

    raw_bools = [None if m is None else m > 0 for m in raw_masks]
    for (x1, y1, x2, y2), sub in text_masks:
        total = int(sub.sum())
        if total == 0:
            continue
        best_idx, best_inside = None, 0
        for i, raw_bool in enumerate(raw_bools):
            if raw_bool is None:
                continue
            inside = int(np.count_nonzero(sub & raw_bool[y1:y2, x1:x2]))
            if inside > best_inside:
                best_idx, best_inside = i, inside
        if best_idx is None or best_inside / total < TEXT_INSIDE_RATIO:
            continue
        if best_inside == total:
            continue  # gia' interamente coperto dalla maschera del balloon
        if extras[best_idx] is None:
            extras[best_idx] = np.zeros(raw_masks[best_idx].shape, dtype=np.uint8)
        extras[best_idx][y1:y2, x1:x2][sub] = 255

    kernel = np.ones((TEXT_DILATE_PX * 2 + 1, TEXT_DILATE_PX * 2 + 1), np.uint8)
    return [None if e is None else cv2.dilate(e, kernel) for e in extras]
