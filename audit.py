"""
Controllo finale di un fumetto gia' lavorato.

Serve a rispondere in un colpo solo alla domanda "posso considerarlo finito?":
scorre le pagine e raccoglie i problemi che altrimenti si scoprono solo
sfogliando le tavole una per una - un balloon rimasto in inglese, un testo
troncato, una pagina che non e' mai passata dalla pulizia. Non modifica
niente: e' solo una lettura dei file di lavoro.

I controlli, in ordine di gravita':

- `non_tradotto`  la traduzione e' identica al testo originale, quindi la
  tavola mostra ancora la lingua di partenza. E' il difetto piu' insidioso
  perche' il render riesce benissimo: non c'e' nessun errore da nessuna parte,
  semplicemente il testo e' quello sbagliato.
- `da_trascrivere` l'OCR ha lasciato vuoto un balloon che pero' contiene
  testo (flag ocr_empty_suspect, vedi ocr.py).
- `troncato`      render.py ha tagliato il testo per farlo stare (_overflow).
- `troppo_lungo`  la traduzione sta nel balloon solo a un font molto piu'
  piccolo dell'originale (vedi fit_check): si legge male.
- `doppione`     due balloon quasi sovrapposti hanno entrambi del testo: il
  render li stampa uno sopra l'altro e la pagina diventa illeggibile in quel
  punto. Nasce da bubble_seg che rileva lo stesso balloon due volte (vedi
  detect.py::_dedupe_overlapping_bubbles).
- `manca_<file>`  la pagina non ha completato uno stage.
"""

import json
import logging
from pathlib import Path

import fit_check

log = logging.getLogger("pipeline.audit")


def _overlap_ratio(a: list[int], b: list[int]) -> float:
    """Quanto il piu' piccolo dei due box e' contenuto nell'altro (0..1).

    Non si usa la IoU classica: due detection dello stesso balloon possono
    avere dimensioni diverse (una e' meta' di una maschera fusa, l'altra il
    balloon intero) e la IoU le penalizzerebbe proprio nel caso che
    interessa riconoscere.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def _find_doppioni(detections: list[dict]) -> list[tuple]:
    """
    Coppie di balloon che sono in realta' lo stesso balloon contato due volte.

    Non basta "il piu' piccolo sta dentro il piu' grande": render.py
    riscrive i bbox espandendoli al balloon disegnato, e su due balloon a
    contatto quell'espansione puo' far inglobare al primo l'area del vicino.
    Quel caso e' innocuo (il testo viene impaginato sulla maschera, non sul
    bbox) e non va segnalato. Un doppione vero ha invece i due box quasi
    coincidenti, quindi si chiede anche che l'intersezione copra buona parte
    del PIU' GRANDE dei due - condizione che il caso del vicino inglobato non
    soddisfa. Testi uguali restano un doppione a prescindere: li' non c'e'
    dubbio su cosa si vede sulla pagina.
    """
    trovati = []
    con_testo = [
        d for d in detections
        if d.get("bbox") and (d.get("testo_tradotto") or "").strip()
    ]
    for i in range(len(con_testo)):
        for j in range(i + 1, len(con_testo)):
            a, b = con_testo[i], con_testo[j]
            ratio = _overlap_ratio(a["bbox"], b["bbox"])
            if ratio < 0.8:
                continue
            uguali = ((a.get("testo_tradotto") or "").strip().lower()
                      == (b.get("testo_tradotto") or "").strip().lower())
            if uguali or _overlap_ratio_max(a["bbox"], b["bbox"]) >= 0.65:
                trovati.append((a, b, ratio))
    return trovati


def _overlap_ratio_max(a: list[int], b: list[int]) -> float:
    """Come _overlap_ratio ma rapportata al box PIU' GRANDE: dice quanto i due
    box sono lo stesso box, invece di quanto il piccolo sta nel grande."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(area_a, area_b)


def _page_dirs(comic_work_dir: Path) -> list[Path]:
    if not comic_work_dir.exists():
        return []
    return sorted(d for d in comic_work_dir.iterdir() if d.is_dir())


def audit_comic(comic_work_dir: Path, comic_output_dir: Path, cfg: dict) -> dict:
    """
    Ritorna {"pages": [...], "totals": {...}}: una riga per pagina con i
    problemi trovati, piu' i totali per tipo.
    """
    pages = []
    totals = {
        "non_tradotto": 0, "da_trascrivere": 0, "troncato": 0,
        "troppo_lungo": 0, "doppione": 0, "pagine_incomplete": 0, "balloon": 0,
    }

    for page_dir in _page_dirs(comic_work_dir):
        page_id = page_dir.name
        issues = []

        translated_path = page_dir / "translated.json"
        if not translated_path.exists():
            issues.append({"tipo": "manca_translated", "dettaglio": "pagina mai tradotta"})
            totals["pagine_incomplete"] += 1
            pages.append({"page": page_id, "issues": issues})
            continue

        if not (page_dir / "cleaned.png").exists():
            issues.append({"tipo": "manca_cleaned", "dettaglio": "pagina mai pulita"})
            totals["pagine_incomplete"] += 1

        if comic_output_dir.exists() and not any(comic_output_dir.glob(f"{page_id}.*")):
            issues.append({"tipo": "manca_render", "dettaglio": "pagina mai renderizzata"})
            totals["pagine_incomplete"] += 1

        try:
            with open(translated_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            issues.append({"tipo": "illeggibile", "dettaglio": str(e)})
            pages.append({"page": page_id, "issues": issues})
            continue

        for det in data.get("detections", []):
            bid = det.get("balloon_id")
            originale = (det.get("testo_originale") or "").strip()
            tradotto = (det.get("testo_tradotto") or "").strip()
            totals["balloon"] += 1

            if det.get("ocr_empty_suspect") and not tradotto:
                issues.append({"tipo": "da_trascrivere", "balloon": bid,
                                "dettaglio": "OCR vuoto ma il balloon contiene testo"})
                totals["da_trascrivere"] += 1

            if originale and tradotto and originale == tradotto:
                # Una parola sola identica nelle due lingue (NO, OK, STOP, i
                # nomi propri) non e' un mancato passaggio dal traduttore:
                # segnalarla riempirebbe il rapporto di falsi allarmi.
                if len(tradotto.split()) > 1:
                    issues.append({"tipo": "non_tradotto", "balloon": bid,
                                    "dettaglio": tradotto[:60]})
                    totals["non_tradotto"] += 1

            if det.get("_overflow") and tradotto:
                issues.append({"tipo": "troncato", "balloon": bid,
                                "dettaglio": tradotto[:60]})
                totals["troncato"] += 1
                continue  # gia' segnalato come problema di spazio, non serve il doppione

            if originale and tradotto:
                try:
                    over, target = fit_check.too_long(det, cfg)
                except Exception:
                    over, target = False, 0
                if over:
                    issues.append({"tipo": "troppo_lungo", "balloon": bid,
                                    "dettaglio": f"{len(tradotto)} caratteri, ne entrano ~{target}"})
                    totals["troppo_lungo"] += 1

        # Un balloon stampato due volte si vede subito ma nessun altro
        # controllo lo intercetta: i due testi sono entrambi "tradotti" e
        # ognuno, preso da solo, ci sta nel suo box.
        for a, b, ratio in _find_doppioni(data.get("detections", [])):
            uguali = ((a.get("testo_tradotto") or "").strip().lower()
                      == (b.get("testo_tradotto") or "").strip().lower())
            issues.append({
                "tipo": "doppione",
                "balloon": a.get("balloon_id"),
                "dettaglio": (
                    f"sovrapposto al {ratio:.0%} al balloon {b.get('balloon_id')}"
                    + (" con lo STESSO testo" if uguali else "")
                    + f": {(a.get('testo_tradotto') or '')[:40]}"
                ),
            })
            totals["doppione"] += 1

        pages.append({"page": page_id, "issues": issues})

    return {"pages": pages, "totals": totals}


def format_report(result: dict, comic: str) -> str:
    """Rapporto leggibile a schermo (usato dalla CLI; la web app usa il dict)."""
    t = result["totals"]
    righe = [f"=== Controllo: {comic or '(root)'} ==="]
    problemi = sum(v for k, v in t.items() if k != "balloon")
    if not problemi:
        righe.append(f"Nessun problema su {len(result['pages'])} pagine / {t['balloon']} balloon.")
        return "\n".join(righe)

    for page in result["pages"]:
        if not page["issues"]:
            continue
        righe.append(f"  {page['page']}:")
        for issue in page["issues"]:
            bal = f" balloon {issue['balloon']}" if issue.get("balloon") is not None else ""
            righe.append(f"    [{issue['tipo']}]{bal} {issue['dettaglio']}")

    righe.append("")
    righe.append(
        f"Totali: {t['non_tradotto']} non tradotti, {t['da_trascrivere']} da trascrivere, "
        f"{t['troncato']} troncati, {t['troppo_lungo']} troppo lunghi, "
        f"{t['doppione']} doppioni, {t['pagine_incomplete']} stage mancanti "
        f"(su {t['balloon']} balloon)"
    )
    return "\n".join(righe)
