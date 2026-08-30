import translate_local
import translate_skip
import argparse
import json
import logging
import time
import subprocess
import requests
from pathlib import Path

import detect
import ocr
import clean
import render
import paths
import audit
import gpu_resources

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

_llama_process = None


def start_llama_server(cfg: dict, mode: str):
    """Avvia llama-server con il modello OCR o traduzione. mode = 'ocr' | 'translate'"""
    global _llama_process
    stop_llama_server()

    ls_cfg = cfg.get("llama_server", {})
    port = ls_cfg.get("port", 8081)

    if mode == "ocr":
        if cfg.get("ocr_backend", "qwen") == "paddleocr_vl":
            model_path = ls_cfg.get("paddleocr_model_path")
            mmproj_path = ls_cfg.get("paddleocr_mmproj_path")
        else:
            model_path = ls_cfg.get("ocr_model_path")
            mmproj_path = ls_cfg.get("ocr_mmproj_path")
    else:
        model_path = ls_cfg.get("translate_model_path")
        mmproj_path = None
    extra_args = [str(a) for a in (ls_cfg.get(f"{mode}_extra_args") or ls_cfg.get("extra_args") or [])]

    if not model_path:
        raise RuntimeError(
            f"Manca il path del modello per lo stage '{mode}' in config.yaml "
            f"(llama_server.{'translate_model_path' if mode == 'translate' else 'ocr_model_path'})."
        )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modello GGUF non trovato: {model_path}")
    if mmproj_path and not Path(mmproj_path).exists():
        raise FileNotFoundError(f"Proiettore multimodale non trovato: {mmproj_path}")

    cmd = [
        "llama-server",
        "-m", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--temp", "0",
        "-c", str(ls_cfg.get("ctx_size", 8192)),
    ]
    # -ngl solo se richiesto esplicitamente: passarlo disattiva l'auto-fit di
    # llama.cpp, che da solo sceglie quanti layer stanno in VRAM. Con -ngl 99
    # un 27B Q4 non ci sta e il server muore in allocazione (access violation).
    n_gpu_layers = ls_cfg.get("n_gpu_layers")
    if n_gpu_layers not in (None, "", "auto"):
        cmd += ["-ngl", str(n_gpu_layers)]
    if mmproj_path:
        cmd += ["--mmproj", mmproj_path]
    cmd += extra_args

    log.info(f"Avvio llama-server ({mode}): {Path(model_path).name}")
    try:
        _llama_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError(
            "Comando 'llama-server' non trovato nel PATH. Installa llama.cpp "
            "(winget install ggml.llamacpp) oppure aggiungi la cartella dei binari al PATH."
        )

    timeout = ls_cfg.get("startup_timeout", 180)
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Un modello grande puo' metterci un minuto a caricarsi: se il
        # processo muore prima (VRAM insufficiente, GGUF corrotto) inutile
        # aspettare la scadenza.
        if _llama_process.poll() is not None:
            raise RuntimeError(
                f"llama-server e' uscito subito (codice {_llama_process.returncode}) "
                f"caricando {Path(model_path).name}: VRAM insufficiente o modello non valido?"
            )
        try:
            if requests.get(url, timeout=2).status_code == 200:
                log.info(f"llama-server pronto su porta {port}")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    stop_llama_server()
    raise RuntimeError(f"llama-server non risponde dopo {timeout}s")


def _point_at_llama_server(cfg: dict) -> None:
    """Allinea local_llm.base_url alla porta di llama-server: e' l'endpoint
    che ocr.py e translate_local.py chiamano, e deve seguire il config anche
    se qualcuno cambia la porta."""
    port = cfg.get("llama_server", {}).get("port", 8081)
    cfg.setdefault("local_llm", {})["base_url"] = f"http://127.0.0.1:{port}/v1"


def stop_llama_server():
    global _llama_process
    if _llama_process and _llama_process.poll() is None:
        log.info("Stop llama-server...")
        _llama_process.terminate()
        try:
            _llama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _llama_process.kill()
        _llama_process = None




def load_config(path: str = None) -> dict:
    return paths.load_cfg(path)


def collect_pages(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not input_dir.exists():
        return []
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)


def find_comics(base_input_dir: Path) -> list[str]:
    if not base_input_dir.exists():
        log.error(f"Cartella input non trovata: {base_input_dir}")
        return []
    subfolders = sorted(p.name for p in base_input_dir.iterdir() if p.is_dir())
    if subfolders:
        return subfolders
    if collect_pages(base_input_dir):
        return [""]
    return []


def run_ocr_stage(cfg: dict, pages: list[Path], work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    failed_pages = []
    total_balloon = 0
    scartati_onomatopea = 0

    # NUOVO: libera la VRAM di ComfyUI, non serve qui
    gpu_resources.free_comfyui(cfg)

    # Il modello di visione (qwen o paddleocr_vl, cambia solo quale GGUF
    # viene caricato) gira su llama-server, avviato per questo stage e chiuso
    # alla fine: e' il processo che finisce a liberare la VRAM, non serve
    # scaricare i modelli a mano.
    _point_at_llama_server(cfg)
    start_llama_server(cfg, mode="ocr")

    try:
        for page_path in pages:
            log.info(f"=== OCR: {page_path.name} ===")
            try:
                detections_json = detect.run(page_path, cfg, work_dir)
                ocr_json_path = ocr.run(detections_json, page_path, cfg, work_dir)

                with open(ocr_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for det in data["detections"]:
                    total_balloon += 1
                    if not det.get("testo_originale", "").strip():
                        scartati_onomatopea += 1

                # Crea subito translated.json (testo originale copiato in
                # testo_tradotto) cosi' e' gia' pronto da modificare a mano
                # con una traduzione esterna, senza dover lanciare lo stage
                # di traduzione prima di poter editare.
                translate_skip.run(ocr_json_path, cfg)

            except Exception as e:
                log.error(f"Errore durante l'elaborazione di {page_path.name}: {e}")
                log.error("Salto questa pagina e continuo con le successive.")
                failed_pages.append(page_path.name)
                continue
    finally:
        stop_llama_server()

    if failed_pages:
        log.warning(f"Pagine fallite in questo stage OCR ({len(failed_pages)}): {failed_pages}")

    stats = {
        "pagine_totali": len(pages),
        "pagine_fallite": len(failed_pages),
        "balloon_totali": total_balloon,
        "balloon_scartati": scartati_onomatopea,
    }

    stats_path = work_dir / "_ocr_stats.json"
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Impossibile salvare stats OCR: {e}")

    return stats


def run_translate_stage(cfg: dict, pages: list[Path], work_dir: Path, force: bool = False):
    """Traduzione col modello locale servito da llama-server: avviato qui,
    chiuso alla fine dello stage (cosi' la VRAM torna libera per la pulizia)."""
    failed_pages = []

    gpu_resources.free_comfyui(cfg)
    _point_at_llama_server(cfg)
    start_llama_server(cfg, mode="translate")

    try:
        for page_path in pages:
            log.info(f"=== Traduzione: {page_path.name} ===")
            page_work_dir = work_dir / page_path.stem
            ocr_json_path = page_work_dir / "ocr.json"

            if not ocr_json_path.exists():
                log.warning(f"Manca ocr.json per {page_path.name}, salto")
                continue

            try:
                translate_local.run(ocr_json_path, cfg, force=force)
            except Exception as e:
                log.error(f"Errore traducendo {page_path.name}: {e}")
                failed_pages.append(page_path.name)
    finally:
        stop_llama_server()

    if failed_pages:
        log.warning(f"Pagine fallite in questo stage traduzione ({len(failed_pages)}): {failed_pages}")


def run_translate_skip_stage(cfg: dict, pages: list[Path], work_dir: Path):
    """Salta la traduzione AI: copia testo_originale in testo_tradotto,
    utile quando il fumetto e' gia' in italiano ma si vuole comunque
    eseguire tutta la pipeline (es. per rifare pulizia/render)."""
    failed_pages = []

    for page_path in pages:
        log.info(f"=== Traduzione (saltata): {page_path.name} ===")
        page_work_dir = work_dir / page_path.stem
        ocr_json_path = page_work_dir / "ocr.json"

        if not ocr_json_path.exists():
            log.warning(f"File OCR non trovato per {page_path.name}, salto.")
            continue

        try:
            translate_skip.run(ocr_json_path, cfg)
        except Exception as e:
            log.error(f"Errore copiando testo OCR per {page_path.name}: {e}")
            failed_pages.append(page_path.name)

    if failed_pages:
        log.warning(f"Pagine fallite in questo stage traduzione ({len(failed_pages)}): {failed_pages}")


def run_clean_stage(cfg: dict, pages: list[Path], work_dir: Path):
    """Pulizia dei balloon con il riempimento deterministico di clean.py: gira
    su CPU, non serve nessun modello ne' nessun server."""
    failed_pages = []

    for page_path in pages:
        log.info(f"=== Pulizia: {page_path.name} ===")
        page_work_dir = work_dir / page_path.stem
        translated_json_path = page_work_dir / "translated.json"

        if not translated_json_path.exists():
            log.warning(f"Manca translated.json per {page_path.name}, salto.")
            continue

        try:
            cleaned_image_path = clean.run(page_path, translated_json_path, cfg, work_dir)
            log.info(f"Pagina pulita: {cleaned_image_path}")
        except Exception as e:
            log.error(f"Errore pulendo {page_path.name}: {e}")
            failed_pages.append(page_path.name)
            continue

    if failed_pages:
        log.warning(f"Pagine fallite in questo stage pulizia ({len(failed_pages)}): {failed_pages}")


def run_text_render_stage(cfg: dict, pages: list[Path], work_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_pages = []

    for page_path in pages:
        log.info(f"=== Render: {page_path.name} ===")
        page_work_dir = work_dir / page_path.stem
        translated_json_path = page_work_dir / "translated.json"
        cleaned_image_path = page_work_dir / "cleaned.png"

        if not translated_json_path.exists():
            log.warning(f"Manca translated.json per {page_path.name}, salto.")
            continue
        if not cleaned_image_path.exists():
            log.warning(f"Manca cleaned.png per {page_path.name} (esegui prima lo stage 'clean'), salto.")
            continue

        try:
            final_image_path = render.run(cleaned_image_path, translated_json_path, cfg, output_dir)
            log.info(f"Pagina completata: {final_image_path}")
        except Exception as e:
            log.error(f"Errore renderizzando {page_path.name}: {e}")
            failed_pages.append(page_path.name)
            continue

    if failed_pages:
        log.warning(f"Pagine fallite in questo stage render ({len(failed_pages)}): {failed_pages}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline traduzione fumetti")
    parser.add_argument("--config", default=str(paths.CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--comic", default=None)
    parser.add_argument("--stage", choices=["ocr", "translate", "translate_lm", "translate_skip", "clean", "render", "audit"], required=True)
    parser.add_argument("--force-translate", action="store_true",
                         help="Ignora la cache traduzioni e richiama sempre l'AI, anche per testi gia' tradotti in precedenza")
    parser.add_argument("--ocr-backend", choices=["qwen", "paddleocr_vl"], default=None,
                         help="Backend di visione per lo stage ocr, sovrascrive ocr_backend in config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ocr_backend:
        cfg["ocr_backend"] = args.ocr_backend

    base_input_dir = Path(cfg["paths"]["input_dir"])
    base_output_dir = Path(cfg["paths"]["output_dir"])
    base_work_dir = Path(cfg["paths"]["work_dir"])

    comics = find_comics(base_input_dir)

    if args.comic:
        if args.comic not in comics:
            log.error(f"Fumetto '{args.comic}' non trovato tra: {comics}")
            comics = []
        else:
            comics = [args.comic]

    if not comics:
        log.error(f"Nessun fumetto trovato in {base_input_dir}")
    else:
        log.info(f"Trovati {len(comics)} fumetti: {[c or '(root)' for c in comics]}")

        for comic in comics:
            input_dir = base_input_dir / comic if comic else base_input_dir
            output_dir = base_output_dir / comic if comic else base_output_dir
            work_dir = base_work_dir / comic if comic else base_work_dir

            pages = collect_pages(input_dir)
            if args.limit:
                pages = pages[:args.limit]
            start = args.start if args.start is not None else 0
            end = args.end if args.end is not None else len(pages)
            pages = pages[start:end]

            sep = "=" * 60
            log.info("")
            log.info(sep)
            log.info(f"FUMETTO: {comic or '(root)'} - {len(pages)} pagine")
            log.info(sep)

            if not pages and args.stage != "audit":
                log.warning(f"Nessuna pagina nell'intervallo per {comic}, salto.")
                continue

            if args.stage == "ocr":
                run_ocr_stage(cfg, pages, work_dir)
            elif args.stage in ("translate", "translate_lm"):
                # C'e' un solo percorso di traduzione (modello locale su
                # llama-server): "translate_lm" resta accettato come alias
                # storico, cosi' i comandi salvati continuano a funzionare.
                run_translate_stage(cfg, pages, work_dir, force=args.force_translate)
            elif args.stage == "translate_skip":
                run_translate_skip_stage(cfg, pages, work_dir)
            elif args.stage == "clean":
                run_clean_stage(cfg, pages, work_dir)
            elif args.stage == "audit":
                # Sola lettura: non tocca nulla, elenca cosa resta da
                # sistemare prima di considerare finito il fumetto.
                result = audit.audit_comic(work_dir, output_dir, cfg)
                for riga in audit.format_report(result, comic).splitlines():
                    log.info(riga)
            else:
                run_text_render_stage(cfg, pages, work_dir, output_dir)

        log.info("Completato per tutti i fumetti.")
