"""
Pretrattamento di pagine grezze via ComfyUI (workflow Flux-Klein.json),
PRIMA di OCR/detection: passa l'intera pagina attraverso il workflow con un
prompt scelto al momento del lancio (es. upscale, pulizia rumore, altro),
senza toccare balloon/maschere (non esistono ancora a questo punto della
pipeline). Il prompt e' passato solo per la singola esecuzione, NON scritto
in Flux-Klein.json: non tocca il prompt di pulizia balloon (nodo 6),
modificabile a parte dalle GUI.

Isolato deliberatamente dal resto della pipeline (nessun import da
clean.py/render.py/main.py, stesso principio del backend lama - vedi
memoria progetto): duplica la sola logica di chiamata a ComfyUI che gli
serve. Scrive in una cartella nuova (pretreated_pages/, accanto a
input_pages/output_pages/work), non sovrascrive mai gli originali.
Va lanciato a mano, main.py non lo conosce.
"""
import argparse
import copy
import json
import logging
import time
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, JpegImagePlugin

import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline.flux_pretreat")

_PROMPT_NODE_ID = "6"  # stesso nodo CLIPTextEncode usato da clean.py per la pulizia balloon

_workflow_cache: dict[str, dict] = {}


def _load_workflow(comfy_cfg: dict) -> dict:
    workflow_path = paths.resolve(comfy_cfg.get("workflow_path", "Flux-Klein.json"))
    key = str(workflow_path)
    if key not in _workflow_cache:
        with open(workflow_path, "r", encoding="utf-8") as f:
            _workflow_cache[key] = json.load(f)
        log.info(f"Workflow ComfyUI caricato da {workflow_path}")
    return _workflow_cache[key]


def _comfyui_is_up(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/system_stats", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _upload_image_to_comfyui(image_bgr: np.ndarray, base_url: str) -> str:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    filename = f"fumetti_pretreat_{uuid.uuid4().hex}.png"
    resp = requests.post(
        f"{base_url}/upload/image",
        files={"image": (filename, buf, "image/png")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["name"]


def _queue_prompt(workflow: dict, base_url: str, client_id: str) -> str:
    resp = requests.post(f"{base_url}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"ComfyUI ha rifiutato il workflow: {resp.text}")
    return resp.json()["prompt_id"]


def _wait_for_result(prompt_id: str, base_url: str, timeout: float, poll_interval: float = 1.5) -> dict:
    elapsed = 0.0
    while elapsed < timeout:
        resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI ha segnalato un errore per il job {prompt_id}: {status}")
            return entry
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"ComfyUI non ha risposto entro {timeout}s per il job {prompt_id}")


def _download_output(image_info: dict, base_url: str) -> np.ndarray:
    params = {
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output"),
    }
    resp = requests.get(f"{base_url}/view", params=params, timeout=60)
    resp.raise_for_status()
    pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


_SCALE_CLASSES = {"ImageScaleToTotalPixels", "ImageScale", "ImageScaleBy",
                  "ImageScaleToMaxDimension", "ImageScaleToMegapixels"}

# Il latente Flux lavora su blocchi da 16px (VAE /8 + patchify /2): a piena
# risoluzione l'immagine va portata a un multiplo di 16, altrimenti ComfyUI
# ritaglia per conto suo e il risultato torna disallineato.
_LATENT_BLOCK = 16


def _apply_resolution(workflow: dict, megapixels: float | None) -> None:
    """Regola il ridimensionamento interno del workflow.

    None  -> lascia il workflow com'e' (comportamento storico: 1.5 MP).
    >0    -> imposta i megapixel sui nodi ImageScaleToTotalPixels.
    0     -> bypassa i nodi di scala: ogni consumatore viene ricollegato
             direttamente alla sorgente del nodo (l'immagine caricata), cosi'
             il modello vede la pagina alla risoluzione originale.
    """
    if megapixels is None:
        return
    scale_ids = [nid for nid, node in workflow.items()
                 if node.get("class_type") in _SCALE_CLASSES]
    if not scale_ids:
        log.warning("Nessun nodo di ridimensionamento trovato nel workflow: risoluzione invariata.")
        return

    if megapixels > 0:
        for nid in scale_ids:
            if "megapixels" in workflow[nid].get("inputs", {}):
                workflow[nid]["inputs"]["megapixels"] = megapixels
                log.info(f"Nodo di scala {nid}: megapixels -> {megapixels}")
        return

    for nid in scale_ids:
        source = workflow[nid].get("inputs", {}).get("image")
        if not isinstance(source, list):
            continue
        for node in workflow.values():
            for key, val in node.get("inputs", {}).items():
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == nid:
                    node["inputs"][key] = source
        workflow.pop(nid, None)
        log.info(f"Nodo di scala {nid} bypassato: piena risoluzione.")


def _call_comfyui(image_bgr: np.ndarray, cfg: dict, prompt_text: str,
                  megapixels: float | None = None) -> np.ndarray:
    comfy_cfg = cfg.get("comfyui", {})
    base_url = comfy_cfg.get("base_url", "http://127.0.0.1:8188").rstrip("/")
    load_node_id = str(comfy_cfg.get("load_node_id", "218"))
    save_node_id = str(comfy_cfg.get("save_node_id", "219"))
    timeout = comfy_cfg.get("timeout", 300)

    if not _comfyui_is_up(base_url):
        raise RuntimeError(f"ComfyUI non raggiungibile su {base_url}. Avvialo prima.")

    workflow_template = _load_workflow(comfy_cfg)
    if load_node_id not in workflow_template:
        raise KeyError(f"Nodo LoadImage '{load_node_id}' non trovato nel workflow ComfyUI")
    if save_node_id not in workflow_template:
        raise KeyError(f"Nodo SaveImage '{save_node_id}' non trovato nel workflow ComfyUI")
    if _PROMPT_NODE_ID not in workflow_template:
        raise KeyError(f"Nodo prompt '{_PROMPT_NODE_ID}' non trovato nel workflow ComfyUI")

    workflow = copy.deepcopy(workflow_template)  # non alterare il template in cache ne' il prompt salvato
    workflow[_PROMPT_NODE_ID]["inputs"]["text"] = prompt_text
    _apply_resolution(workflow, megapixels)

    h, w = image_bgr.shape[:2]
    to_upload = image_bgr
    if megapixels == 0:
        pad_h = (-h) % _LATENT_BLOCK
        pad_w = (-w) % _LATENT_BLOCK
        if pad_h or pad_w:
            to_upload = cv2.copyMakeBorder(image_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        log.info(f"Piena risoluzione: {w}x{h} -> {to_upload.shape[1]}x{to_upload.shape[0]} "
                 f"({to_upload.shape[0] * to_upload.shape[1] / 1e6:.1f} MP)")

    uploaded_name = _upload_image_to_comfyui(to_upload, base_url)
    workflow[load_node_id]["inputs"]["image"] = uploaded_name

    client_id = str(uuid.uuid4())
    prompt_id = _queue_prompt(workflow, base_url, client_id)
    result = _wait_for_result(prompt_id, base_url, timeout)

    outputs = result.get("outputs", {}).get(save_node_id)
    if not outputs or not outputs.get("images"):
        raise RuntimeError(f"ComfyUI non ha prodotto un'immagine per il nodo {save_node_id} (job {prompt_id})")

    treated = _download_output(outputs["images"][0], base_url)

    # Come clean.py: se il workflow ha ridimensionato l'immagine internamente
    # (ImageScaleToTotalPixels) va riportata alla risoluzione di partenza. A
    # piena risoluzione questo passaggio non fa nulla, resta solo il ritaglio
    # del padding di allineamento.
    if treated.shape[:2] != to_upload.shape[:2]:
        treated = cv2.resize(
            treated, (to_upload.shape[1], to_upload.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )
    treated = treated[:h, :w]

    return treated


_DEFAULT_JPEG_QUALITY = 95


def _jpeg_save_params(src_path: Path) -> dict:
    """Parametri di salvataggio JPEG che riproducono la compressione del
    sorgente: si copiano le tabelle di quantizzazione (e il sottocampionamento
    croma) della pagina originale, cosi' la pagina trattata non viene ne'
    ricompressa piu' del fumetto ne' gonfiata a qualita' 100. Se il sorgente
    non e' un JPEG (png/webp) non c'e' compressione da copiare: si usa una
    qualita' alta di default."""
    try:
        with Image.open(src_path) as src:
            if src.format != "JPEG" or not getattr(src, "quantization", None):
                return {"quality": _DEFAULT_JPEG_QUALITY}
            params = {"qtables": src.quantization}
            try:
                sampling = JpegImagePlugin.get_sampling(src)
                if sampling != -1:
                    params["subsampling"] = sampling
            except Exception:
                pass
            if src.info.get("progressive"):
                params["progressive"] = True
            return params
    except Exception as e:
        log.warning(f"Parametri JPEG di {src_path.name} illeggibili ({e}): uso qualita' {_DEFAULT_JPEG_QUALITY}")
        return {"quality": _DEFAULT_JPEG_QUALITY}


def run(page_path: Path, cfg: dict, output_dir: Path, prompt_text: str,
        megapixels: float | None = None) -> Path:
    image = cv2.imread(str(page_path))
    if image is None:
        raise FileNotFoundError(f"Impossibile leggere immagine: {page_path}")

    treated = _call_comfyui(image, cfg, prompt_text, megapixels)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{page_path.stem}.jpg"
    rgb = cv2.cvtColor(treated, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_path, format="JPEG", **_jpeg_save_params(page_path))
    return out_path


def _collect_pages(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not input_dir.exists():
        return []
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)


def _find_comics(base_input_dir: Path) -> list[str]:
    if not base_input_dir.exists():
        log.error(f"Cartella input non trovata: {base_input_dir}")
        return []
    subfolders = sorted(p.name for p in base_input_dir.iterdir() if p.is_dir())
    if subfolders:
        return subfolders
    if _collect_pages(base_input_dir):
        return [""]
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrattamento pagine grezze via ComfyUI (Flux-Klein.json) con prompt libero, "
                    "prima di OCR/detection. Isolato dalla pipeline principale, scrive in pretreated_pages/."
    )
    parser.add_argument("--config", default=str(paths.CONFIG_PATH))
    parser.add_argument("--comic", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--prompt", required=True, help="Prompt per questa esecuzione (non viene salvato)")
    parser.add_argument("--megapixels", type=float, default=None,
                        help="Ridimensionamento interno del workflow: 0 = piena risoluzione "
                             "(bypassa il nodo di scala), >0 = megapixel di lavoro, "
                             "omesso = come salvato nel workflow.")
    parser.add_argument("--input-dir", default=None,
                        help="Cartella base di partenza (default: paths.input_dir dal config). "
                             "Le pagine sono lette da <input-dir>/<fumetto>/.")
    parser.add_argument("--output-dir", default=None,
                        help="Cartella base di destinazione (default: pretreated_pages/ accanto alla cartella input). "
                             "Le pagine finiscono in <output-dir>/<fumetto>/.")
    args = parser.parse_args()

    cfg = paths.load_cfg(args.config)
    if args.input_dir:
        base_input_dir = Path(args.input_dir).expanduser()
        if not base_input_dir.exists():
            log.error(f"Cartella input non trovata: {base_input_dir}")
            raise SystemExit(1)
    else:
        base_input_dir = Path(cfg["paths"]["input_dir"])
    log.info(f"Sorgente: {base_input_dir}")
    # Cartella sorella della cartella input, mai dentro di essa: non deve mai
    # poter essere scambiata per l'input originale ne' rischiare di
    # sovrascriverlo.
    if args.output_dir:
        base_pretreated_dir = Path(args.output_dir).expanduser()
        if base_pretreated_dir.resolve() == base_input_dir.resolve():
            log.error("La cartella di destinazione coincide con quella di input: sovrascriverebbe gli originali.")
            raise SystemExit(1)
    else:
        base_pretreated_dir = base_input_dir.parent / "pretreated_pages"
    log.info(f"Destinazione: {base_pretreated_dir}")

    comics = _find_comics(base_input_dir)
    if args.comic:
        comics = [args.comic] if args.comic in comics else []
        if not comics:
            log.error(f"Fumetto '{args.comic}' non trovato.")

    for comic in comics:
        input_dir = base_input_dir / comic if comic else base_input_dir
        output_dir = base_pretreated_dir / comic if comic else base_pretreated_dir

        pages = _collect_pages(input_dir)
        if args.limit:
            pages = pages[:args.limit]
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else len(pages)
        pages = pages[start:end]

        log.info(f"=== Pretrattamento: {comic or '(root)'} - {len(pages)} pagine ===")
        for page_path in pages:
            log.info(f"=== Pretrattamento: {page_path.name} ===")
            try:
                out_path = run(page_path, cfg, output_dir, args.prompt, args.megapixels)
                log.info(f"Pagina pretrattata: {out_path}")
            except Exception as e:
                log.error(f"Errore pretrattando {page_path.name}: {e}")

    log.info("Completato.")
