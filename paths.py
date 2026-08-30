"""
Fonte unica di verita' per i percorsi del progetto e per il caricamento
della configurazione. Prima di questo modulo, ~20 percorsi assoluti a
"/home/marco/fumetti/..." o "~/fumetti/..." erano duplicati in piu' file
(detect.py, clean.py, main.py, pipeline_gui_tk.py, script shell), e
CONFIG_FILE = "config.yaml" era relativo alla cwd di lancio: lanciare uno
script da una directory diversa da quella del progetto rompeva tutto in
modo silenzioso. PROJECT_ROOT si ancora sempre alla posizione di questo
file, indipendentemente da dove viene lanciato lo script chiamante.
"""

import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("FUMETTI_ROOT", Path(__file__).resolve().parent))

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

VENV_CTD = PROJECT_ROOT / "venv_ctd"
VENV_BUBBLESEG = PROJECT_ROOT / "venv_bubbleseg"
VENV_IOPAINT = PROJECT_ROOT / "venv_iopaint"

CTD_SCRIPT = PROJECT_ROOT / "comic-text-detector" / "detect_export.py"
BUBBLESEG_SCRIPT = PROJECT_ROOT / "bubble_seg" / "detect_bubbles.py"
BUBBLESEG_MODEL_DEFAULT = PROJECT_ROOT / "bubble_seg" / "model.pt"


def venv_bin(venv_dir: Path, name: str) -> str:
    """Percorso all'eseguibile `name` dentro un venv, indipendente dal SO:
    su Windows vive in Scripts/ con estensione .exe, su Linux/macOS in bin/
    senza estensione."""
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / f"{name}.exe")
    return str(venv_dir / "bin" / name)


def detached_popen_kwargs() -> dict:
    """Kwargs da passare a subprocess.Popen per staccare il processo figlio
    dal padre (sopravvive alla chiusura dello script che lo ha lanciato),
    indipendente dal SO: start_new_session su POSIX, CREATE_NEW_PROCESS_GROUP
    su Windows (start_new_session non e' supportato da subprocess su Windows)."""
    if sys.platform == "win32":
        import subprocess
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def resolve(path_str: str) -> Path:
    """Risolve un path (da config.yaml o default) contro PROJECT_ROOT se
    relativo, lasciandolo invariato se gia' assoluto. Serve per le chiavi
    di config che puntano dentro il progetto (es. bubble_seg.model_path,
    rendering.font_path) invece che ai dati pesanti su disco esterno
    (quelli restano assoluti verso /media/... e non passano di qui)."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_cfg(config_path: str | Path | None = None) -> dict:
    """Carica config.yaml. Sostituisce sia il load_cfg() di
    pipeline_gui_tk.py sia il load_config(path) di main.py: firma
    compatibile con entrambi (default a CONFIG_PATH, ma accetta un path
    esplicito per --config da CLI)."""
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
