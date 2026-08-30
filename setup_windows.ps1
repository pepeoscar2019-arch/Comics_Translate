# Ricrea da zero i 4 venv del progetto con Python nativo di Windows.
# I venv esistenti (venv, venv_ctd, venv_bubbleseg) sono stati
# creati su Linux e copiati qui: i loro eseguibili sono symlink rotti verso
# /usr/bin/python3, quindi vanno ricreati, non riparati. Le versioni esatte
# del venv Linux non vengono preservate (erano comunque irriproducibili su
# Windows/Python 3.14): si installano le versioni correnti compatibili.
$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$TorchIndex = "https://download.pytorch.org/whl/cpu"

function New-CleanVenv($name) {
    Write-Host "=== $name ==="
    if (Test-Path $name) {
        Write-Host "Rimuovo il venv Linux esistente..."
        try { Remove-Item -Recurse -Force $name -ErrorAction Stop } catch {
            Write-Host "Alcuni file erano occupati, procedo comunque (verranno sovrascritti)."
        }
    }
    Write-Host "Creo il venv..."
    py -3.14 -m venv $name
}

# --- venv principale (pipeline: Flask, GUI, OCR/traduzione via API) ---
New-CleanVenv "venv"
$py = "venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install flask pyyaml requests groq openai opencv-python-headless pillow pydantic pyphen tqdm

# --- venv_ctd (comic-text-detector: rilevamento testo, torch CPU) ---
New-CleanVenv "venv_ctd"
$py = "venv_ctd\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install --index-url $TorchIndex torch torchvision
& $py -m pip install opencv-python numpy tqdm pyyaml shapely pyclipper pillow requests torchsummary wandb

# --- venv_bubbleseg (rilevamento baloon con YOLO, torch CPU) ---
New-CleanVenv "venv_bubbleseg"
$py = "venv_bubbleseg\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install --index-url $TorchIndex torch torchvision
& $py -m pip install ultralytics opencv-python numpy

Write-Host ""
Write-Host "Fatto. Ricontrolla config.yaml (paths, modelli GGUF) prima di avviare start_pipeline.ps1."
