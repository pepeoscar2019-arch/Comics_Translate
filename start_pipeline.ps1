# Avvia la GUI della pipeline, dopo un paio di controlli d'ambiente.
$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Cartella di lavoro; impostabile con $env:FUMETTI_WORK senza toccare lo script.
$InputDir = if ($env:FUMETTI_WORK) { $env:FUMETTI_WORK } else { "C:\Fumetti_Work" }

Write-Host "=== Avvio pipeline fumetti ==="

# 1) Cartella con le pagine da tradurre
if (-not (Test-Path $InputDir)) {
    Write-Host "ATTENZIONE: $InputDir non trovato."
}

# 2) Modello locale: nessun server da avviare qui. OCR e traduzione girano su
# llama-server (llama.cpp), che main.py lancia all'inizio dello stage e chiude
# alla fine, cosi' la VRAM torna libera per la pulizia. Qui si controlla solo
# che il binario ci sia.
if (Get-Command llama-server -ErrorAction SilentlyContinue) {
    Write-Host "llama-server: trovato nel PATH."
} else {
    Write-Host "ATTENZIONE: 'llama-server' non e' nel PATH: OCR e traduzione falliranno."
    Write-Host "Installalo con: winget install ggml.llamacpp"
}

# 3) GUI della pipeline (Tkinter)
$guiRunning = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*pipeline_gui_tk.py*" }
if ($guiRunning) {
    Write-Host "GUI: gia' in esecuzione."
} else {
    Write-Host "Avvio GUI pipeline..."
    & (Join-Path $ScriptDir "venv\Scripts\python.exe") (Join-Path $ScriptDir "pipeline_gui_tk.py")
}
