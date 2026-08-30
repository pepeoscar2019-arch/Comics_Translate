# Avvia ComfyUI (Desktop standalone) in background e attende che sia pronto.
$ErrorActionPreference = "Stop"

# Percorsi della tua installazione di ComfyUI. Si possono impostare da fuori
# come variabili d'ambiente, senza modificare questo script:
#   $env:COMFYUI_DIR = "D:\ComfyUI"
#   $env:COMFYUI_EXTRA_MODEL_PATHS = "...\extra_model_paths.yaml"
$InstallDir = if ($env:COMFYUI_DIR) { $env:COMFYUI_DIR } else { "D:\ComfyUI" }
$Python = if ($env:COMFYUI_PYTHON) { $env:COMFYUI_PYTHON } else { Join-Path $InstallDir "standalone-env\python.exe" }
$MainPy = Join-Path $InstallDir "ComfyUI\main.py"
# Facoltativo: serve solo se i modelli stanno fuori dalla cartella di ComfyUI.
$ExtraModelPaths = $env:COMFYUI_EXTRA_MODEL_PATHS

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Log = Join-Path $ScriptDir "comfyui.log"
$PidFile = Join-Path $ScriptDir "comfyui.pid"
$Port = 8188

if (-not (Test-Path $Python)) {
    Write-Host "ERRORE: $Python non trovato."
    exit 1
}
if (-not (Test-Path $MainPy)) {
    Write-Host "ERRORE: $MainPy non trovato."
    exit 1
}
if ($ExtraModelPaths -and -not (Test-Path $ExtraModelPaths)) {
    Write-Host "ERRORE: $ExtraModelPaths non trovato (COMFYUI_EXTRA_MODEL_PATHS)."
    exit 1
}

$portInUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "ComfyUI risulta gia' in ascolto sulla $Port, non lo riavvio."
    exit 0
}

Write-Host "Avvio ComfyUI in background..."
$argomenti = @("`"$MainPy`"")
if ($ExtraModelPaths) { $argomenti += @("--extra-model-paths-config", "`"$ExtraModelPaths`"") }
$argomenti += @("--enable-manager", "--use-sage-attention", "--port", "$Port")
$proc = Start-Process -FilePath $Python `
    -ArgumentList $argomenti `
    -WorkingDirectory (Join-Path $InstallDir "ComfyUI") `
    -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File -FilePath $PidFile -Encoding ascii
Write-Host "PID: $($proc.Id)"

Write-Host "Attendo che sia pronto (il primo avvio carica i modelli, puo' richiedere piu' tempo)..."
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/system_stats" -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) {
            Write-Host "ComfyUI pronto dopo $($i*2)s."
            exit 0
        }
    } catch {}
    Write-Host "  attesa... $($i*2)s"
}

Write-Host "ComfyUI non risponde dopo 120s, controlla $Log"
Get-Content $Log -Tail 30 -ErrorAction SilentlyContinue
exit 1
