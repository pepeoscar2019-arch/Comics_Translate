#!/usr/bin/env python3
"""
Web app Flask per la pipeline traduzione fumetti.
Rimpiazza pipeline_gui_tk.py con un'interfaccia browser.
Avvia con: python web_app.py
"""

import os, sys, json, shutil, subprocess, threading, time, re, base64, secrets

# La console di Windows puo' essere in cp1252 (non UTF-8): senza questo, i
# print con emoji (es. l'avvio qui sotto) crashano con UnicodeEncodeError
# quando lo script gira fuori da un terminale UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context, send_file
import yaml
import requests

import translate_cache as tc
import audit
import clean
import render as render_mod
import paths
import i18n
from i18n import t
import export_prompt

app = Flask(__name__)
# Chiave di sessione generata a ogni avvio: l'app gira in locale e non ha
# login, ma una chiave fissa scritta nel codice sarebbe identica per chiunque
# scarichi il progetto. FUMETTI_SECRET_KEY la fissa, se serve che le sessioni
# sopravvivano a un riavvio.
app.secret_key = os.environ.get("FUMETTI_SECRET_KEY") or secrets.token_hex(32)

CONFIG_FILE = "config.yaml"
MAIN_SCRIPT = "main.py"


# Stato globale del processo in esecuzione
_process = None
_running = False
_log_buffer = []
_log_lock = threading.Lock()

# Stato globale della rigenerazione pagina (tab Revisione)
_review_regen = {"running": False, "message": "", "error": None}
_review_regen_lock = threading.Lock()


# ── helpers ──────────────────────────────────────────────────────────────────

def load_cfg():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_cfg(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

def llama_server_url(cfg):
    port = cfg.get("llama_server", {}).get("port", 8081)
    return f"http://127.0.0.1:{port}"


def llama_server_state(cfg):
    """(attivo, descrizione) di llama-server. Fuori dagli stage e' spento di
    proposito: main.py lo avvia per OCR/traduzione e lo chiude alla fine."""
    if not shutil.which("llama-server"):
        return False, "llama-server non trovato nel PATH"
    try:
        resp = requests.get(f"{llama_server_url(cfg)}/v1/models", timeout=2)
        if resp.status_code == 200:
            data = resp.json().get("data") or []
            return True, (data[0].get("id") if data else "modello sconosciuto")
    except Exception:
        pass
    return False, "spento (si avvia all'inizio dello stage)"


def llama_model_rows(cfg):
    """(etichetta, path, esiste) dei GGUF configurati."""
    ls = cfg.get("llama_server", {})
    # Il proiettore mostrato e' quello del backend di visione attivo: qwen e
    # paddleocr hanno mmproj diversi e non intercambiabili.
    mmproj_key = ("paddleocr_mmproj_path"
                  if cfg.get("ocr_backend", "qwen") == "paddleocr_vl" else "ocr_mmproj_path")
    righe = []
    for etichetta, chiave in (("OCR (qwen)", "ocr_model_path"),
                              ("OCR (paddleocr_vl)", "paddleocr_model_path"),
                              ("Proiettore mmproj", mmproj_key),
                              ("Traduzione", "translate_model_path")):
        path = ls.get(chiave, "")
        righe.append((etichetta, path, bool(path) and Path(path).exists()))
    return righe

# Il workflow Flux-Klein.json serve ancora al solo pretrattamento
# (flux_pretreat.py), che gli passa il proprio prompt a ogni esecuzione. La
# pulizia dei balloon non usa piu' modelli generativi.

def collect_comics(cfg):
    base = Path(cfg["paths"]["input_dir"])
    if not base.exists():
        return []
    subdirs = sorted(d.name for d in base.iterdir() if d.is_dir())
    if subdirs:
        return subdirs
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if any(p.suffix.lower() in exts for p in base.iterdir()):
        return [""]
    return []

def count_pages(cfg, comic):
    base = Path(cfg["paths"]["input_dir"])
    d = base / comic if comic else base
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return len([p for p in d.iterdir() if p.suffix.lower() in exts]) if d.exists() else 0

def push_log(msg):
    with _log_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        _log_buffer.append(f"[{ts}] {msg}")
        if len(_log_buffer) > 500:
            _log_buffer.pop(0)

def build_command(stage, comic=None, start=None, end=None, force=False, ocr_backend=None, prompt=None):
    # Pretrattamento ComfyUI (flux_pretreat.py): script isolato, main.py non lo
    # conosce. Il prompt e' passato a ogni esecuzione, mai scritto nel workflow.
    if stage == "pretreat":
        cmd = [sys.executable, "-u", str(paths.PROJECT_ROOT / "flux_pretreat.py"),
               "--config", CONFIG_FILE, "--prompt", prompt or ""]
    else:
        cmd = [sys.executable, "-u", MAIN_SCRIPT, "--config", CONFIG_FILE, "--stage", stage]
    if comic:
        cmd += ["--comic", comic]
    if start is not None:
        cmd += ["--start", str(start)]
    if end is not None:
        cmd += ["--end", str(end)]
    # --force-translate non esiste in flux_pretreat.py (non traduce), solo in main.py.
    if force and stage != "pretreat":
        cmd += ["--force-translate"]
    if ocr_backend and stage == "ocr":
        cmd += ["--ocr-backend", ocr_backend]
    return cmd


# ── HTML template (HTMX + TailwindCDN) ───────────────────────────────────────

HTML = r"""
<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Pipeline Fumetti</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/htmx.org@1.9.12"></script>
<style>
  #console { font-family: 'Consolas','Courier New',monospace; font-size:0.78rem; }
  .tab-btn.active { @apply bg-indigo-600 text-white; }
  .tab-content { display:none; }
  .tab-content.active { display:block; }
</style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen p-4">

<h1 class="text-2xl font-bold text-indigo-400 mb-4">🎌 Pipeline Traduzione Fumetti</h1>

<!-- Status bar -->
<div id="status-bar" class="bg-gray-800 rounded p-3 mb-4 flex flex-wrap gap-4 text-sm"
     hx-get="/api/status" hx-trigger="load, every 5s" hx-swap="innerHTML">
  Caricamento…
</div>

<!-- Progress -->
<div class="bg-gray-800 rounded p-3 mb-4">
  <div class="flex justify-between text-xs mb-1">
    <span id="stage-label">In attesa</span>
    <span id="pct-label">0%</span>
  </div>
  <div class="w-full bg-gray-700 rounded h-3 mb-2">
    <div id="progress-bar" class="bg-indigo-500 h-3 rounded transition-all" style="width:0%"></div>
  </div>
  <div id="stage-detail" class="text-xs text-gray-400"></div>
</div>

<!-- Tabs -->
<div class="mb-2 flex gap-2 flex-wrap">
  {% for t in tabs %}
  <button onclick="switchTab('{{t[0]}}')"
    class="tab-btn px-4 py-1.5 rounded text-sm font-medium bg-gray-700 hover:bg-indigo-500 transition"
    id="btn-{{t[0]}}">{{t[1]}}</button>
  {% endfor %}
</div>

<!-- ── TAB: Pipeline ── -->
<div id="tab-pipeline" class="tab-content bg-gray-800 rounded p-4">
  <h2 class="font-semibold text-indigo-300 mb-3">Stage</h2>
  <div class="grid grid-cols-2 gap-2 mb-4">
    {% for v,l in stages %}
    <label class="flex items-center gap-2 cursor-pointer">
      <input type="radio" name="stage" value="{{v}}" {{'checked' if v=='full' else ''}}
             class="accent-indigo-500"> {{l}}
    </label>
    {% endfor %}
  </div>

  <div class="flex flex-wrap gap-4 mb-4 text-sm">
    <label>Fumetto:
      <select id="sel-comic" class="ml-2 bg-gray-700 rounded px-2 py-1">
        <option value="">tutti</option>
        {% for c in comics %}
        <option value="{{c}}">{{c or '(root)'}}</option>
        {% endfor %}
      </select>
    </label>
    <label>Pagine (es. 5 o 3-10):
      <input id="inp-limit" type="text" placeholder="tutte"
             class="ml-2 bg-gray-700 rounded px-2 py-1 w-24">
    </label>
    <label class="flex items-center gap-2 cursor-pointer" title="Ignora la cache traduzioni e richiama sempre l'AI, anche per pagine già tradotte">
      <input id="chk-force" type="checkbox" class="accent-indigo-500">
      Forza ritraduzione (ignora cache)
    </label>
    <label title="Modello di visione usato dallo stage OCR">OCR via:
      <select id="sel-ocr-backend" class="ml-2 bg-gray-700 rounded px-2 py-1">
        <option value="qwen" {{'selected' if ocr_backend=='qwen' else ''}}>Qwen3-VL-8B</option>
        <option value="paddleocr_vl" {{'selected' if ocr_backend=='paddleocr_vl' else ''}}>PaddleOCR-VL-1.6 (veloce)</option>
      </select>
    </label>
  </div>

  <div class="flex gap-3">
    <button onclick="startPipeline()"
      class="bg-indigo-600 hover:bg-indigo-500 px-5 py-2 rounded font-medium transition">
      ▶ Avvia
    </button>
    <button onclick="stopPipeline()"
      class="bg-red-700 hover:bg-red-600 px-5 py-2 rounded font-medium transition">
      ■ Ferma
    </button>
  </div>

  <h2 class="font-semibold text-indigo-300 mt-6 mb-2">Pretrattamento pagine (ComfyUI)</h2>
  <p class="text-xs text-gray-400 mb-2">
    Passa le pagine grezze (fumetto/intervallo scelti sopra) attraverso ComfyUI/Flux-Klein
    con un prompt libero, PRIMA di OCR — utile per upscale/pulizia rumore/altro. Scrive in una
    cartella separata (pretreated_pages/), non tocca mai gli originali in input_pages/. Il
    prompt qui sotto resta salvato per la prossima volta.
  </p>
  <textarea id="inp-pretreat-prompt" rows="4" placeholder="Es: upscale and sharpen the image, keep composition and colors unchanged..."
            class="w-full mt-1 bg-gray-700 rounded px-3 py-2 font-mono text-xs max-w-2xl">{{pretreat_prompt}}</textarea>
  <div class="mt-2">
    <button onclick="startPretreat()"
      class="bg-indigo-600 hover:bg-indigo-500 px-5 py-2 rounded font-medium transition">
      ▶ Avvia pretrattamento
    </button>
  </div>
</div>

<!-- ── TAB: Config ── -->
<div id="tab-config" class="tab-content bg-gray-800 rounded p-4">
  <h2 class="font-semibold text-indigo-300 mb-3">Lingua interfaccia</h2>
  <p class="text-xs text-gray-400 mb-2">
    Lingua dell'interfaccia. I messaggi di log della pipeline restano in italiano.
  </p>
  <div class="flex items-center gap-3 mb-4">
    <select name="ui_language" class="bg-gray-700 rounded px-3 py-1.5 text-sm">
      {% for codice, nome in lingue.items() %}
      <option value="{{codice}}" {{'selected' if lingua_attuale==codice else ''}}>{{nome}}</option>
      {% endfor %}
    </select>
    <button hx-post="/api/save-language" hx-include="[name='ui_language']"
            hx-target="#cfg-msg" hx-swap="innerHTML"
            class="bg-indigo-600 hover:bg-indigo-500 px-4 py-1.5 rounded text-sm">
      Salva lingua
    </button>
  </div>
  <div id="cfg-msg" class="text-green-400 text-sm mb-4"></div>

  <h2 class="font-semibold text-indigo-300 mt-4 mb-3">Modelli GGUF (llama-server)</h2>
  <p class="text-xs text-gray-400 mb-2">
    Path dei .gguf che llama-server carica per OCR e traduzione. Un modello per volta:
    il server parte all'inizio dello stage e viene chiuso alla fine, cosi' la VRAM
    torna libera per la pulizia.
  </p>
  <div class="grid grid-cols-1 gap-2 mb-3 max-w-3xl">
    <label class="text-sm">Modello OCR (qwen)
      <input name="ocr_model_path" value="{{ocr_model_path}}"
             class="w-full mt-1 bg-gray-700 rounded px-3 py-1.5 font-mono text-xs">
    </label>
    <label class="text-sm">Modello OCR (paddleocr_vl)
      <input name="paddleocr_model_path" value="{{paddleocr_model_path}}"
             class="w-full mt-1 bg-gray-700 rounded px-3 py-1.5 font-mono text-xs">
    </label>
    <label class="text-sm">Proiettore multimodale (mmproj)
      <input name="ocr_mmproj_path" value="{{ocr_mmproj_path}}"
             class="w-full mt-1 bg-gray-700 rounded px-3 py-1.5 font-mono text-xs">
    </label>
    <label class="text-sm">Modello traduzione
      <input name="translate_model_path" value="{{translate_model_path}}"
             class="w-full mt-1 bg-gray-700 rounded px-3 py-1.5 font-mono text-xs">
    </label>
    <label class="text-sm">Porta llama-server
      <input name="llama_port" value="{{llama_port}}"
             class="w-32 mt-1 bg-gray-700 rounded px-3 py-1.5 font-mono text-xs">
    </label>
  </div>
  <button hx-post="/api/save-lm-models"
          hx-include="[name='ocr_model_path'],[name='paddleocr_model_path'],[name='ocr_mmproj_path'],[name='translate_model_path'],[name='llama_port']"
          hx-target="#cfg-msg" hx-swap="innerHTML"
          class="bg-indigo-600 hover:bg-indigo-500 px-4 py-1.5 rounded text-sm">
    Salva modelli
  </button>

  <h2 class="font-semibold text-indigo-300 mt-4 mb-3">Dimensione font fissa</h2>
  <p class="text-xs text-gray-400 mb-2">
    Se impostata, forza questa dimensione (px) su tutti i balloon normali di
    ogni pagina, al posto del calcolo automatico. Lascia vuoto per l'automatico.
  </p>
  <label class="text-sm">
    <input name="fixed_font_size" type="number" min="1" step="1" value="{{fixed_font_size}}"
           placeholder="automatico"
           class="w-32 mt-1 bg-gray-700 rounded px-3 py-1.5 font-mono text-xs">
  </label>
  <div>
    <button hx-post="/api/save-fixed-font-size" hx-include="[name='fixed_font_size']"
            hx-target="#cfg-msg" hx-swap="innerHTML"
            class="bg-indigo-600 hover:bg-indigo-500 px-4 py-1.5 rounded text-sm mt-2">
      Salva dimensione
    </button>
  </div>
</div>

<!-- ── TAB: Correzioni ── -->
<div id="tab-edit" class="tab-content bg-gray-800 rounded p-4">
  <div class="flex flex-wrap gap-4 mb-4 text-sm">
    <label>Fumetto:
      <select id="edit-comic" class="ml-2 bg-gray-700 rounded px-2 py-1"
              onchange="loadEditPages()">
        {% for c in comics %}
        <option value="{{c}}">{{c or '(root)'}}</option>
        {% endfor %}
      </select>
    </label>
    <label>Pagina:
      <select id="edit-page" class="ml-2 bg-gray-700 rounded px-2 py-1">
      </select>
    </label>
    <button onclick="loadEditData()"
      class="bg-indigo-600 hover:bg-indigo-500 px-4 py-1.5 rounded">Carica</button>
    <button onclick="saveEdits()"
      class="bg-green-700 hover:bg-green-600 px-4 py-1.5 rounded">Salva</button>
  </div>
  <div id="edit-table" class="overflow-x-auto"></div>
  <div id="edit-msg" class="mt-2 text-sm text-green-400"></div>

  <h3 class="font-semibold text-indigo-300 mt-6 mb-2">Esporta / Importa testo (TXT)</h3>
  <p class="text-xs text-gray-400 mb-2">
    Esporta tutte le pagine del fumetto selezionato (o un intervallo) in un unico file
    TXT per correggerlo con un editor esterno, poi reimportalo per riportare le modifiche
    nei translated.json. L'import scrive direttamente sui file: se la pagina caricata sopra
    e' tra quelle importate, ricaricala per vedere le modifiche.
  </p>
  <div class="flex flex-wrap gap-3 items-center text-sm">
    <label>Intervallo pagine (es. 5 o 3-10):
      <input id="edit-export-range" type="text" placeholder="tutte"
             class="ml-2 bg-gray-700 rounded px-2 py-1 w-28">
    </label>
    <label>Contenuto:
      <select id="edit-export-mode" class="ml-2 bg-gray-700 rounded px-2 py-1">
        <option value="both">Originale + Tradotto</option>
        <option value="orig">Solo originale</option>
        <option value="trans">Solo tradotto</option>
      </select>
    </label>
    <button onclick="exportEditText()"
      class="bg-indigo-600 hover:bg-indigo-500 px-4 py-1.5 rounded">Esporta in TXT…</button>
    <label class="bg-indigo-600 hover:bg-indigo-500 px-4 py-1.5 rounded cursor-pointer">
      Importa da TXT…
      <input id="edit-import-file" type="file" accept=".txt" class="hidden" onchange="importEditText()">
    </label>
  </div>
</div>

<!-- ── TAB: Revisione ── -->
<div id="tab-review" class="tab-content bg-gray-800 rounded p-4">
  <div class="flex flex-wrap gap-4 mb-3 text-sm items-center">
    <label>Fumetto:
      <select id="rev-comic" class="ml-2 bg-gray-700 rounded px-2 py-1" onchange="revLoadPages()">
        {% for c in comics %}
        <option value="{{c}}">{{c or '(root)'}}</option>
        {% endfor %}
      </select>
    </label>
    <label>Pagina:
      <select id="rev-page" class="ml-2 bg-gray-700 rounded px-2 py-1" onchange="revLoadPage()"></select>
    </label>
    <button onclick="revNav(-1)" class="bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded text-sm">‹ Precedente</button>
    <button onclick="revNav(1)" class="bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded text-sm">Successiva ›</button>
    <button onclick="revNextIssue()" title="Salta al prossimo balloon da trascrivere o con testo troncato (tasto N)"
      class="bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded text-sm">⚠ Prossimo problema</button>
    <button id="rev-regen-btn" onclick="revSaveAndRegenerate()" disabled
      class="bg-green-700 hover:bg-green-600 disabled:opacity-40 disabled:cursor-not-allowed px-4 py-1.5 rounded text-sm font-medium">
      ⟳ Rigenera pagina
    </button>
  </div>
  <p class="text-xs text-gray-400 mb-2">🖱️ Trascina un balloon per spostarlo · trascina gli angoli/lati per ridimensionare · doppio-click per modificare testo e font · <b>N</b> prossimo problema · <b>PagSu/PagGiù</b> cambia pagina</p>
  <div id="rev-canvas-wrap" class="relative bg-black rounded flex items-center justify-center" style="height:70vh;overflow:hidden;">
    <img id="rev-img" style="max-width:100%;max-height:100%;display:block;user-select:none;-webkit-user-drag:none;">
    <div id="rev-boxes" class="absolute top-0 left-0 w-full h-full" style="pointer-events:none;"></div>
  </div>
  <div id="rev-status" class="mt-2 text-sm text-gray-400"></div>
  <div class="mt-3 flex items-center gap-3">
    <button onclick="revAudit()" title="Controlla tutte le pagine del fumetto: testi rimasti in lingua originale, da trascrivere, troncati, stage mancanti"
      class="bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded text-sm">🔍 Controlla fumetto</button>
    <span id="rev-audit-summary" class="text-sm text-gray-400"></span>
  </div>
  <div id="rev-audit" class="mt-2 text-xs"></div>
</div>

<!-- ── TAB: Cache ── -->
<div id="tab-cache" class="tab-content bg-gray-800 rounded p-4">
  <div id="cache-stats" hx-get="/api/cache-stats" hx-trigger="load" hx-swap="innerHTML"
       class="mb-4 text-sm">Caricamento…</div>
  <button hx-post="/api/cache-clear" hx-target="#cache-stats" hx-swap="innerHTML"
          hx-confirm="Svuotare tutta la cache?"
          class="bg-red-700 hover:bg-red-600 px-4 py-1.5 rounded text-sm">
    Svuota cache
  </button>
</div>

<!-- Console -->
<div class="mt-4">
  <div class="flex justify-between items-center mb-1">
    <span class="text-xs text-gray-400 font-semibold uppercase tracking-wide">Output</span>
    <button onclick="document.getElementById('console').innerHTML=''"
            class="text-xs text-gray-500 hover:text-gray-300">Pulisci</button>
  </div>
  <div id="console"
       class="bg-black rounded p-3 h-56 overflow-y-auto text-green-300 text-xs leading-5"
       id="console"></div>
</div>

<script>
// ── Tabs ──
const TABS = {{tab_ids|tojson}};
function switchTab(id) {
  TABS.forEach(t => {
    document.getElementById('tab-' + t).classList.remove('active');
    document.getElementById('btn-' + t).classList.remove('active','bg-indigo-600','text-white');
  });
  document.getElementById('tab-' + id).classList.add('active');
  document.getElementById('btn-' + id).classList.add('active','bg-indigo-600','text-white');
  if (id === 'review' && window._rev) {
    requestAnimationFrame(revRenderBoxes);
  }
}
switchTab('pipeline');

// ── Console SSE ──
const con = document.getElementById('console');
const evtSrc = new EventSource('/api/log-stream');
evtSrc.onmessage = e => {
  const line = document.createElement('div');
  line.textContent = e.data;
  con.appendChild(line);
  con.scrollTop = con.scrollHeight;
};

// ── Progress polling ──
let _progressInterval = null;
function startProgressPolling() {
  if (_progressInterval) return;
  _progressInterval = setInterval(async () => {
    const r = await fetch('/api/progress');
    const d = await r.json();
    document.getElementById('progress-bar').style.width = d.pct + '%';
    document.getElementById('pct-label').textContent = Math.round(d.pct) + '%';
    document.getElementById('stage-label').textContent = d.stage || 'In attesa';
    document.getElementById('stage-detail').textContent = d.detail || '';
    if (!d.running) {
      clearInterval(_progressInterval);
      _progressInterval = null;
    }
  }, 800);
}

// ── Pipeline controls ──
async function startPipeline() {
  const stage = document.querySelector('[name=stage]:checked')?.value || 'full';
  const comic = document.getElementById('sel-comic').value;
  const limit = document.getElementById('inp-limit').value.trim();
  const force = document.getElementById('chk-force').checked;
  const ocr_backend = document.getElementById('sel-ocr-backend').value;
  const r = await fetch('/api/run', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({stage, comic, limit, force, ocr_backend})
  });
  const d = await r.json();
  if (d.error) alert(d.error);
  else startProgressPolling();
}
async function stopPipeline() {
  await fetch('/api/stop', {method:'POST'});
}
async function startPretreat() {
  const comic = document.getElementById('sel-comic').value;
  const limit = document.getElementById('inp-limit').value.trim();
  const prompt = document.getElementById('inp-pretreat-prompt').value.trim();
  if (!prompt) { alert('Scrivi un prompt prima di avviare il pretrattamento'); return; }
  const r = await fetch('/api/run', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({stage: 'pretreat', comic, limit, prompt})
  });
  const d = await r.json();
  if (d.error) alert(d.error);
  else startProgressPolling();
}

// ── Edit tab ──
async function loadEditPages() {
  const comic = document.getElementById('edit-comic').value;
  const r = await fetch('/api/edit-pages?comic=' + encodeURIComponent(comic));
  const pages = await r.json();
  const sel = document.getElementById('edit-page');
  sel.innerHTML = pages.map(p => `<option value="${p}">${p}</option>`).join('');
}
async function loadEditData() {
  const comic = document.getElementById('edit-comic').value;
  const page  = document.getElementById('edit-page').value;
  const r = await fetch('/api/edit-load?comic=' + encodeURIComponent(comic) + '&page=' + encodeURIComponent(page));
  const data = await r.json();
  if (data.error) { document.getElementById('edit-msg').textContent = data.error; return; }
  window._editData = data;
  renderEditTable(data);
}
function renderEditTable(data) {
  const rows = data.detections.map((d,i) => `
    <tr class="border-b border-gray-700">
      <td class="px-2 py-1 text-gray-400 text-xs">${d.balloon_id}</td>
      <td class="px-2 py-1 text-xs max-w-xs">${escHtml(d.testo_originale||'')}</td>
      <td class="px-2 py-1">
        <input data-idx="${i}" value="${escHtml(d.testo_tradotto||'')}"
               class="edit-trans w-full bg-gray-700 rounded px-2 py-0.5 text-xs">
      </td>
      <td class="px-2 py-1 text-center whitespace-nowrap">
        <button onclick="markOno(${i})" class="text-xs text-yellow-400 hover:underline" title="Marca come onomatopea (svuota la traduzione)">-</button>
        <button onclick="restoreOriginal(${i})" class="text-xs text-blue-400 hover:underline ml-2" title="Ripristina il testo originale come traduzione">↺</button>
      </td>
    </tr>`).join('');
  document.getElementById('edit-table').innerHTML = `
    <table class="w-full text-sm">
      <thead><tr class="text-left text-gray-400 text-xs border-b border-gray-600">
        <th class="px-2 py-1">ID</th>
        <th class="px-2 py-1">Originale</th>
        <th class="px-2 py-1">Traduzione</th>
        <th class="px-2 py-1">Azioni</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}
function markOno(i) {
  window._editData.detections[i].testo_tradotto = '-';
  renderEditTable(window._editData);
}
function restoreOriginal(i) {
  window._editData.detections[i].testo_tradotto = window._editData.detections[i].testo_originale || '';
  renderEditTable(window._editData);
}
function exportEditText() {
  const comic = document.getElementById('edit-comic').value;
  const range = document.getElementById('edit-export-range').value.trim();
  const mode  = document.getElementById('edit-export-mode').value;
  const url = '/api/edit-export?comic=' + encodeURIComponent(comic)
    + '&mode=' + encodeURIComponent(mode) + '&range=' + encodeURIComponent(range);
  window.location.href = url;
}
async function importEditText() {
  const fileInput = document.getElementById('edit-import-file');
  const file = fileInput.files[0];
  if (!file) return;
  const comic = document.getElementById('edit-comic').value;
  const fd = new FormData();
  fd.append('comic', comic);
  fd.append('file', file);
  document.getElementById('edit-msg').textContent = '⏳ Importazione in corso...';
  const r = await fetch('/api/edit-import', { method: 'POST', body: fd });
  const d = await r.json();
  fileInput.value = '';
  if (d.error) {
    document.getElementById('edit-msg').textContent = 'Errore: ' + d.error;
    return;
  }
  document.getElementById('edit-msg').textContent =
    `Importato: ${d.pages} pagine modificate, ${d.balloons} balloon aggiornati`;
  const page = document.getElementById('edit-page').value;
  if (d.pages_touched && d.pages_touched.includes(page)) {
    loadEditData();
  }
}
async function saveEdits() {
  if (!window._editData) return;
  document.querySelectorAll('.edit-trans').forEach((inp, i) => {
    window._editData.detections[i].testo_tradotto = inp.value;
  });
  const comic = document.getElementById('edit-comic').value;
  const page  = document.getElementById('edit-page').value;
  document.getElementById('edit-msg').textContent = '⏳ Salvataggio e rigenerazione in corso...';
  // Usa lo stesso endpoint della scheda Revisione: salva il JSON e rilancia
  // pulizia+render, altrimenti un balloon marcato come onomatopea (o
  // qualsiasi altra modifica al testo) resterebbe con il vecchio render
  // gia' presente nell'immagine, mai rigenerata.
  const r = await fetch('/api/review-regenerate', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({comic, page, data: window._editData})
  });
  const d = await r.json();
  if (d.error) {
    document.getElementById('edit-msg').textContent = d.error;
    return;
  }
  editPollRegenerate();
}
async function editPollRegenerate() {
  const r = await fetch('/api/review-regenerate-status');
  const s = await r.json();
  if (s.running) {
    setTimeout(editPollRegenerate, 700);
    return;
  }
  document.getElementById('edit-msg').textContent = s.error
    ? ('Errore: ' + s.error)
    : (s.message || 'Pagina rigenerata');
}
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Review tab ──
window._rev = null;      // {comic, page, data, naturalW, naturalH, scale, offsetX, offsetY, dirty}
window._revFonts = null; // {name: path}
let _revDrag = null;
let _revEditorIdx = null;   // idx del balloon in editing (per l'anteprima live), null se editor chiuso
let _revPreviewTimer = null;
let _revLastTap = null;     // { idx, t, x, y } dell'ultimo tap semplice, per rilevare il doppio tap a mano
let _revPreviewSeq = 0;     // scarta risposte di preview arrivate fuori ordine

document.getElementById('rev-img').addEventListener('load', () => {
  if (!window._rev) return;
  const img = document.getElementById('rev-img');
  window._rev.naturalW = img.naturalWidth;
  window._rev.naturalH = img.naturalHeight;
  revRenderBoxes();
});
window.addEventListener('resize', () => { if (window._rev && window._rev.data) revRenderBoxes(); });
// Scorciatoie della Revisione. Attive solo quando la scheda e' quella aperta e
// non si sta scrivendo in un campo, altrimenti "n" finirebbe nel testo del
// balloon invece di far navigare.
window.addEventListener('keydown', (e) => {
  if (!window._rev) return;
  if (!document.getElementById('tab-review').classList.contains('active')) return;
  const el = document.activeElement;
  if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT')) return;
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key === 'n' || e.key === 'N') { e.preventDefault(); revNextIssue(); }
  else if (e.key === 'PageDown') { e.preventDefault(); revNav(1); }
  else if (e.key === 'PageUp') { e.preventDefault(); revNav(-1); }
});
window.addEventListener('beforeunload', (e) => {
  if (window._rev && window._rev.dirty) { e.preventDefault(); e.returnValue = ''; }
});

async function revLoadPages() {
  const comic = document.getElementById('rev-comic').value;
  const r = await fetch('/api/review-pages?comic=' + encodeURIComponent(comic));
  const pages = await r.json();
  const sel = document.getElementById('rev-page');
  sel.innerHTML = pages.map(p => `<option value="${escHtml(p)}">${escHtml(p)}</option>`).join('');
  if (pages.length) revLoadPage();
  else {
    window._rev = null;
    document.getElementById('rev-img').removeAttribute('src');
    document.getElementById('rev-boxes').innerHTML = '';
    document.getElementById('rev-status').textContent = 'Nessuna pagina renderizzata trovata.';
  }
}

async function revLoadPage() {
  if (window._rev && window._rev.dirty) {
    if (!confirm('Ci sono modifiche non rigenerate. Vuoi scartarle e continuare?')) return;
  }
  revCloseEditor();
  const comic = document.getElementById('rev-comic').value;
  const page  = document.getElementById('rev-page').value;
  if (!page) return;
  const r = await fetch('/api/review-load?comic=' + encodeURIComponent(comic) + '&page=' + encodeURIComponent(page));
  const res = await r.json();
  if (res.error) {
    document.getElementById('rev-status').innerHTML = `<span class="text-red-400">${escHtml(res.error)}</span>`;
    return;
  }
  if (!window._revFonts) {
    const rf = await fetch('/api/review-fonts');
    window._revFonts = await rf.json();
  }
  window._rev = {comic, page, data: res.data, dirty: false, scale: 1, offsetX: 0, offsetY: 0};
  document.getElementById('rev-regen-btn').disabled = true;
  const dets = res.data.detections || [];
  const toTranscribe = dets.filter(d => d.ocr_empty_suspect &&
                                        !(d.testo_tradotto || '').trim()).length;
  const overflowing = dets.filter(d => d._overflow &&
                                       (d.testo_tradotto || '').trim()).length;
  document.getElementById('rev-status').innerHTML =
    `Pagina: ${escHtml(page)} | ${dets.length} balloon` +
    (toTranscribe
      ? ` | <span class="text-red-400">⚠ ${toTranscribe} da trascrivere ` +
        `(OCR vuoto ma il balloon contiene testo)</span>`
      : '') +
    (overflowing
      ? ` | <span class="text-orange-400">✂ ${overflowing} con testo troncato ` +
        `(da accorciare)</span>`
      : '');
  document.getElementById('rev-img').src = res.image_url;
}

// Balloon che richiedono un intervento, nell'ordine di lettura della pagina:
// quelli che l'OCR ha lasciato vuoti pur avendo testo, e quelli il cui testo
// e' stato troncato nel render. Sono gli unici due casi in cui la tavola e'
// oggettivamente sbagliata (manca del testo), il resto e' questione di gusto.
// Rapporto sull'intero fumetto (audit.py). Ogni riga porta alla pagina e al
// balloon interessato: e' il punto in cui si passa da "quali problemi ci sono"
// a "sistemali", senza dover cercare la pagina a mano nel menu a tendina.
async function revAudit() {
  const comic = document.getElementById('rev-comic').value;
  const box = document.getElementById('rev-audit');
  const sum = document.getElementById('rev-audit-summary');
  sum.textContent = 'Controllo in corso…';
  box.innerHTML = '';
  const r = await fetch('/api/audit?comic=' + encodeURIComponent(comic));
  const res = await r.json();
  if (res.error) { sum.innerHTML = `<span class="text-red-400">${escHtml(res.error)}</span>`; return; }

  const t = res.totals;
  const problemi = t.non_tradotto + t.da_trascrivere + t.troncato + t.troppo_lungo +
                   (t.doppione || 0) + t.pagine_incomplete;
  sum.innerHTML = problemi
    ? `<span class="text-orange-400">${problemi} da sistemare</span> su ${t.balloon} balloon: ` +
      `${t.non_tradotto} non tradotti · ${t.da_trascrivere} da trascrivere · ` +
      `${t.troncato} troncati · ${t.troppo_lungo} troppo lunghi · ` +
      `${t.doppione} doppioni · ${t.pagine_incomplete} stage mancanti`
    : `<span class="text-green-400">Nessun problema</span> su ${t.balloon} balloon.`;

  const colori = {non_tradotto:'text-red-400', da_trascrivere:'text-red-400',
                  doppione:'text-red-400',
                  troncato:'text-orange-400', troppo_lungo:'text-yellow-400'};
  const righe = [];
  res.pages.forEach(pg => {
    pg.issues.forEach(is => {
      const bal = (is.balloon !== undefined && is.balloon !== null) ? ` balloon ${is.balloon}` : '';
      righe.push(
        `<div class="py-0.5"><a href="#" onclick="revGoTo('${escHtml(pg.page)}', ${is.balloon ?? -1});return false;"
           class="text-indigo-400 hover:underline">${escHtml(pg.page)}${bal}</a>
         <span class="${colori[is.tipo] || 'text-gray-400'}">[${escHtml(is.tipo)}]</span>
         <span class="text-gray-400">${escHtml(is.dettaglio || '')}</span></div>`);
    });
  });
  box.innerHTML = righe.length
    ? `<div class="bg-gray-900 rounded p-2 max-h-64 overflow-y-auto">${righe.join('')}</div>`
    : '';
}

// Apre la pagina indicata e, se il problema riguarda un balloon preciso, ne
// apre direttamente l'editor: l'immagine va attesa perche' i box si disegnano
// solo quando si conoscono le dimensioni reali della tavola.
async function revGoTo(page, balloonId) {
  const sel = document.getElementById('rev-page');
  if (sel.value !== page) {
    sel.value = page;
    await revLoadPage();
  }
  if (balloonId === null || balloonId < 0) return;
  const apri = () => {
    const idx = (window._rev.data.detections || []).findIndex(d => d.balloon_id === balloonId);
    if (idx >= 0) revOpenEditor(idx);
  };
  if (window._rev && window._rev.naturalW) apri();
  else document.getElementById('rev-img').addEventListener('load', () => setTimeout(apri, 50), {once: true});
}

function revIssueIndexes() {
  const dets = (window._rev && window._rev.data && window._rev.data.detections) || [];
  const out = [];
  dets.forEach((det, i) => {
    const hasText = (det.testo_tradotto || '').trim().length > 0;
    if ((det.ocr_empty_suspect && !hasText) || (det._overflow && hasText)) out.push(i);
  });
  return out;
}

// Apre il prossimo balloon problematico della pagina; se non ce ne sono piu',
// passa alla pagina successiva e riparte da li', cosi' si revisiona un volume
// intero premendo sempre lo stesso tasto invece di cercare a occhio.
function revNextIssue() {
  const idxs = revIssueIndexes();
  if (!idxs.length) {
    document.getElementById('rev-status').insertAdjacentHTML('beforeend',
      ' <span class="text-gray-500">— nessun problema in questa pagina</span>');
    revNav(1);
    return;
  }
  const cur = (window._revLastIssue === undefined) ? -1 : window._revLastIssue;
  const next = idxs.find(i => i > cur);
  window._revLastIssue = (next === undefined) ? idxs[0] : next;
  revOpenEditor(window._revLastIssue);
}

function revNav(delta) {
  window._revLastIssue = undefined;
  const sel = document.getElementById('rev-page');
  const idx = sel.selectedIndex + delta;
  if (idx >= 0 && idx < sel.options.length) {
    sel.selectedIndex = idx;
    revLoadPage();
  }
}

function revHandlePos(h) {
  const map = {nw:[0,0], n:[0.5,0], ne:[1,0], w:[0,0.5], e:[1,0.5], sw:[0,1], s:[0.5,1], se:[1,1]};
  return {x: map[h][0], y: map[h][1]};
}

function revRenderBoxes() {
  const rev = window._rev;
  const img = document.getElementById('rev-img');
  const wrap = document.getElementById('rev-canvas-wrap');
  const layer = document.getElementById('rev-boxes');
  layer.innerHTML = '';
  if (!rev || !rev.data || !rev.naturalW || !img.clientWidth) return;

  const imgRect = img.getBoundingClientRect();
  const wrapRect = wrap.getBoundingClientRect();
  const scale = imgRect.width / rev.naturalW;
  rev.scale = scale;
  rev.offsetX = imgRect.left - wrapRect.left;
  rev.offsetY = imgRect.top - wrapRect.top;

  const cursorMap = {nw:'nwse-resize', n:'ns-resize', ne:'nesw-resize', w:'ew-resize',
                      e:'ew-resize', sw:'nesw-resize', s:'ns-resize', se:'nwse-resize'};

  (rev.data.detections || []).forEach((det, idx) => {
    if (!det.bbox) return;
    const [x1, y1, x2, y2] = det.bbox;
    const left = rev.offsetX + x1 * scale;
    const top  = rev.offsetY + y1 * scale;
    const w = (x2 - x1) * scale;
    const h = (y2 - y1) * scale;
    const hasText = (det.testo_tradotto || '').trim().length > 0;
    // ocr_empty_suspect: ocr.py ha trovato testo nel ritaglio ma il modello
    // non ha letto nulla (vedi ocr.py::_looks_like_text). Senza evidenziarlo
    // qui, il balloon resta in lingua originale fino alla tavola finita: e'
    // rosso, non grigio come un balloon vuoto qualsiasi, perche' va
    // trascritto a mano. Appena gli si da' un testo torna verde.
    const needsTranscription = !!det.ocr_empty_suspect && !hasText;
    // _overflow: render.py non e' riuscito a far stare il testo nel balloon e
    // lo ha TRONCATO. E' perdita di contenuto, va accorciata la traduzione (o
    // allargato il box): arancione, distinto dal rosso "manca proprio il
    // testo". Il flag viene ricalcolato ad ogni render, quindi si spegne da
    // solo quando il problema e' risolto.
    const isOverflow = !!det._overflow && hasText;
    const color = det._modified ? '#ffc107'
                : needsTranscription ? '#ff1744'
                : isOverflow ? '#ff9100'
                : (hasText ? '#00c853' : '#888888');

    const box = document.createElement('div');
    box.style.cssText = `position:absolute;left:${left}px;top:${top}px;width:${w}px;height:${h}px;` +
      `border:2px solid ${color};cursor:move;box-sizing:border-box;pointer-events:auto;touch-action:none;`;
    box.addEventListener('pointerdown', (e) => revBoxMouseDown(e, idx, 'move'));
    box.addEventListener('dblclick', (e) => { e.stopPropagation(); revOpenEditor(idx); });

    // Handle di resize: dimensione proporzionale al box visualizzato, non fissa,
    // altrimenti su balloon piccoli/schermi mobili gli 8 handle si sovrappongono
    // tra loro riempiendo l'interno del box e coprendo il testo.
    const handleSize = Math.max(4, Math.min(9, Math.min(w, h) / 4));
    const showEdgeHandles = Math.min(w, h) >= handleSize * 4;
    const handleNames = showEdgeHandles ? Object.keys(cursorMap) : ['nw', 'ne', 'sw', 'se'];
    handleNames.forEach(hName => {
      const pos = revHandlePos(hName);
      const handle = document.createElement('div');
      handle.style.cssText = `position:absolute;width:${handleSize}px;height:${handleSize}px;background:${color};` +
        `border:1px solid white;left:${pos.x * 100}%;top:${pos.y * 100}%;` +
        `transform:translate(-50%,-50%);cursor:${cursorMap[hName]};pointer-events:auto;touch-action:none;`;
      handle.addEventListener('pointerdown', (e) => { e.stopPropagation(); revBoxMouseDown(e, idx, 'resize', hName); });
      box.appendChild(handle);
    });
    if (needsTranscription) {
      const badge = document.createElement('div');
      badge.textContent = '⚠';
      badge.title = "OCR vuoto ma nel balloon c'e' del testo: da trascrivere a mano";
      badge.style.cssText = `position:absolute;left:-9px;top:-11px;font-size:16px;line-height:1;` +
        `color:#ff1744;pointer-events:none;text-shadow:0 0 3px white,0 0 3px white,0 0 3px white;`;
      box.appendChild(badge);
    }
    if (isOverflow) {
      const badge = document.createElement('div');
      badge.textContent = '✂';
      badge.title = 'Testo troppo lungo: e\' stato troncato nel render. Accorcia la traduzione o allarga il box.';
      badge.style.cssText = `position:absolute;right:-9px;top:-11px;font-size:15px;line-height:1;` +
        `color:#ff9100;pointer-events:none;text-shadow:0 0 3px white,0 0 3px white,0 0 3px white;`;
      box.appendChild(badge);
    }
    if (det.manual_balloon_shape === 'ellipse') {
      // Balloon ricreato da zero (pulizia automatica che ha cancellato
      // l'intera sagoma): icona ben visibile, non un piccolo quadratino
      // colore, cosi' si individua a colpo d'occhio tra tanti balloon.
      const badge = document.createElement('div');
      badge.textContent = '🔁';
      badge.title = 'Balloon ricreato manualmente';
      badge.style.cssText = `position:absolute;left:-11px;top:-11px;font-size:16px;line-height:1;` +
        `pointer-events:none;text-shadow:0 0 3px white,0 0 3px white,0 0 3px white;`;
      box.appendChild(badge);
    } else if (det.manual_fill_color || det.manual_border) {
      const swatch = document.createElement('div');
      swatch.style.cssText = `position:absolute;left:-3px;top:-3px;width:10px;height:10px;` +
        `background:${det.manual_fill_color || 'transparent'};` +
        `border:2px solid ${det.manual_border ? '#000' : '#fff'};pointer-events:none;`;
      box.appendChild(swatch);
    }
    layer.appendChild(box);
  });
}

function revBoxMouseDown(e, idx, type, handle) {
  e.preventDefault();
  const det = window._rev.data.detections[idx];
  _revDrag = { idx, type, handle, startX: e.clientX, startY: e.clientY, origBbox: det.bbox.slice() };
  e.target.setPointerCapture(e.pointerId);
  document.addEventListener('pointermove', revMouseMove);
  document.addEventListener('pointerup', revMouseUp);
}

function revMouseMove(e) {
  if (!_revDrag) return;
  const rev = window._rev;
  const scale = rev.scale || 1;
  const dx = (e.clientX - _revDrag.startX) / scale;
  const dy = (e.clientY - _revDrag.startY) / scale;
  const orig = _revDrag.origBbox;
  const MIN = 15;
  let [x1, y1, x2, y2] = orig;

  if (_revDrag.type === 'move') {
    x1 = orig[0] + dx; y1 = orig[1] + dy; x2 = orig[2] + dx; y2 = orig[3] + dy;
  } else {
    const h = _revDrag.handle;
    if (h.includes('n')) y1 = Math.min(orig[1] + dy, orig[3] - MIN);
    if (h.includes('s')) y2 = Math.max(orig[3] + dy, orig[1] + MIN);
    if (h.includes('w')) x1 = Math.min(orig[0] + dx, orig[2] - MIN);
    if (h.includes('e')) x2 = Math.max(orig[2] + dx, orig[0] + MIN);
  }
  rev.data.detections[_revDrag.idx].bbox = [x1, y1, x2, y2];
  revRenderBoxes();
  revSchedulePreview(_revDrag.idx, false);
}

function revMouseUp(e) {
  if (!_revDrag) return;
  const rev = window._rev;
  const det = rev.data.detections[_revDrag.idx];
  const orig = _revDrag.origBbox;
  const W = rev.naturalW, H = rev.naturalH;
  let [x1, y1, x2, y2] = det.bbox;

  // Un click semplice (nessun trascinamento reale, es. il primo click di un
  // doppio click) non deve contare come modifica: se lo trattassimo come tale,
  // ricostruiremmo subito i div dei box (revRenderBoxes distrugge e ricrea
  // tutto), sostituendo l'elemento appena cliccato prima che il browser
  // riesca a riconoscere il secondo click come "dblclick" — di fatto
  // impedendo per sempre l'apertura dell'editor con doppio click.
  const moved = Math.round(x1) !== orig[0] || Math.round(y1) !== orig[1] ||
                Math.round(x2) !== orig[2] || Math.round(y2) !== orig[3];
  const idx = _revDrag.idx;
  _revDrag = null;
  document.removeEventListener('pointermove', revMouseMove);
  document.removeEventListener('pointerup', revMouseUp);
  if (!moved) {
    // Il dblclick nativo del browser nasce dai click sintetici generati dal
    // touch, ma revBoxMouseDown chiama preventDefault() sul pointerdown per
    // poter trascinare i box: questo sopprime anche quei click sintetici, e
    // su telefono il doppio tap non apre mai l'editor. Rileviamo quindi il
    // doppio tap a mano confrontando istante/posizione con l'ultimo tap.
    const now = Date.now();
    const last = _revLastTap;
    _revLastTap = { idx, t: now, x: e.clientX, y: e.clientY };
    if (last && last.idx === idx && now - last.t < 400 &&
        Math.abs(e.clientX - last.x) < 20 && Math.abs(e.clientY - last.y) < 20) {
      _revLastTap = null;
      revOpenEditor(idx);
    }
    return;
  }

  x1 = Math.max(0, Math.min(Math.round(x1), W - 1));
  y1 = Math.max(0, Math.min(Math.round(y1), H - 1));
  x2 = Math.max(x1 + 1, Math.min(Math.round(x2), W));
  y2 = Math.max(y1 + 1, Math.min(Math.round(y2), H));
  det.bbox = [x1, y1, x2, y2];
  det._modified = true;
  det._bbox_modified = true;
  det.manual_text_box = true;
  revMarkDirty();
  revRenderBoxes();
  revSchedulePreview(idx, true);
}

function revOpenEditor(idx) {
  revCloseEditor();
  const rev = window._rev;
  const det = rev.data.detections[idx];
  const wrap = document.getElementById('rev-canvas-wrap');
  const left0 = rev.offsetX + det.bbox[0] * rev.scale;
  const top0  = rev.offsetY + det.bbox[1] * rev.scale;
  // Dimensione minima generosa e indipendente da quella del box sul balloon:
  // sotto la textarea ci sono altre 6-7 righe di controlli (font, colori,
  // forma, codino, bottoni), quindi un box piccolo lasciava alla textarea
  // solo una manciata di pixel, a malapena una riga di testo visibile.
  const minW = Math.min(320, wrap.clientWidth - 20);
  const minH = Math.min(460, wrap.clientHeight - 20);
  const w = Math.max((det.bbox[2] - det.bbox[0]) * rev.scale, minW);
  const h = Math.max((det.bbox[3] - det.bbox[1]) * rev.scale, minH);
  const left = Math.max(0, Math.min(left0, wrap.clientWidth - w));
  const top  = Math.max(0, Math.min(top0, wrap.clientHeight - h));

  const fonts = window._revFonts || {};
  const fontOptions = Object.keys(fonts).map(name =>
    `<option value="${escHtml(name)}" ${fonts[name] === det.font_path ? 'selected' : ''}>${escHtml(name)}</option>`
  ).join('');

  const div = document.createElement('div');
  div.id = 'rev-editor';
  div.style.cssText = `position:absolute;left:${left}px;top:${top}px;width:${w}px;height:${h}px;` +
    `background:white;color:#111;border:2px solid #333;border-radius:4px;` +
    `display:flex;flex-direction:column;z-index:50;pointer-events:auto;`;
  const overflowNotice = det._overflow
    ? `<div class="bg-orange-100 text-orange-800 text-xs px-2 py-1">✂ Nel render questo testo
         non ci stava ed e' stato troncato: accorcialo, oppure allarga il box o
         imposta una dimensione font piu' piccola qui sotto.</div>`
    : '';
  const suspectNotice = det.ocr_empty_suspect
    ? `<div class="bg-red-100 text-red-700 text-xs px-2 py-1">⚠ L'OCR non ha letto nulla,
         ma nel ritaglio c'e' del testo: trascrivilo qui (o scrivi <b>-</b> se e'
         un'insegna/onomatopea da lasciare com'e').</div>`
    : '';
  div.innerHTML = `
    ${suspectNotice}
    ${overflowNotice}
    <div class="bg-gray-200 text-gray-700 text-xs px-2 py-1"><b>Originale:</b>
      <span class="italic">${escHtml(det.testo_originale || '(vuoto)')}</span></div>
    <textarea id="rev-editor-text" class="flex-1 w-full p-2 text-sm"
      style="resize:none;border:none;outline:none;min-height:120px;">${escHtml(det.testo_tradotto || '')}</textarea>
    <div class="flex items-center gap-1 px-2 py-1 bg-gray-100 text-xs">
      <b>Font${det.font_auto ? ' (auto)' : ''}:</b>
      <select id="rev-editor-font" class="flex-1 border rounded px-1 py-0.5 text-black">${fontOptions}</select>
    </div>
    <div class="flex items-center gap-2 px-2 py-1 bg-gray-100 text-xs flex-wrap">
      <label class="flex items-center gap-1">
        <input type="checkbox" id="rev-editor-fill-enable" ${det.manual_fill_color ? 'checked' : ''}>
        Riempi box
      </label>
      <input type="color" id="rev-editor-fill-color"
        value="${det.manual_fill_color || '#ffffff'}" class="w-6 h-6 p-0 border-0 rounded">
      <label class="flex items-center gap-1 ml-2">
        <input type="checkbox" id="rev-editor-border" ${det.manual_border ? 'checked' : ''}>
        Bordo nero
      </label>
      <button onclick="revRecreateBalloon()" title="Reimposta riempimento bianco + bordo nero: usa questo se la pulizia automatica ha cancellato l'intero balloon, non solo il testo"
        class="ml-auto bg-blue-600 hover:bg-blue-500 text-white text-xs px-2 py-1 rounded">🔁 Ricrea balloon</button>
    </div>
    <div class="flex items-center gap-2 px-2 py-1 bg-gray-100 text-xs flex-wrap">
      <label class="flex items-center gap-1">
        <input type="checkbox" id="rev-editor-text-color-enable" ${det.manual_text_color ? 'checked' : ''}>
        Colore testo manuale
      </label>
      <input type="color" id="rev-editor-text-color"
        value="${det.manual_text_color || '#000000'}" class="w-6 h-6 p-0 border-0 rounded">
      <span class="text-gray-500">(altrimenti auto nero/bianco)</span>
    </div>
    <div class="flex items-center gap-1 px-2 py-1 bg-gray-100 text-xs">
      <label class="flex items-center gap-1">
        <input type="checkbox" id="rev-editor-recreate" ${det.manual_balloon_shape === 'ellipse' ? 'checked' : ''}>
        Forma a ellisse (testo nel rettangolo inscritto, non nel bbox intero)
      </label>
    </div>
    <div class="flex items-center gap-2 px-2 py-1 bg-gray-100 text-xs">
      <b>Dim.:</b>
      <input type="number" id="rev-editor-font-size" min="1" style="width:3.5em"
        class="border rounded px-1 py-0.5 text-black" value="${det.manual_font_size || ''}">
      <b class="ml-2">Interlinea:</b>
      <input type="number" id="rev-editor-line-spacing" min="0" step="0.1" style="width:3.5em"
        class="border rounded px-1 py-0.5 text-black" value="${det.manual_line_spacing || ''}">
      <span class="text-gray-500">(vuoto = automatico)</span>
    </div>
    <div class="flex items-center gap-1 px-2 py-1 bg-gray-100 text-xs">
      <b>Codino:</b>
      <select id="rev-editor-tail" class="flex-1 border rounded px-1 py-0.5 text-black">
        <option value="">Nessuno</option>
        <option value="n">Su</option>
        <option value="ne">Su-destra</option>
        <option value="e">Destra</option>
        <option value="se">Giù-destra</option>
        <option value="s">Giù</option>
        <option value="sw">Giù-sinistra</option>
        <option value="w">Sinistra</option>
        <option value="nw">Su-sinistra</option>
      </select>
    </div>
    <div class="flex gap-1 px-2 py-1 bg-gray-100">
      <button onclick="revConfirmEditor(${idx})" class="bg-indigo-600 text-white text-xs px-2 py-1 rounded">✓ OK</button>
      <button onclick="revCloseEditor()" class="bg-gray-400 text-white text-xs px-2 py-1 rounded">✗ Annulla</button>
    </div>`;
  wrap.appendChild(div);
  document.getElementById('rev-editor-tail').value = det.manual_balloon_tail || '';
  const ta = document.getElementById('rev-editor-text');
  ta.focus();
  ta.select();

  _revEditorIdx = idx;
  ['rev-editor-text', 'rev-editor-font', 'rev-editor-fill-enable', 'rev-editor-fill-color',
   'rev-editor-border', 'rev-editor-text-color-enable', 'rev-editor-text-color',
   'rev-editor-recreate', 'rev-editor-tail', 'rev-editor-font-size', 'rev-editor-line-spacing'].forEach(id => {
    document.getElementById(id).addEventListener('input', () => revSchedulePreview(idx, false));
  });
  revSchedulePreview(idx, true);
}

function revConfirmEditor(idx) {
  const rev = window._rev;
  const det = rev.data.detections[idx];
  const text = document.getElementById('rev-editor-text').value.trim();
  const fontName = document.getElementById('rev-editor-font').value;
  if (text === '-') {
    det.testo_originale = '';
    det.testo_tradotto = '';
  } else {
    det.testo_tradotto = text;
  }
  // L'utente ha guardato il balloon e deciso (trascritto o marcato come da
  // lasciare): la segnalazione automatica ha esaurito il suo compito e non
  // deve piu' ricomparire ad ogni ricaricamento della pagina.
  if (det.ocr_empty_suspect) det.ocr_empty_suspect = false;
  const fonts = window._revFonts || {};
  if (fontName && fonts[fontName] && fonts[fontName] !== det.font_path) {
    det.font_path = fonts[fontName];
    det.font_auto = false;
  }
  const fillEnable = document.getElementById('rev-editor-fill-enable').checked;
  det.manual_fill_color = fillEnable ? document.getElementById('rev-editor-fill-color').value : null;
  det.manual_border = document.getElementById('rev-editor-border').checked;
  det.manual_balloon_shape = document.getElementById('rev-editor-recreate').checked ? 'ellipse' : null;
  det.manual_balloon_tail = document.getElementById('rev-editor-tail').value || null;
  const textColorEnable = document.getElementById('rev-editor-text-color-enable').checked;
  det.manual_text_color = textColorEnable ? document.getElementById('rev-editor-text-color').value : null;
  const fontSize = parseInt(document.getElementById('rev-editor-font-size').value, 10);
  det.manual_font_size = Number.isFinite(fontSize) && fontSize > 0 ? fontSize : null;
  const lineSpacing = parseFloat(document.getElementById('rev-editor-line-spacing').value);
  det.manual_line_spacing = Number.isFinite(lineSpacing) && lineSpacing > 0 ? lineSpacing : null;
  det._modified = true;
  revCloseEditor();
  revMarkDirty();
  revRenderBoxes();
}

function revRecreateBalloon() {
  // Preimposta i controlli di riempimento/bordo con i valori tipici di un
  // balloon (bianco pieno + contorno nero), da confermare con OK: usato
  // quando la pulizia automatica ha cancellato l'intera sagoma del
  // balloon (non solo il testo) e va ridisegnata a mano. Marca anche la
  // richiesta esplicita di forma a ellisse: senza, un box che l'utente ha
  // gia' ridimensionato a mano (per farlo combaciare con l'area del
  // balloon sparito) verrebbe riempito come rettangolo esatto invece che
  // come balloon ovale.
  document.getElementById('rev-editor-fill-enable').checked = true;
  document.getElementById('rev-editor-fill-color').value = '#ffffff';
  document.getElementById('rev-editor-border').checked = true;
  document.getElementById('rev-editor-recreate').checked = true;
  const tailSel = document.getElementById('rev-editor-tail');
  if (!tailSel.value) tailSel.value = 's';
  if (_revEditorIdx !== null) revSchedulePreview(_revEditorIdx, false);
}

function revCloseEditor() {
  const el = document.getElementById('rev-editor');
  if (el) el.remove();
  _revEditorIdx = null;
  revClearPreview();
}

// Costruisce il balloon da inviare all'anteprima: se il suo editor e' aperto,
// usa i valori correnti dei campi (non ancora confermati con OK), altrimenti
// il balloon cosi' com'e' salvato in memoria (es. durante il trascinamento).
function revEditorLiveDet(idx) {
  const det = window._rev.data.detections[idx];
  // Durante un trascinamento (sposta/ridimensiona) il bbox in rev.data e'
  // gia' stato aggiornato alla posizione live (vedi revMouseMove), ma
  // manual_text_box diventa True solo al rilascio del mouse (revMouseUp).
  // Nel frattempo, se il balloon ha una forma reale (balloon_source ===
  // "yolov8seg"), il fitting userebbe ancora quella maschera per decidere
  // dove e come entra il testo: mask_path e' pero' un'immagine posizionata
  // in assoluto sulla pagina intera, ferma alla posizione DI PRIMA del
  // trascinamento, non relativa al bbox. Il risultato e' che l'anteprima
  // durante il drag disegna balloon e testo alla vecchia posizione (spesso
  // sovrapposti al disegno sottostante) invece di seguire il trascinamento.
  // Forzare qui lo stesso manual_text_box che revMouseUp committera' comunque
  // al rilascio rende l'anteprima coerente con l'esito finale fin da subito.
  const dragging = _revDrag && _revDrag.idx === idx;
  if (_revEditorIdx !== idx || !document.getElementById('rev-editor')) {
    return dragging ? Object.assign({}, det, {manual_text_box: true}) : det;
  }
  const text = document.getElementById('rev-editor-text').value.trim();
  const fontName = document.getElementById('rev-editor-font').value;
  const fonts = window._revFonts || {};
  const fillEnable = document.getElementById('rev-editor-fill-enable').checked;
  const textColorEnable = document.getElementById('rev-editor-text-color-enable').checked;
  return Object.assign({}, det, {
    testo_tradotto: text === '-' ? '' : text,
    font_path: (fontName && fonts[fontName]) ? fonts[fontName] : det.font_path,
    manual_fill_color: fillEnable ? document.getElementById('rev-editor-fill-color').value : null,
    manual_border: document.getElementById('rev-editor-border').checked,
    manual_balloon_shape: document.getElementById('rev-editor-recreate').checked ? 'ellipse' : null,
    manual_balloon_tail: document.getElementById('rev-editor-tail').value || null,
    manual_text_color: textColorEnable ? document.getElementById('rev-editor-text-color').value : null,
    manual_font_size: (() => {
      const v = parseInt(document.getElementById('rev-editor-font-size').value, 10);
      return Number.isFinite(v) && v > 0 ? v : null;
    })(),
    manual_line_spacing: (() => {
      const v = parseFloat(document.getElementById('rev-editor-line-spacing').value);
      return Number.isFinite(v) && v > 0 ? v : null;
    })(),
    manual_text_box: dragging ? true : det.manual_text_box,
  });
}

function revSchedulePreview(idx, immediate) {
  if (_revPreviewTimer) clearTimeout(_revPreviewTimer);
  _revPreviewTimer = setTimeout(() => revFetchPreview(idx), immediate ? 0 : 220);
}

async function revFetchPreview(idx) {
  const rev = window._rev;
  if (!rev || !rev.data) return;
  const det = revEditorLiveDet(idx);
  if (!det.bbox || !(det.testo_tradotto || '').trim()) { revClearPreview(); return; }
  const seq = ++_revPreviewSeq;
  try {
    const r = await fetch('/api/review-preview-balloon', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({comic: rev.comic, page: rev.page, det}),
    });
    const d = await r.json();
    if (seq !== _revPreviewSeq) return; // arrivata dopo una richiesta piu' recente, scartala
    if (d.error || !d.image) { revClearPreview(); return; }
    revShowPreview(d.image, d.bbox);
  } catch (e) {
    revClearPreview();
  }
}

function revShowPreview(dataUrl, bbox) {
  const rev = window._rev;
  const layer = document.getElementById('rev-boxes');
  let img = document.getElementById('rev-preview-img');
  if (!img) {
    img = document.createElement('img');
    img.id = 'rev-preview-img';
    img.style.cssText = 'position:absolute;pointer-events:none;z-index:40;';
    layer.appendChild(img);
  }
  const [x1, y1, x2, y2] = bbox;
  img.style.left = (rev.offsetX + x1 * rev.scale) + 'px';
  img.style.top = (rev.offsetY + y1 * rev.scale) + 'px';
  img.style.width = ((x2 - x1) * rev.scale) + 'px';
  img.style.height = ((y2 - y1) * rev.scale) + 'px';
  img.src = dataUrl;
}

function revClearPreview() {
  if (_revPreviewTimer) { clearTimeout(_revPreviewTimer); _revPreviewTimer = null; }
  _revPreviewSeq++;
  const img = document.getElementById('rev-preview-img');
  if (img) img.remove();
}

function revMarkDirty() {
  window._rev.dirty = true;
  document.getElementById('rev-regen-btn').disabled = false;
  document.getElementById('rev-status').innerHTML =
    '<span class="text-yellow-400">⚠ Modifiche in sospeso. Clicca "Rigenera pagina" per applicarle.</span>';
}

async function revSaveAndRegenerate() {
  const rev = window._rev;
  if (!rev) return;
  document.getElementById('rev-regen-btn').disabled = true;
  document.getElementById('rev-status').innerHTML = '<span class="text-yellow-400">⏳ Rigenerazione in corso...</span>';
  const r = await fetch('/api/review-regenerate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({comic: rev.comic, page: rev.page, data: rev.data})
  });
  const res = await r.json();
  if (res.error) {
    document.getElementById('rev-status').innerHTML = `<span class="text-red-400">${escHtml(res.error)}</span>`;
    document.getElementById('rev-regen-btn').disabled = false;
    return;
  }
  revPollRegenerate();
}

async function revPollRegenerate() {
  const r = await fetch('/api/review-regenerate-status');
  const s = await r.json();
  if (s.running) {
    setTimeout(revPollRegenerate, 700);
    return;
  }
  if (s.error) {
    document.getElementById('rev-status').innerHTML = `<span class="text-red-400">Errore: ${escHtml(s.error)}</span>`;
    document.getElementById('rev-regen-btn').disabled = false;
    return;
  }
  window._rev.dirty = false;
  document.getElementById('rev-status').innerHTML =
    `<span class="text-green-400">✅ ${escHtml(s.message || 'Pagina rigenerata')}</span>`;
  revLoadPage();
}

// Init edit pages on tab open
loadEditPages();
revLoadPages();
</script>
</body>
</html>
"""


# ── Progress tracking ─────────────────────────────────────────────────────────

_progress = {"pct": 0, "stage": "", "detail": "", "running": False}

def _update_progress(line, stages_total, stages_done, pages_total, pages_done):
    if stages_total > 0:
        stage_pct = (stages_done / stages_total) * 100
        page_pct  = (pages_done  / pages_total  * 100 / stages_total) if pages_total > 0 else 0
        _progress["pct"] = min(stage_pct + page_pct, 100)
    m = re.search(r"===\s*(OCR|Traduzione|Pulizia|Render|Pretrattamento):\s*(.+?)\s*===", line)
    if m:
        _progress["stage"] = m.group(1)
        _progress["detail"] = m.group(2)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    try:
        cfg = load_cfg()
        i18n.language_from_config(cfg)
        comics = collect_comics(cfg)
        ls_cfg = cfg.get("llama_server", {})
        ocr_model_path = ls_cfg.get("ocr_model_path", "")
        paddleocr_model_path = ls_cfg.get("paddleocr_model_path", "")
        ocr_mmproj_path = ls_cfg.get("ocr_mmproj_path", "")
        translate_model_path = ls_cfg.get("translate_model_path", "")
        llama_port = ls_cfg.get("port", 8081)
        ocr_backend = cfg.get("ocr_backend", "qwen")
        fixed_font_size = cfg.get("rendering", {}).get("fixed_font_size")
        fixed_font_size = fixed_font_size if fixed_font_size else ""
    except Exception:
        comics = []
        ocr_model_path = paddleocr_model_path = ocr_mmproj_path = translate_model_path = ""
        llama_port = 8081
        ocr_backend = "qwen"
        fixed_font_size = ""

    try:
        pretreat_prompt = cfg.get("flux_pretreat", {}).get("last_prompt", "")
    except Exception:
        pretreat_prompt = ""

    tabs = [
        ("pipeline",  "Pipeline"),
        ("config",    "Config"),
        ("edit",      "Correzioni"),
        ("review",    "Revisione"),
        ("cache",     "Cache"),
    ]
    stages = [
        ("full",           "Pipeline COMPLETA"),
        ("ocr",            "Solo OCR"),
        ("auto_translate", "Solo Traduzione"),
        ("clean",          "Solo Pulizia"),
        ("render",         "Solo Render (testo)"),
    ]
    # Le etichette sono letterali italiani: il template si traduce una volta
    # per lingua (risultato in cache dentro i18n), tab e stage a parte perche'
    # vivono in liste Python, non nell'HTML.
    return render_template_string(
        i18n.translate_html(HTML),
        tabs=[(tid, t(lab)) for tid, lab in tabs],
        tab_ids=[tid for tid, _ in tabs],
        stages=[(sid, t(lab)) for sid, lab in stages],
        lingue=i18n.LINGUE,
        lingua_attuale=i18n.get_language(),
        comics=comics,
        ocr_model_path=ocr_model_path,
        paddleocr_model_path=paddleocr_model_path,
        ocr_mmproj_path=ocr_mmproj_path,
        translate_model_path=translate_model_path,
        llama_port=llama_port,
        ocr_backend=ocr_backend,
        pretreat_prompt=pretreat_prompt,
        fixed_font_size=fixed_font_size,
    )


@app.route("/api/status")
def api_status():
    try:
        cfg = load_cfg()
        comics = collect_comics(cfg)
        _, lm_txt = llama_server_state(cfg)
        return f"""
          <span class="text-indigo-300">Config: OK</span> |
          Fumetti: <b>{len(comics)}</b> |
          llama-server: <b>{lm_txt}</b>
        """
    except Exception as e:
        return f'<span class="text-red-400">Errore: {e}</span>'


@app.route("/api/progress")
def api_progress():
    return jsonify({**_progress, "running": _running})


@app.route("/api/log-stream")
def api_log_stream():
    def gen():
        sent = 0
        while True:
            with _log_lock:
                if sent < len(_log_buffer):
                    for line in _log_buffer[sent:]:
                        yield f"data: {line}\n\n"
                    sent = len(_log_buffer)
            time.sleep(0.3)
    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/run", methods=["POST"])
def api_run():
    global _running, _process
    if _running:
        return jsonify({"error": t("Pipeline già in esecuzione")})

    body  = request.get_json()
    stage = body.get("stage", "full")
    comic = body.get("comic", "") or None
    limit = body.get("limit", "").strip()
    force = bool(body.get("force", False))
    ocr_backend = body.get("ocr_backend") or None
    prompt = body.get("prompt") or None

    if stage == "pretreat":
        if not prompt or not prompt.strip():
            return jsonify({"error": t("Prompt pretrattamento vuoto")})
        prompt = prompt.strip()
        try:
            cfg = load_cfg()
            cfg.setdefault("flux_pretreat", {})["last_prompt"] = prompt
            save_cfg(cfg)
        except Exception:
            pass  # non bloccare l'avvio solo perche' il salvataggio del prompt e' fallito

    start = end = None
    if limit:
        if "-" in limit:
            parts = limit.split("-")
            try:
                start = int(parts[0]) if parts[0] else 0
                end   = int(parts[1]) if parts[1] else None
            except ValueError:
                return jsonify({"error": t("Formato intervallo non valido")})
        else:
            try:
                end = int(limit); start = 0
            except ValueError:
                return jsonify({"error": t("Numero non valido")})

    if stage == "full":
        stages = ["ocr", "translate_lm", "clean", "render"]
    elif stage == "auto_translate":
        stages = ["translate_lm"]
    else:
        stages = [stage]

    _progress.update({"pct": 0, "stage": "", "detail": "", "running": True})

    def runner():
        global _running, _process
        _running = True
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        pages_total = 0
        try:
            cfg = load_cfg()
            pages_total = count_pages(cfg, comic) if comic is not None else \
                          sum(count_pages(cfg, c) for c in (collect_comics(cfg) or [""]))
        except Exception:
            pass

        try:
            cfg2 = load_cfg()
            all_comics = [comic] if comic else (collect_comics(cfg2) or [""])
        except Exception:
            all_comics = [comic or ""]

        total_steps = len(stages) * len(all_comics)
        step = 0
        outer_break = False

        for cidx, cur_comic in enumerate(all_comics):
            if outer_break:
                break
            label = cur_comic or "(root)"
            push_log(f"📚 Fumetto {cidx+1}/{len(all_comics)}: {label}")
            for i, s in enumerate(stages):
                step += 1
                push_log(f"▶ [{label}] Stage {i+1}/{len(stages)}: {s}")
                _progress["stage"] = f"{label} › {s}"
                cmd = build_command(s, comic=cur_comic if cur_comic else None, start=start, end=end, force=force, ocr_backend=ocr_backend, prompt=prompt)
                try:
                    _process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace", env=env)
                    pages_done = [0]
                    for line in _process.stdout:
                        line = line.rstrip()
                        push_log(line)
                        if re.search(r"===\s*(OCR|Traduzione|Pulizia|Render|Pretrattamento):", line):
                            pages_done[0] += 1
                        _update_progress(line, total_steps, step - 1, pages_total, pages_done[0])
                    _process.wait()
                    if _process.returncode != 0:
                        push_log(f"❌ [{label}] Stage {s} fallito (codice {_process.returncode}), passo al fumetto successivo")
                        outer_break = True
                        break
                    push_log(f"✅ [{label}] Stage {s} completato")
                except Exception as e:
                    push_log(f"❌ Errore: {e}")
                    outer_break = True
                    break

        _running = False
        _progress.update({"pct": 100, "running": False, "detail": ""})
        push_log("🏁 Pipeline terminata")

    threading.Thread(target=runner, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _running, _process
    if _process and _process.poll() is None:
        _process.terminate()
        push_log("⏹ Pipeline fermata dall'utente")
    _running = False
    _progress["running"] = False
    return jsonify({"ok": True})


# ── Config endpoints ───────────────────────────────────────────────────────────

@app.route("/api/save-lm-models", methods=["POST"])
def api_save_lm_models():
    cfg = load_cfg()
    ls = cfg.setdefault("llama_server", {})
    for chiave in ("ocr_model_path", "paddleocr_model_path",
                   "ocr_mmproj_path", "translate_model_path"):
        ls[chiave] = request.form.get(chiave, "").strip()
    porta = request.form.get("llama_port", "").strip()
    if porta:
        try:
            ls["port"] = int(porta)
        except ValueError:
            return f'<span class="text-red-400">{t("Porta non valida")}</span>'
    cfg.setdefault("local_llm", {})["base_url"] = f"http://127.0.0.1:{ls.get('port', 8081)}/v1"
    save_cfg(cfg)
    mancanti = [e for e, path, ok in llama_model_rows(cfg) if path and not ok]
    if mancanti:
        return ('<span class="text-yellow-400">Salvato, ma file non trovati: '
                + ", ".join(mancanti) + '</span>')
    return f'<span class="text-green-400">{t("Modelli salvati")}</span>'

@app.route("/api/save-language", methods=["POST"])
def api_save_language():
    codice = request.form.get("ui_language", i18n.LINGUA_DEFAULT)
    if codice not in i18n.LINGUE:
        return f'<span class="text-red-400">{t("Lingua non valida")}</span>'
    cfg = load_cfg()
    cfg.setdefault("ui", {})["language"] = codice
    save_cfg(cfg)
    i18n.set_language(codice)
    return f'<span class="text-green-400">{t("Lingua salvata: riavvia la pagina per applicarla.")}</span>'


@app.route("/api/save-fixed-font-size", methods=["POST"])
def api_save_fixed_font_size():
    raw = request.form.get("fixed_font_size", "").strip()
    cfg = load_cfg()
    if not raw:
        cfg.setdefault("rendering", {})["fixed_font_size"] = None
        save_cfg(cfg)
        return f'<span class="text-green-400">{t("Dimensione font fissa disattivata (automatico)")}</span>'
    try:
        size = int(raw)
        if size <= 0:
            raise ValueError
    except ValueError:
        return f'<span class="text-red-400">{t("Inserisci un numero intero positivo, o lascia vuoto")}</span>'
    cfg.setdefault("rendering", {})["fixed_font_size"] = size
    save_cfg(cfg)
    return f'<span class="text-green-400">Dimensione font fissa impostata: {size}px</span>'


# ── Edit endpoints ───────────────────────────────────────

@app.route("/api/edit-pages")
def api_edit_pages():
    comic = request.args.get("comic", "")
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        base = work_dir / comic if comic else work_dir
        pages = sorted(p.name for p in base.iterdir()
                       if p.is_dir() and (p / "translated.json").exists()) if base.exists() else []
    except Exception:
        pages = []
    return jsonify(pages)

@app.route("/api/edit-load")
def api_edit_load():
    comic = request.args.get("comic", "")
    page  = request.args.get("page", "")
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        path = (work_dir / comic / page / "translated.json") if comic else (work_dir / page / "translated.json")
        if not path.exists():
            return jsonify({"error": f"translated.json non trovato: {path}"})
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/edit-save", methods=["POST"])
def api_edit_save():
    body  = request.get_json()
    comic = body.get("comic", "")
    page  = body.get("page", "")
    data  = body.get("data", {})
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        path = (work_dir / comic / page / "translated.json") if comic else (work_dir / page / "translated.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        push_log(f"💾 Salvato: {path}")
        return jsonify({"message": f"Salvato: {path}"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/edit-export")
def api_edit_export():
    comic = request.args.get("comic", "")
    mode  = request.args.get("mode", "both")
    range_str = request.args.get("range", "").strip()
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        base = work_dir / comic if comic else work_dir
        if not base.exists():
            return jsonify({"error": t("Fumetto non valido")}), 400
        pages = sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "translated.json").exists())
        if not pages:
            return jsonify({"error": t("Nessuna pagina trovata")}), 400
        if range_str:
            if "-" in range_str:
                parts = range_str.split("-")
                start = (int(parts[0]) - 1) if parts[0] else 0
                end = int(parts[1]) if parts[1] else len(pages)
            else:
                n = int(range_str)
                start, end = n - 1, n
            start = max(0, start)
            selected = pages[start:end]
        else:
            selected = pages
        if not selected:
            return jsonify({"error": t("Nessuna pagina nell'intervallo indicato")}), 400
        want_orig = mode in ("both", "orig")
        want_trans = mode in ("both", "trans")
        src_lang, dst_lang = export_prompt.langs_from_cfg(cfg)
        # Il prompt sta prima del primo marcatore === pagina ===, dove
        # l'import lo ignora: il file resta reimportabile com'e'.
        lines = [export_prompt.build(mode, comic, base, src_lang, dst_lang)]
        for page in selected:
            path = base / page / "translated.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            lines.append(f"=== {page} ===")
            for det in data.get("detections", []):
                bid = det.get("balloon_id", "?")
                orig = (det.get("testo_originale") or "").replace("\n", " ")
                trans = (det.get("testo_tradotto") or "").replace("\n", " ")
                if want_orig:
                    lines.append(f"[{bid}] ORIGINALE: {orig}")
                if want_trans:
                    lines.append(f"[{bid}] TRADOTTO: {trans}")
            lines.append("")
        suffix = {"both": "originale_tradotto", "orig": "originale", "trans": "tradotto"}[mode]
        filename = f"{comic or 'testo'}_{suffix}.txt"
        return Response(
            "\n".join(lines),
            mimetype="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError:
        return jsonify({"error": t("Formato intervallo non valido")}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

_BACKUP_KEEP = 20
"""Quante copie tenere per pagina: le revisioni sono iterative, un errore lo
si scopre dopo qualche giro, ma non serve conservare la storia completa."""


def _backup_translated(path, motivo: str):
    """Copia translated.json in _backups/ prima di sovrascriverlo.

    L'import da TXT e la rigenerazione da Revisione riscrivono il file in
    blocco: un file di import con i marcatori sbagliati, o un giro di
    modifiche partito da una pagina sbagliata, cancella lavoro di revisione
    senza modo di tornare indietro. La copia costa pochi millisecondi e
    qualche decina di KB.
    """
    try:
        if not path.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Un livello per fumetto/pagina: cosi' i backup restano navigabili e
        # la potatura piu' sotto guarda solo le copie di quella pagina.
        dest_dir = Path(paths.resolve("_backups")) / "revisione" / path.parent.parent.name / path.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"translated.{stamp}.{motivo}.json"
        shutil.copy2(path, dest)

        copie = sorted(dest_dir.glob("translated.*.json"))
        for vecchia in copie[:-_BACKUP_KEEP]:
            try:
                vecchia.unlink()
            except OSError:
                pass
        return dest
    except Exception as e:
        # Un backup fallito non deve impedire il salvataggio: si segnala e si
        # prosegue, altrimenti si perde anche la modifica appena fatta.
        push_log(f"Avviso: backup di {path} non riuscito ({e})")
        return None


@app.route("/api/edit-import", methods=["POST"])
def api_edit_import():
    comic = request.form.get("comic", "")
    file = request.files.get("file")
    if not file:
        return jsonify({"error": t("Nessun file caricato")})
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        base = work_dir / comic if comic else work_dir
        raw_text = file.read().decode("utf-8", errors="replace")
        page_re = re.compile(r"^===\s*(.+?)\s*===\s*$")
        line_re = re.compile(r"^\[(.+?)\]\s+(ORIGINALE|TRADOTTO):\s?(.*)$")
        updates = {}
        current_page = None
        for line in raw_text.splitlines():
            m = page_re.match(line)
            if m:
                current_page = m.group(1)
                continue
            m = line_re.match(line)
            if m and current_page:
                bid, kind, text = m.group(1), m.group(2), m.group(3)
                balloon_updates = updates.setdefault(current_page, {}).setdefault(bid, {})
                balloon_updates["testo_originale" if kind == "ORIGINALE" else "testo_tradotto"] = text
        if not updates:
            return jsonify({"error": t("Nessun dato riconosciuto nel file")})
        import translate_local
        pages_touched = []
        balloons_done = 0
        cached = 0
        for page, balloon_updates in updates.items():
            if not _is_safe_component(page):
                continue
            path = base / page / "translated.json"
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _backup_translated(path, "import")
            changed = False
            for det in data.get("detections", []):
                bid = str(det.get("balloon_id", "?"))
                if bid not in balloon_updates:
                    continue
                upd = balloon_updates[bid]
                if "testo_originale" in upd and det.get("testo_originale", "") != upd["testo_originale"]:
                    det["testo_originale"] = upd["testo_originale"]
                    changed = True
                if "testo_tradotto" in upd and det.get("testo_tradotto", "") != upd["testo_tradotto"]:
                    det["testo_tradotto"] = upd["testo_tradotto"]
                    changed = True
                balloons_done += 1
                # La traduzione importata entra in cache: e' una scelta umana
                # e deve sopravvivere a un rilancio dello stage di traduzione,
                # che altrimenti la ritradurrebbe da zero. Si usa il testo
                # originale FINALE (l'import puo' aver corretto anche quello)
                # e la chiave e' la stessa dello stage, glossario incluso.
                if "testo_tradotto" in upd:
                    try:
                        if translate_local.remember(
                            path.parent, cfg,
                            det.get("testo_originale", ""), det.get("testo_tradotto", ""),
                        ):
                            cached += 1
                    except Exception as e:
                        push_log(f"Avviso: cache non aggiornata per balloon {bid} di {page} ({e})")
            if changed:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                pages_touched.append(page)
        push_log(f"📥 Importate correzioni da TXT: {len(pages_touched)} pagine, {balloons_done} balloon, {cached} in cache")
        return jsonify({"pages": len(pages_touched), "balloons": balloons_done,
                        "cached": cached, "pages_touched": pages_touched})
    except Exception as e:
        return jsonify({"error": str(e)})


# ── Review endpoints ───────────────────────────────────────────────────────────

def _is_safe_component(s):
    return isinstance(s, str) and "/" not in s and "\\" not in s and ".." not in s

@app.route("/api/audit")
def api_audit():
    comic = request.args.get("comic", "")
    if not _is_safe_component(comic):
        return jsonify({"error": t("Parametro non valido")})
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        output_dir = Path(cfg["paths"]["output_dir"])
        result = audit.audit_comic(
            work_dir / comic if comic else work_dir,
            output_dir / comic if comic else output_dir,
            cfg,
        )
        tot = result["totals"]
        push_log(
            f"[*] Controllo {comic or '(root)'}: {tot['non_tradotto']} non tradotti, "
            f"{tot['da_trascrivere']} da trascrivere, {tot['troncato']} troncati, "
            f"{tot['troppo_lungo']} troppo lunghi, {tot['doppione']} doppioni, "
            f"{tot['pagine_incomplete']} stage mancanti"
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/review-pages")
def api_review_pages():
    comic = request.args.get("comic", "")
    if not _is_safe_component(comic):
        return jsonify([])
    try:
        cfg = load_cfg()
        output_dir = Path(cfg["paths"]["output_dir"])
        base = output_dir / comic if comic else output_dir
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        pages = sorted(p.stem for p in base.iterdir() if p.suffix.lower() in exts) if base.exists() else []
    except Exception:
        pages = []
    return jsonify(pages)

@app.route("/api/review-load")
def api_review_load():
    comic = request.args.get("comic", "")
    page  = request.args.get("page", "")
    if not _is_safe_component(comic) or not _is_safe_component(page):
        return jsonify({"error": t("Parametro non valido")})
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        output_dir = Path(cfg["paths"]["output_dir"])
        translated_path = (work_dir / comic / page / "translated.json") if comic else (work_dir / page / "translated.json")
        if not translated_path.exists():
            return jsonify({"error": f"translated.json non trovato: {translated_path}"})
        with open(translated_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_dir = (output_dir / comic) if comic else output_dir
        if not any((page_dir / f"{page}{ext}").exists() for ext in (".png", ".webp", ".jpg", ".jpeg")):
            return jsonify({"error": f"Pagina renderizzata non trovata: {page_dir / page}"})

        return jsonify({
            "data": data,
            "image_url": f"/api/review-image?comic={comic}&page={page}&kind=rendered&t={int(time.time()*1000)}",
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/review-image")
def api_review_image():
    comic = request.args.get("comic", "")
    page  = request.args.get("page", "")
    kind  = request.args.get("kind", "rendered")
    if not _is_safe_component(comic) or not _is_safe_component(page):
        return ("Parametro non valido", 400)
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        output_dir = Path(cfg["paths"]["output_dir"])
        path = None
        if kind == "cleaned":
            cand = (work_dir / comic / page / "cleaned.png") if comic else (work_dir / page / "cleaned.png")
            if cand.exists():
                path = cand
        else:
            page_dir = (output_dir / comic) if comic else output_dir
            for ext in (".png", ".webp", ".jpg", ".jpeg"):
                cand = page_dir / f"{page}{ext}"
                if cand.exists():
                    path = cand
                    break
        if path is None:
            return ("Immagine non trovata", 404)
        return send_file(path)
    except Exception as e:
        return (str(e), 500)

@app.route("/api/review-preview-balloon", methods=["POST"])
def api_review_preview_balloon():
    # Anteprima live di UN balloon durante l'editing in Revisione: renderizza
    # solo quel balloon sopra cleaned.png (nessuna chiamata a ComfyUI, nessun
    # salvataggio su disco) e ritorna un ritaglio della pagina, cosi' il
    # client vede lo stesso identico rendering che otterrebbe con "Rigenera"
    # senza aspettarlo e senza doverlo riprodurre approssimativamente in JS.
    body = request.get_json(silent=True) or {}
    comic = body.get("comic", "")
    page  = body.get("page", "")
    det   = body.get("det", {})
    if not _is_safe_component(comic) or not _is_safe_component(page):
        return jsonify({"error": t("Parametro non valido")}), 400
    if not det.get("bbox"):
        return jsonify({"error": t("bbox mancante")}), 400
    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        cleaned_path = (work_dir / comic / page / "cleaned.png") if comic else (work_dir / page / "cleaned.png")
        if not cleaned_path.exists():
            return jsonify({"error": t("Pagina pulita non ancora disponibile")}), 404
        crop, crop_bbox = render_mod.render_balloon_preview(cleaned_path, det, cfg)
        buf = BytesIO()
        crop.save(buf, format="PNG", compress_level=1)
        return jsonify({
            "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "bbox": list(crop_bbox),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/review-fonts")
def api_review_fonts():
    try:
        cfg = load_cfg()
        fonts_dir = paths.resolve(cfg.get("rendering", {}).get("fonts_dir", "fonts"))
        fonts = {}
        if fonts_dir.exists():
            for f in sorted(list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf"))):
                fonts[f.stem] = str(f)
        return jsonify(fonts)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/review-regenerate", methods=["POST"])
def api_review_regenerate():
    body  = request.get_json()
    comic = body.get("comic", "")
    page  = body.get("page", "")
    data  = body.get("data", {})
    if not _is_safe_component(comic) or not _is_safe_component(page):
        return jsonify({"error": t("Parametro non valido")})

    with _review_regen_lock:
        if _review_regen["running"]:
            return jsonify({"error": t("Una rigenerazione è già in corso")})
        _review_regen.update({"running": True, "message": "", "error": None})

    try:
        cfg = load_cfg()
        work_dir = Path(cfg["paths"]["work_dir"])
        output_dir = Path(cfg["paths"]["output_dir"])
        input_dir = Path(cfg["paths"]["input_dir"])
    except Exception as e:
        with _review_regen_lock:
            _review_regen.update({"running": False, "error": str(e)})
        return jsonify({"error": str(e)})

    comic_work_dir = work_dir / comic if comic else work_dir
    comic_output_dir = output_dir / comic if comic else output_dir
    comic_input_dir = input_dir / comic if comic else input_dir

    detections = data.get("detections", [])
    any_bbox_modified = any(d.get("_bbox_modified") for d in detections)
    for d in detections:
        d.pop("_modified", None)
        d.pop("_bbox_modified", None)

    translated_path = comic_work_dir / page / "translated.json"

    # Se un balloon passa da "testo da tradurre" a onomatopea (o viceversa),
    # cambia l'insieme dei balloon da mascherare/pulire con IOPaint, anche
    # se il bbox non si e' spostato: senza questo controllo, piu' sotto la
    # pipeline riuserebbe cleaned.png cosi' com'e', lasciando quel balloon
    # con la vecchia pulizia (fatta per il vecchio testo) mai aggiornata.
    def _wants_mask(text):
        pulito = (text or "").strip()
        return bool(pulito) and pulito != "-"

    # Un balloon che PERDE il testo (cancellato in Revisione) non richiede di
    # ripassare da ComfyUI: l'area resta pulita/vuota com'era. Serve
    # ripulire via ComfyUI solo quando un balloon GUADAGNA la necessita' di
    # pulizia (prima saltato, ora con testo), perche' solo li' c'e' ancora
    # testo originale da cancellare che il workflow deve elaborare.
    newly_needs_mask = False
    if translated_path.exists():
        try:
            with open(translated_path, "r", encoding="utf-8") as f:
                old_detections = json.load(f).get("detections", [])
            # Confronto per balloon_id, non per posizione in lista: uno split
            # (vedi "Dividi in due") aggiunge un balloon_id nuovo e sposta gli
            # indici di quelli dopo, quindi un confronto posizionale (o per
            # lunghezza) segnalerebbe un cambiamento anche senza che nessun
            # balloon abbia davvero cambiato bisogno di pulizia, facendo
            # ripartire IOPaint a vuoto.
            old_wants_by_id = {d.get("balloon_id"): _wants_mask(d.get("testo_tradotto")) for d in old_detections}
            new_wants_by_id = {d.get("balloon_id"): _wants_mask(d.get("testo_tradotto")) for d in detections}
            common_ids = old_wants_by_id.keys() & new_wants_by_id.keys()
            newly_needs_mask = any(new_wants_by_id[bid] and not old_wants_by_id[bid] for bid in common_ids)
        except Exception:
            newly_needs_mask = True

    try:
        translated_path.parent.mkdir(parents=True, exist_ok=True)
        _backup_translated(translated_path, "revisione")
        with open(translated_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        with _review_regen_lock:
            _review_regen.update({"running": False, "error": str(e)})
        return jsonify({"error": str(e)})

    push_log(f"💾 Salvato: {translated_path}")

    def worker():
        try:
            cleaned_path = comic_work_dir / page / "cleaned.png"
            # Spostare/ridimensionare un box, o cancellare la traduzione di un
            # balloon, non richiede di ripassare da ComfyUI: nel secondo caso
            # l'area resta semplicemente pulita/vuota com'era. Serve ripulire
            # via ComfyUI solo quando un balloon GUADAGNA la necessita' di
            # pulizia (prima saltato, ora con testo: c'e' ancora testo
            # originale da cancellare che il workflow deve elaborare), o se
            # manca ancora cleaned.png.
            if cleaned_path.exists() and not newly_needs_mask:
                push_log(f"Uso pagina già pulita: {cleaned_path}")
            else:
                exts = {".png", ".jpg", ".jpeg", ".webp"}
                candidates = [p for p in comic_input_dir.iterdir()
                              if p.stem == page and p.suffix.lower() in exts] if comic_input_dir.exists() else []
                if not candidates:
                    raise FileNotFoundError(f"Immagine originale non trovata per {page}")
                page_path = candidates[0]
                if newly_needs_mask:
                    push_log("Un balloon ha guadagnato testo da tradurre, rieseguo la pulizia (inpainting)...")
                else:
                    push_log("Pagina pulita non trovata, eseguo pulizia (inpainting)...")
                cleaned_path = clean.run(page_path, translated_path, cfg, comic_work_dir)

            final_path = render_mod.run(cleaned_path, translated_path, cfg, comic_output_dir)
            push_log(f"✅ Pagina rigenerata: {final_path}")
            with _review_regen_lock:
                _review_regen.update({"running": False, "message": f"Pagina rigenerata: {final_path}", "error": None})
        except Exception as e:
            push_log(f"❌ Errore rigenerazione: {e}")
            with _review_regen_lock:
                _review_regen.update({"running": False, "error": str(e)})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "started": True})

@app.route("/api/review-regenerate-status")
def api_review_regenerate_status():
    with _review_regen_lock:
        return jsonify(dict(_review_regen))


# ── Cache endpoints ────────────────────────────────────────────────────────────

@app.route("/api/cache-stats")
def api_cache_stats():
    try:
        s = tc.stats()
        return f'<p>Entries: <b>{s["entries"]}</b> | File: <code class="text-xs">{s["file"]}</code></p>'
    except Exception as e:
        return f'<p class="text-red-400">Errore: {e}</p>'

@app.route("/api/cache-clear", methods=["POST"])
def api_cache_clear():
    try:
        tc.clear()
        return f'<p class="text-green-400">{t("Cache svuotata. Entries: 0")}</p>'
    except Exception as e:
        return f'<p class="text-red-400">Errore: {e}</p>'


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    print("🚀  Web app disponibile su http://127.0.0.1:5000")
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

