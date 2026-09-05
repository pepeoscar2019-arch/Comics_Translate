#!/usr/bin/env python3
"""
Pipeline GUI grafica per la traduzione fumetti con progress bar.
Usa: python pipeline_gui_tk.py
"""
import os
import sys
import json
import subprocess
import time
import threading
import hashlib
import shutil
import re
from pathlib import Path
from datetime import datetime
from tkinter import (SEL,
    Tk, Toplevel, Frame, Label, Button, Text, Scrollbar, Listbox, Entry,
    StringVar, IntVar, BooleanVar, DoubleVar, ttk, messagebox, filedialog,
    Canvas, END, INSERT, BOTH, X, Y, LEFT, RIGHT, TOP, BOTTOM, W, E, N, S,
    DISABLED, NORMAL, CENTER, HORIZONTAL, VERTICAL, colorchooser
)
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import requests
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
import audit
import translate_cache as tc
import clean
import render
import paths
import i18n
from i18n import t

import export_prompt

from functools import lru_cache
render._load_font = lru_cache(maxsize=256)(render._load_font)

CONFIG_FILE = str(paths.CONFIG_PATH)
MAIN_SCRIPT = str(paths.PROJECT_ROOT / "main.py")
EDIT_SCRIPT = "edit_translations.py"

def load_cfg():
    return paths.load_cfg(CONFIG_FILE)

def save_cfg(cfg):
    import yaml
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

FLUX_WORKFLOW_FILE = str(paths.PROJECT_ROOT / "Flux-Klein.json")
# Il workflow Flux-Klein.json serve ancora al solo pretrattamento
# (flux_pretreat.py), che gli passa il proprio prompt a ogni esecuzione. La
# pulizia dei balloon non usa piu' modelli generativi, quindi qui non c'e'
# nessun prompt da salvare.

def build_command(stage, limit=None, comic=None, start=None, end=None, ocr_backend=None, prompt=None, input_dir=None, output_dir=None, megapixels=None):
    if stage == "pretreat":
        # Pretrattamento ComfyUI (flux_pretreat.py): script isolato, main.py
        # non lo conosce. Il prompt e' passato a ogni esecuzione, mai scritto
        # in Flux-Klein.json.
        cmd = [sys.executable, "-u", str(paths.PROJECT_ROOT / "flux_pretreat.py"),
               "--config", CONFIG_FILE, "--prompt", prompt or ""]
        if megapixels is not None:
            cmd.extend(["--megapixels", str(megapixels)])
        if input_dir:
            cmd.extend(["--input-dir", input_dir])
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
    else:
        cmd = [sys.executable, "-u", MAIN_SCRIPT, "--config", CONFIG_FILE, "--stage", stage]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if comic:
        cmd.extend(["--comic", comic])
    if start is not None:
        cmd.extend(["--start", str(start)])
    if end is not None:
        cmd.extend(["--end", str(end)])
    if ocr_backend and stage == "ocr":
        cmd.extend(["--ocr-backend", ocr_backend])
    return cmd

def llama_server_url(cfg: dict) -> str:
    """Endpoint OpenAI-compatible di llama-server, ricavato dalla porta in
    config: e' l'unico backend di inferenza della pipeline."""
    port = cfg.get("llama_server", {}).get("port", 8081)
    return f"http://127.0.0.1:{port}"


def llama_server_is_up(cfg: dict) -> bool:
    try:
        return requests.get(f"{llama_server_url(cfg)}/health", timeout=2).status_code == 200
    except Exception:
        return False


def llama_server_loaded_model(cfg: dict) -> str | None:
    """Nome del modello attualmente servito, o None se il server non risponde.
    llama-server ne tiene uno solo: main.py lo avvia e lo chiude a ogni stage."""
    try:
        resp = requests.get(f"{llama_server_url(cfg)}/v1/models", timeout=2)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data") or []
        return data[0].get("id") if data else None
    except Exception:
        return None


def collect_comics(cfg: dict) -> list[str]:
    base_input = Path(cfg["paths"]["input_dir"])
    if not base_input.exists():
        return []
    subdirs = sorted(d.name for d in base_input.iterdir() if d.is_dir())
    if subdirs:
        return subdirs
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if any(p.suffix.lower() in exts for p in base_input.iterdir()):
        return [""]
    return []

def count_pages(cfg: dict, comic: str | None) -> int:
    base_input = Path(cfg["paths"]["input_dir"])
    if comic:
        input_dir = base_input / comic
    else:
        input_dir = base_input
    if not input_dir.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return len([p for p in input_dir.iterdir() if p.suffix.lower() in exts])

# ============================================================
# NUOVA CLASSE: ReviewWindow Interattiva
# ============================================================
class ReviewWindow:
    """Finestra di revisione interattiva con drag, resize e editor inline."""

    HANDLE_SIZE = 8
    MIN_BBOX_SIZE_IMG = 15

    # Zoom: self._zoom e' un moltiplicatore sopra la scala "adatta alla
    # finestra" di default (zoom=1.0 = comportamento di sempre). Il fattore
    # di scala finale e' comunque limitato da SCALE_HARD_CAP per non provare
    # a generare una PhotoImage enorme (es. una pagina 3000x4000px a 8x).
    ZOOM_MIN = 0.2
    ZOOM_MAX = 8.0
    ZOOM_STEP = 1.15
    SCALE_HARD_CAP = 4.0

    def __init__(self, parent_app: "PipelineGUI"):
        self.app = parent_app
        self.win = Toplevel(parent_app.root)
        self.win.title(t("Revisione fumetto - Interattiva"))
        if sys.platform == "win32":
            self.win.state("zoomed")
        else:
            self.win.attributes("-zoomed", True)
        self.win.minsize(900, 600)

        self._review_data = None
        self._review_page_id = None
        self._review_comic = None
        self._review_photo = None
        self._current_image_path = None

        self._review_scale = 1.0
        self._img_offset = (0, 0)
        self._review_boxes = []
        self._zoom = 1.0
        self._zoom_job = None
        self._pending_zoom_anchor = None
        self._page_image_full = None

        self._drag_state = None
        self._resize_job = None
        self._review_dirty = False

        self._drag_previews = {}  # idx -> (photo, canvas_item_id)
        self._erase_overlays = {}  # idx -> (photo, canvas_item_id)
        self._drag_preview_last_ts = 0.0
        self._render_cfg = None
        self._cleaned_image_full = None
        self._inline_editor = None
        self._review_selected_idx = None
        self._new_box_mode = False

        self._build_ui()
        i18n.translate_tk_tree(self.win)
        self._refresh_comics()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        top_frame = Frame(self.win)
        top_frame.pack(fill=X, padx=10, pady=8)

        Label(top_frame, text="Fumetto:").pack(side=LEFT)
        self.comic_var = StringVar()
        self.combo_comics = ttk.Combobox(top_frame, textvariable=self.comic_var, state="readonly", width=35)
        self.combo_comics.pack(side=LEFT, padx=5)
        self.combo_comics.bind("<<ComboboxSelected>>", lambda e: self._refresh_pages())
        ttk.Button(top_frame, text="Aggiorna", command=self._refresh_comics).pack(side=LEFT, padx=5)

        Label(top_frame, text="Pagina:").pack(side=LEFT, padx=(15, 0))
        self.page_var = StringVar()
        self.combo_pages = ttk.Combobox(top_frame, textvariable=self.page_var, state="readonly", width=10)
        self.combo_pages.pack(side=LEFT, padx=5)
        self.combo_pages.bind("<<ComboboxSelected>>", lambda e: self._load_page())

        ttk.Button(top_frame, text="< Precedente", command=lambda: self._nav_page(-1)).pack(side=LEFT, padx=5)
        ttk.Button(top_frame, text="Successiva >", command=lambda: self._nav_page(1)).pack(side=LEFT, padx=5)

        self.btn_regenerate = ttk.Button(
            top_frame, text=" Rigenera pagina", command=self._save_and_regenerate, state=DISABLED
        )
        self.btn_regenerate.pack(side=LEFT, padx=15)

        self.btn_new_box = ttk.Button(top_frame, text="➕ Nuovo box", command=self._toggle_new_box_mode)
        self.btn_new_box.pack(side=LEFT, padx=5)
        self.btn_delete_box = ttk.Button(
            top_frame, text="🗑 Elimina box", command=self._delete_selected_box, state=DISABLED
        )
        self.btn_delete_box.pack(side=LEFT, padx=5)

        self.btn_audit = ttk.Button(top_frame, text="\U0001F50D Controlla fumetto", command=self._audit_comic)
        self.btn_audit.pack(side=LEFT, padx=5)

        ttk.Button(top_frame, text="Chiudi", command=self._on_close).pack(side=RIGHT, padx=5)

        zoom_frame = Frame(top_frame)
        zoom_frame.pack(side=RIGHT, padx=(5, 15))
        ttk.Button(zoom_frame, text="Adatta", command=self._zoom_reset, width=6).pack(side=LEFT)
        ttk.Button(zoom_frame, text="🔍+", width=3, command=lambda: self._zoom_by(self.ZOOM_STEP)).pack(side=LEFT, padx=(5, 0))
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=5, anchor=CENTER)
        self.zoom_label.pack(side=LEFT, padx=3)
        ttk.Button(zoom_frame, text="🔍−", width=3, command=lambda: self._zoom_by(1 / self.ZOOM_STEP)).pack(side=LEFT)

        hint = ttk.Label(
            self.win,
            text="🖱️ Clicca e trascina un balloon per spostarlo | Trascina gli handle per ridimensionare | Doppio-click per modificare il testo | ➕ Nuovo box per aggiungerne uno, poi Canc/🗑 per eliminare quello selezionato | 🔴 Rosso ⚠ = testo mancante, da trascrivere | 🟠 Arancione ✂ = testo troncato, da accorciare | 🔎 Rotella o frecce ↑↓ per zoomare | ✋ Trascina area vuota (o tasto centrale) per scorrere",
            foreground="gray"
        )
        hint.pack(anchor=W, padx=10, pady=(0, 5))

        canvas_frame = Frame(self.win, bg="#222")
        canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0, 5))
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = Canvas(canvas_frame, bg="#222", cursor="hand2", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vbar = Scrollbar(canvas_frame, orient=VERTICAL, command=self.canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = Scrollbar(canvas_frame, orient=HORIZONTAL, command=self.canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows/Mac
        self.canvas.bind("<Button-4>", self._on_mousewheel)    # Linux, rotella su
        self.canvas.bind("<Button-5>", self._on_mousewheel)    # Linux, rotella giu'
        # Pan col tasto centrale, funziona anche sopra un balloon (a
        # differenza del trascinamento con tasto sinistro su area vuota).
        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)

        self.win.bind("<Up>", lambda e: self._zoom_by(self.ZOOM_STEP))
        self.win.bind("<Down>", lambda e: self._zoom_by(1 / self.ZOOM_STEP))
        # Solo quando il focus e' sul canvas (non su una textarea/entry
        # dell'editor inline, dove Canc deve cancellare caratteri come da
        # comportamento normale del widget).
        self.win.bind("<Delete>", self._on_delete_key)

        self.status = ttk.Label(self.win, text="", foreground="black")
        self.status.pack(anchor=W, padx=10, pady=(0, 8))

    def _refresh_comics(self):
        try:
            cfg = load_cfg()
            comics = collect_comics(cfg)
            values = [c or "(root)" for c in comics]
            self.combo_comics.config(values=values)
            if values:
                self.comic_var.set(values[0])
                self._refresh_pages()
        except Exception as e:
            self.status.config(text=f"Errore: {e}", foreground="red")

    def _refresh_pages(self):
        try:
            cfg = load_cfg()
            output_dir = Path(cfg["paths"]["output_dir"])
            comic = self.comic_var.get()
            if comic == "(root)":
                comic = ""
            base = output_dir / comic if comic else output_dir
            pages = []
            if base.exists():
                pages = sorted(p.stem for p in base.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
            self.combo_pages.config(values=pages)
            if pages:
                self.page_var.set(pages[0])
                self._load_page()
            else:
                self.canvas.delete("all")
                self.status.config(text=t("Nessuna pagina renderizzata trovata."), foreground="orange")
        except Exception as e:
            self.status.config(text=f"Errore: {e}", foreground="red")

    def _nav_page(self, delta: int):
        pages = list(self.combo_pages["values"])
        if not pages:
            return
        current = self.page_var.get()
        if current not in pages:
            return
        idx = pages.index(current) + delta
        if 0 <= idx < len(pages):
            self.page_var.set(pages[idx])
            self._load_page()

    def _load_page(self):
        if self._review_dirty:
            if not messagebox.askyesno(
                t("Modifiche non salvate"),
                t("Ci sono modifiche non rigenerate. Vuoi scartarle e continuare?")
            ):
                self.page_var.set(self._review_page_id or "")
                return
        self._review_dirty = False
        self.btn_regenerate.config(state=DISABLED)
        self._close_inline_editor()
        self._review_selected_idx = None
        self.btn_delete_box.config(state=DISABLED)
        if self._new_box_mode:
            self._toggle_new_box_mode()

        try:
            cfg = load_cfg()
            work_dir = Path(cfg["paths"]["work_dir"])
            output_dir = Path(cfg["paths"]["output_dir"])
            comic = self.comic_var.get()
            if comic == "(root)":
                comic = ""
            page_id = self.page_var.get()
            if not page_id:
                return

            self._review_comic = comic
            self._review_page_id = page_id

            translated_path = (work_dir / comic / page_id / "translated.json") if comic else (work_dir / page_id / "translated.json")
            if not translated_path.exists():
                messagebox.showwarning(t("Attenzione"), f"translated.json non trovato per {page_id}")
                return

            with open(translated_path, "r", encoding="utf-8") as f:
                self._review_data = json.load(f)

            page_dir = (output_dir / comic) if comic else output_dir
            image_path = None
            for ext in (".png", ".webp", ".jpg", ".jpeg"):
                candidate = page_dir / f"{page_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                messagebox.showwarning(t("Attenzione"), f"Pagina renderizzata non trovata: {page_dir / page_id}")
                return

            self._current_image_path = image_path
            # Aperta/decodificata una sola volta per pagina invece che ad ogni
            # frame di zoom/resize: decodificare da disco una pagina intera
            # (spesso alcuni MB) ad ogni rotellata era la causa principale
            # dello scatto durante uno zoom rapido.
            self._page_image_full = Image.open(image_path)
            self._page_image_full.load()

            cleaned_path = (work_dir / comic / page_id / "cleaned.png") if comic else (work_dir / page_id / "cleaned.png")
            self._cleaned_image_full = Image.open(cleaned_path).convert("RGB") if cleaned_path.exists() else None

            self._render_canvas()
            dets = self._review_data.get("detections", [])
            da_trascrivere = sum(
                1 for d in dets
                if d.get("ocr_empty_suspect") and not (d.get("testo_tradotto") or "").strip()
            )
            troncati = sum(
                1 for d in dets
                if d.get("_overflow") and (d.get("testo_tradotto") or "").strip()
            )
            testo = f"Pagina: {page_id} | {len(dets)} balloon"
            colore = "black"
            if da_trascrivere:
                testo += f" | \u26a0 {da_trascrivere} da trascrivere (OCR vuoto ma il balloon contiene testo)"
                colore = "red"
            if troncati:
                testo += f" | \u2702 {troncati} con testo troncato (da accorciare)"
                colore = "red" if da_trascrivere else "#cc6600"
            self.status.config(text=testo, foreground=colore)
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _audit_comic(self):
        """Controlla tutte le pagine del fumetto (audit.py) e apre l'elenco
        dei problemi. La misura di capienza di ogni balloon costa qualche
        decina di secondi su un volume lungo, quindi gira in un thread: la
        finestra resta usabile e il bottone si disabilita nel frattempo."""
        comic = self.comic_var.get()
        if comic == "(root)":
            comic = ""
        self.btn_audit.config(state=DISABLED)
        self.status.config(text=t("Controllo del fumetto in corso..."), foreground="black")

        def lavoro():
            try:
                cfg = load_cfg()
                work_dir = Path(cfg["paths"]["work_dir"])
                output_dir = Path(cfg["paths"]["output_dir"])
                result = audit.audit_comic(
                    work_dir / comic if comic else work_dir,
                    output_dir / comic if comic else output_dir,
                    cfg,
                )
                errore = None
            except Exception as e:
                result, errore = None, str(e)
            # Tkinter non e' thread-safe: il risultato torna al thread della
            # GUI con after(), che e' l'unico modo supportato.
            self.win.after(0, lambda: self._audit_done(result, errore))

        threading.Thread(target=lavoro, daemon=True).start()

    def _audit_done(self, result, errore):
        self.btn_audit.config(state=NORMAL)
        if errore:
            self.status.config(text=f"Controllo fallito: {errore}", foreground="red")
            return

        t = result["totals"]
        problemi = (t["non_tradotto"] + t["da_trascrivere"] + t["troncato"]
                    + t["troppo_lungo"] + t.get("doppione", 0) + t["pagine_incomplete"])
        riepilogo = (
            f"{problemi} da sistemare su {t['balloon']} balloon: "
            f"{t['non_tradotto']} non tradotti, {t['da_trascrivere']} da trascrivere, "
            f"{t['troncato']} troncati, {t['troppo_lungo']} troppo lunghi, "
            f"{t.get('doppione', 0)} doppioni, {t['pagine_incomplete']} stage mancanti"
        )
        self.status.config(text=riepilogo, foreground="red" if problemi else "green")
        if not problemi:
            messagebox.showinfo(t("Controllo fumetto"), f"Nessun problema.\n\n{riepilogo}")
            return
        self._show_audit_window(result, riepilogo)

    def _show_audit_window(self, result, riepilogo: str):
        """Elenco dei problemi; doppio click su una riga apre quella pagina e
        seleziona il balloon, che e' il motivo per cui questo elenco esiste."""
        win = Toplevel(self.win)
        win.title(t("Controllo fumetto"))
        win.geometry("900x500")
        Label(win, text=riepilogo, anchor=W, justify=LEFT, fg="gray20").pack(fill=X, padx=10, pady=(8, 4))
        Label(win, text="Doppio click su una riga per aprire la pagina sul balloon indicato.",
              anchor=W, fg="gray40", font=("Arial", 8)).pack(fill=X, padx=10)

        cols = ("pagina", "balloon", "tipo", "dettaglio")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for col, larghezza in zip(cols, (80, 70, 130, 560)):
            tree.heading(col, text=col.capitalize())
            tree.column(col, width=larghezza, anchor=W)
        scroll = Scrollbar(win, orient=VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0), pady=8)
        scroll.pack(side=LEFT, fill=Y, pady=8, padx=(0, 10))

        righe = {}
        for pagina in result["pages"]:
            for problema in pagina["issues"]:
                bid = problema.get("balloon")
                item = tree.insert("", END, values=(
                    pagina["page"],
                    "" if bid is None else bid,
                    problema["tipo"],
                    (problema.get("dettaglio") or "")[:120],
                ))
                righe[item] = (pagina["page"], bid)

        def apri(_event=None):
            sel = tree.selection()
            if not sel:
                return
            page_id, bid = righe.get(sel[0], (None, None))
            self._goto_issue(page_id, bid)

        tree.bind("<Double-Button-1>", apri)
        tree.bind("<Return>", apri)

    def _goto_issue(self, page_id: str | None, balloon_id=None) -> bool:
        """Apre `page_id` e seleziona il balloon indicato. Ritorna True se il
        salto e' andato a buon fine.

        Il balloon si cerca per `balloon_id` e non per posizione nella lista:
        una divisione in due (vedi "Dividi") inserisce un id nuovo e sposta
        gli indici di tutti quelli dopo, quindi un indice memorizzato nel
        rapporto punterebbe al balloon sbagliato.
        """
        if not page_id:
            return False
        if page_id not in list(self.combo_pages["values"]):
            messagebox.showwarning(
                t("Pagina non disponibile"),
                f"La pagina {page_id} non risulta fra quelle renderizzate: "
                "va prima completato il render."
            )
            return False
        if self.page_var.get() != page_id:
            self.page_var.set(page_id)
            self._load_page()
        if balloon_id is None:
            return True
        for i, det in enumerate((self._review_data or {}).get("detections", [])):
            if det.get("balloon_id") == balloon_id:
                self._select_box(i)
                return True
        return False

    def _on_canvas_resize(self, event):
        if self._resize_job:
            self.win.after_cancel(self._resize_job)
        self._resize_job = self.win.after(150, self._render_canvas)

    # Colori dei box, in ordine di priorita' decrescente. Sono gli stessi
    # della Revisione web: passando da una interfaccia all'altra un colore
    # deve voler dire la stessa cosa.
    COLOR_MODIFIED = "#ffc107"        # modificato, non ancora rigenerato
    COLOR_TO_TRANSCRIBE = "#ff1744"   # manca il testo: da trascrivere a mano
    COLOR_OVERFLOW = "#ff9100"        # testo troncato nel render: da accorciare
    COLOR_OK = "#00c853"              # tradotto, a posto
    COLOR_EMPTY = "#888888"           # vuoto di suo (insegna, onomatopea)

    def _box_style(self, det: dict, modified: bool = False) -> tuple[str, str]:
        """(colore, badge) del box di questo balloon.

        Le due segnalazioni automatiche vengono da monte:
        `ocr_empty_suspect` lo mette ocr.py quando il modello ha letto vuoto
        ma nel ritaglio c'e' del testo (il balloon resterebbe in lingua
        originale); `_overflow` lo mette render.py quando ha dovuto tagliare
        il testo per farlo stare (contenuto perso). Entrambe valgono solo
        finche' il testo non c'e' / c'e': un balloon segnalato ma gia'
        trascritto e' a posto e torna verde.
        """
        has_text = bool((det.get("testo_tradotto") or "").strip())
        needs_transcription = bool(det.get("ocr_empty_suspect")) and not has_text
        is_overflow = bool(det.get("_overflow")) and has_text

        if modified:
            # Chi sta modificando vuole vedere cosa ha toccato: il giallo
            # vince, il problema resta segnalato dal badge.
            color = self.COLOR_MODIFIED
        elif needs_transcription:
            color = self.COLOR_TO_TRANSCRIBE
        elif is_overflow:
            color = self.COLOR_OVERFLOW
        else:
            color = self.COLOR_OK if has_text else self.COLOR_EMPTY

        badge = "\u26a0" if needs_transcription else ("\u2702" if is_overflow else "")
        return color, badge

    def _render_canvas(self):
        if not self._current_image_path or not self._review_data:
            return
        self._close_inline_editor()
        self.canvas.delete("all")
        self._review_boxes = []
        self._drag_previews = {}
        self._erase_overlays = {}

        if self._page_image_full is None:
            self._page_image_full = Image.open(self._current_image_path)
            self._page_image_full.load()
        img = self._page_image_full
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            self.win.after(100, self._render_canvas)
            return

        fit_scale = min(canvas_w / img.width, canvas_h / img.height, 1.0)
        if fit_scale == 1.0:
            fit_scale = min(canvas_w / img.width, canvas_h / img.height)
        scale = min(fit_scale * self._zoom, self.SCALE_HARD_CAP)
        self._review_scale = scale
        if hasattr(self, "zoom_label"):
            self.zoom_label.config(text=f"{round(scale * 100)}%")

        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        # BILINEAR invece di LANCZOS: molto piu' veloce, differenza di
        # qualita' trascurabile per un'anteprima interattiva (non e' l'export
        # finale). LANCZOS su una pagina intera ad ogni rotellata era l'altra
        # causa principale dello scatto.
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        self._review_photo = ImageTk.PhotoImage(img_resized)

        offset_x = max(0, (canvas_w - new_w) // 2)
        offset_y = max(0, (canvas_h - new_h) // 2)
        self._img_offset = (offset_x, offset_y)

        scroll_w = max(canvas_w, new_w)
        scroll_h = max(canvas_h, new_h)
        self.canvas.configure(scrollregion=(0, 0, scroll_w, scroll_h))

        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self._review_photo)

        for idx, det in enumerate(self._review_data.get("detections", [])):
            bbox = det.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            sx1 = offset_x + x1 * scale
            sy1 = offset_y + y1 * scale
            sx2 = offset_x + x2 * scale
            sy2 = offset_y + y2 * scale

            color, badge = self._box_style(det, modified=self._review_dirty and bool(det.get("_modified")))
            needs_transcription = badge == "\u26a0"
            is_overflow = badge == "\u2702"

            rect_id = self.canvas.create_rectangle(
                sx1, sy1, sx2, sy2,
                outline=color, width=2, fill="",
                tags=(f"balloon_{idx}", "bbox")
            )

            handle_ids = []
            handles = self._get_handle_positions(sx1, sy1, sx2, sy2)
            for h_name, (hx, hy) in handles.items():
                hid = self.canvas.create_rectangle(
                    hx - self.HANDLE_SIZE // 2, hy - self.HANDLE_SIZE // 2,
                    hx + self.HANDLE_SIZE // 2, hy + self.HANDLE_SIZE // 2,
                    fill=color, outline="white", width=1,
                    tags=(f"handle_{idx}_{h_name}", "handle")
                )
                handle_ids.append((h_name, hid))

            if badge:
                # Un balloon rosso in mezzo a tanti verdi si vede, ma a pagina
                # intera (zoom ~20%) il bordo e' un filo di 2px: il simbolo
                # resta leggibile anche rimpicciolito e distingue subito i due
                # casi senza doverli ricordare a memoria dai colori.
                self.canvas.create_text(
                    sx2, sy1, text=badge,
                    font=("Segoe UI Emoji", 13), fill=color,
                    anchor="center", tags=(f"balloon_{idx}", "bbox")
                )

            if det.get("manual_balloon_shape") == "ellipse":
                # Balloon ricreato da zero (pulizia automatica che ha
                # cancellato l'intera sagoma): icona ben visibile, non un
                # piccolo quadratino colore, cosi' si individua a colpo
                # d'occhio tra tanti balloon.
                self.canvas.create_text(
                    sx1, sy1, text="\U0001F501", font=("Segoe UI Emoji", 14),
                    anchor="center", tags=(f"balloon_{idx}", "bbox")
                )
            elif det.get("manual_fill_color") or det.get("manual_border"):
                self.canvas.create_rectangle(
                    sx1 - 3, sy1 - 3, sx1 + 7, sy1 + 7,
                    fill=det.get("manual_fill_color") or "",
                    outline="black" if det.get("manual_border") else "white",
                    width=2, tags=(f"balloon_{idx}", "bbox")
                )

            self._review_boxes.append({
                "rect_id": rect_id,
                "handle_ids": handle_ids,
                "idx": idx,
                "bbox_screen": (sx1, sy1, sx2, sy2)
            })

        if self._review_selected_idx is not None:
            for box in self._review_boxes:
                if box["idx"] == self._review_selected_idx:
                    self.canvas.itemconfig(box["rect_id"], width=4)

    def _screen_to_image_point(self, cx: float, cy: float) -> tuple:
        """Coordinate canvas (spazio scroll, non widget) -> coordinate immagine."""
        offset_x, offset_y = self._img_offset
        scale = self._review_scale or 1.0
        return (cx - offset_x) / scale, (cy - offset_y) / scale

    def _zoom_reset(self):
        if self._zoom_job:
            self.win.after_cancel(self._zoom_job)
            self._zoom_job = None
        self._zoom = 1.0
        self._render_canvas()

    def _zoom_by(self, factor: float, widget_xy: tuple = None):
        """Applica lo zoom mantenendo fermo sotto il cursore (rotella) o al
        centro della vista (tasti/pulsanti) lo stesso punto dell'immagine,
        cosi' zoomare non "salta" via dalla zona che si stava guardando.

        Il valore di zoom si aggiorna subito, ma il render vero e proprio
        (resize dell'immagine intera + ridisegno di tutti i balloon) e'
        posticipato di qualche decina di ms e annullato/riprogrammato ad
        ogni chiamata successiva: durante uno scroll rapido della rotella
        arrivano molti eventi in pochi ms, e renderizzare ad ogni singolo
        tick invece che una volta sola alla fine e' la causa principale
        dello scatto.
        """
        if not self._current_image_path:
            return
        if widget_xy is None:
            widget_xy = (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

        old_zoom = self._zoom
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        self._zoom = new_zoom

        # L'ancora (punto immagine da tenere fermo) va calcolata ORA, con la
        # scala ancora precedente, non al momento del render posticipato.
        cx_before = self.canvas.canvasx(widget_xy[0])
        cy_before = self.canvas.canvasy(widget_xy[1])
        img_point = self._screen_to_image_point(cx_before, cy_before)
        self._pending_zoom_anchor = (widget_xy, img_point)

        if self._zoom_job:
            self.win.after_cancel(self._zoom_job)
        self._zoom_job = self.win.after(20, self._apply_pending_zoom)

    def _apply_pending_zoom(self):
        self._zoom_job = None
        if not self._current_image_path or not self._pending_zoom_anchor:
            return
        (wx, wy), (img_x, img_y) = self._pending_zoom_anchor

        self._render_canvas()

        offset_x, offset_y = self._img_offset
        scale = self._review_scale or 1.0
        new_cx = offset_x + img_x * scale
        new_cy = offset_y + img_y * scale

        bbox = self.canvas.bbox("all")
        scroll_w = bbox[2] if bbox else self.canvas.winfo_width()
        scroll_h = bbox[3] if bbox else self.canvas.winfo_height()
        if scroll_w > 0:
            self.canvas.xview_moveto(max(0.0, (new_cx - wx) / scroll_w))
        if scroll_h > 0:
            self.canvas.yview_moveto(max(0.0, (new_cy - wy) / scroll_h))

    def _on_mousewheel(self, event):
        if not self._current_image_path:
            return
        delta = getattr(event, "delta", 0)
        if getattr(event, "num", None) == 4 or delta > 0:
            factor = self.ZOOM_STEP
        elif getattr(event, "num", None) == 5 or delta < 0:
            factor = 1 / self.ZOOM_STEP
        else:
            return
        self._zoom_by(factor, (event.x, event.y))

    def _get_handle_positions(self, x1, y1, x2, y2):
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        return {
            "nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1),
            "w": (x1, my), "e": (x2, my),
            "sw": (x1, y2), "s": (mx, y2), "se": (x2, y2),
        }

    def _on_canvas_press(self, event):
        self.canvas.focus_set()
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        if self._new_box_mode:
            self._drag_state = {"type": "new_box", "start_x": cx, "start_y": cy, "rect_id": None}
            return

        for box in self._review_boxes:
            for h_name, h_id in box["handle_ids"]:
                hx, hy = self.canvas.coords(h_id)[:2]
                if abs(cx - hx) <= self.HANDLE_SIZE and abs(cy - hy) <= self.HANDLE_SIZE:
                    self._select_box(box["idx"])
                    self._drag_state = {
                        "type": "resize",
                        "idx": box["idx"],
                        "handle": h_name,
                        "start_x": cx,
                        "start_y": cy,
                        "orig_bbox": list(box["bbox_screen"]),
                    }
                    self.canvas.config(cursor="fleur")
                    self._start_drag_erase_overlay(box["idx"], box["bbox_screen"])
                    return

        for box in self._review_boxes:
            sx1, sy1, sx2, sy2 = box["bbox_screen"]
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                self._select_box(box["idx"])
                self._drag_state = {
                    "type": "move",
                    "idx": box["idx"],
                    "start_x": cx,
                    "start_y": cy,
                    "orig_bbox": list(box["bbox_screen"]),
                }
                self.canvas.config(cursor="fleur")
                self._start_drag_erase_overlay(box["idx"], box["bbox_screen"])
                return

        # Click su area vuota (nessun balloon/handle sotto il cursore): pan
        # della vista invece di non fare nulla. Le scrollbar da sole sono
        # facili da non notare quando si e' zoomati.
        self._select_box(None)
        self._drag_state = {"type": "pan"}
        self.canvas.config(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def _select_box(self, idx: int | None):
        """Evidenzia il box scelto (bordo piu' spesso) senza rifare tutto
        _render_canvas (costoso: ridimensiona l'intera pagina), cosi'
        selezionare resta immediato anche su pagine grandi/zoomate."""
        prev = self._review_selected_idx
        if prev is not None:
            for box in self._review_boxes:
                if box["idx"] == prev:
                    self.canvas.itemconfig(box["rect_id"], width=2)
        self._review_selected_idx = idx
        if idx is not None:
            for box in self._review_boxes:
                if box["idx"] == idx:
                    self.canvas.itemconfig(box["rect_id"], width=4)
        self.btn_delete_box.config(state=NORMAL if idx is not None else DISABLED)

    def _on_delete_key(self, event):
        if self.win.focus_get() is self.canvas:
            self._delete_selected_box()

    def _toggle_new_box_mode(self):
        self._new_box_mode = not self._new_box_mode
        if self._new_box_mode:
            self._close_inline_editor()
            self.btn_new_box.config(text=t("✖ Annulla nuovo box"))
            self.canvas.config(cursor="crosshair")
            self.status.config(text=t("Trascina sul canvas per disegnare il nuovo box."), foreground="black")
        else:
            self.btn_new_box.config(text=t("➕ Nuovo box"))
            self.canvas.config(cursor="hand2")

    def _delete_selected_box(self):
        idx = self._review_selected_idx
        if idx is None or not self._review_data:
            return
        if not messagebox.askyesno(t("Elimina box"), t("Eliminare questo balloon dalla pagina?")):
            return
        self._close_inline_editor()
        del self._review_data["detections"][idx]
        self._review_selected_idx = None
        self.btn_delete_box.config(state=DISABLED)
        self._mark_dirty()
        self._render_canvas()

    def _on_canvas_drag(self, event):
        if not self._drag_state:
            return

        if self._drag_state["type"] == "pan":
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            return

        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        if self._drag_state["type"] == "new_box":
            if self._drag_state["rect_id"] is not None:
                self.canvas.delete(self._drag_state["rect_id"])
            self._drag_state["rect_id"] = self.canvas.create_rectangle(
                self._drag_state["start_x"], self._drag_state["start_y"], cx, cy,
                outline="#2979ff", width=2, dash=(4, 2),
            )
            self._drag_state["end_x"], self._drag_state["end_y"] = cx, cy
            return

        dx = cx - self._drag_state["start_x"]
        dy = cy - self._drag_state["start_y"]
        orig = self._drag_state["orig_bbox"]

        if self._drag_state["type"] == "move":
            new_x1 = orig[0] + dx
            new_y1 = orig[1] + dy
            new_x2 = orig[2] + dx
            new_y2 = orig[3] + dy
            new_bbox = (new_x1, new_y1, new_x2, new_y2)
            self._update_bbox_screen(self._drag_state["idx"], new_bbox)
            self._update_drag_text_preview(self._drag_state["idx"], new_bbox)

        elif self._drag_state["type"] == "resize":
            h = self._drag_state["handle"]
            x1, y1, x2, y2 = orig
            min_size = self.MIN_BBOX_SIZE_IMG * self._review_scale
            if "n" in h:
                y1 = min(orig[1] + dy, orig[3] - min_size)
            if "s" in h:
                y2 = max(orig[3] + dy, orig[1] + min_size)
            if "w" in h:
                x1 = min(orig[0] + dx, orig[2] - min_size)
            if "e" in h:
                x2 = max(orig[2] + dx, orig[0] + min_size)
            new_bbox = (x1, y1, x2, y2)
            self._update_bbox_screen(self._drag_state["idx"], new_bbox)
            self._update_drag_text_preview(self._drag_state["idx"], new_bbox)

    def _on_canvas_release(self, event):
        if self._drag_state and self._drag_state["type"] == "new_box":
            self._finish_new_box(self._drag_state)
            self._drag_state = None
            return
        if self._drag_state:
            self.canvas.config(cursor="hand2")
            if self._drag_state["type"] != "pan":
                self._commit_bbox_changes(self._drag_state["idx"])
                self._mark_dirty()
            self._drag_state = None

    def _finish_new_box(self, drag_state: dict):
        self.canvas.config(cursor="hand2")
        if drag_state.get("rect_id") is not None:
            self.canvas.delete(drag_state["rect_id"])
        self._new_box_mode = False
        self.btn_new_box.config(text=t("➕ Nuovo box"))

        x0, y0 = drag_state["start_x"], drag_state["start_y"]
        x1, y1 = drag_state.get("end_x", x0), drag_state.get("end_y", y0)
        sx1, sx2 = sorted((x0, x1))
        sy1, sy2 = sorted((y0, y1))
        min_size = self.MIN_BBOX_SIZE_IMG * self._review_scale
        if (sx2 - sx1) < min_size or (sy2 - sy1) < min_size:
            self.status.config(text=t("Box troppo piccolo, riprova trascinando un'area piu' ampia."), foreground="orange")
            return

        ix1, iy1 = self._screen_to_image_point(sx1, sy1)
        ix2, iy2 = self._screen_to_image_point(sx2, sy2)

        existing_ids = [d.get("balloon_id") for d in self._review_data["detections"] if isinstance(d.get("balloon_id"), int)]
        new_id = (max(existing_ids) + 1) if existing_ids else len(self._review_data["detections"])
        default_font = next(iter(self._get_available_fonts().values()), None)
        # Nessun mask_path: un box aggiunto a mano qui non ha una maschera di
        # detection, quindi clean.py (che filtra su "mask_path" in det) non
        # lo include nella pulizia ComfyUI. Pensato per testo/onomatopee su
        # aree gia' pulite; se serve anche pulire sotto, l'utente puo' usare
        # "Riempi box"/"Ricrea balloon" gia' esistenti nell'editor.
        new_det = {
            "balloon_id": new_id,
            "bbox": [int(ix1), int(iy1), int(ix2), int(iy2)],
            "testo_originale": "",
            "testo_tradotto": "",
            "font_path": default_font,
            "font_auto": False,
            "manual_text_box": True,
            "_modified": True,
            "_overflow": False,
        }
        self._review_data["detections"].append(new_det)
        new_idx = len(self._review_data["detections"]) - 1
        self._mark_dirty()
        self._render_canvas()
        self._select_box(new_idx)
        box = next((b for b in self._review_boxes if b["idx"] == new_idx), None)
        if box:
            self._open_inline_editor(new_idx, *box["bbox_screen"])

    def _on_pan_press(self, event):
        """Pan col tasto centrale, utilizzabile anche sopra un balloon
        (a differenza del click sinistro su area vuota)."""
        self.canvas.focus_set()
        self._drag_state = {"type": "pan"}
        self.canvas.config(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_drag(self, event):
        if self._drag_state and self._drag_state["type"] == "pan":
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_pan_release(self, event):
        if self._drag_state and self._drag_state["type"] == "pan":
            self.canvas.config(cursor="hand2")
            self._drag_state = None

    def _update_bbox_screen(self, idx: int, new_screen_bbox: tuple):
        for box in self._review_boxes:
            if box["idx"] == idx:
                sx1, sy1, sx2, sy2 = new_screen_bbox
                self.canvas.coords(box["rect_id"], sx1, sy1, sx2, sy2)
                handles = self._get_handle_positions(sx1, sy1, sx2, sy2)
                for h_name, h_id in box["handle_ids"]:
                    hx, hy = handles[h_name]
                    self.canvas.coords(
                        h_id,
                        hx - self.HANDLE_SIZE // 2, hy - self.HANDLE_SIZE // 2,
                        hx + self.HANDLE_SIZE // 2, hy + self.HANDLE_SIZE // 2,
                    )
                box["bbox_screen"] = new_screen_bbox
                return

    def _clear_drag_preview(self, idx: int):
        old = self._drag_previews.pop(idx, None)
        if old:
            self.canvas.delete(old[1])

    def _screen_to_image_bbox(self, screen_bbox: tuple) -> tuple:
        sx1, sy1, sx2, sy2 = screen_bbox
        offset_x, offset_y = self._img_offset
        scale = self._review_scale or 1.0
        img_w, img_h = self._cleaned_image_full.size
        x1 = int((sx1 - offset_x) / scale)
        y1 = int((sy1 - offset_y) / scale)
        x2 = int((sx2 - offset_x) / scale)
        y2 = int((sy2 - offset_y) / scale)
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return x1, y1, x2, y2

    def _start_drag_erase_overlay(self, idx: int, orig_screen_bbox: tuple):
        """All'inizio del trascinamento, copre la posizione originale del
        balloon con la porzione di immagine gia' pulita (senza testo), cosi'
        il vecchio testo "cotto" nel render non resta visibile mentre lo
        sposti altrove."""
        if self._cleaned_image_full is None:
            return
        sx1, sy1, sx2, sy2 = orig_screen_bbox
        width = max(1, int(sx2 - sx1))
        height = max(1, int(sy2 - sy1))
        x1, y1, x2, y2 = self._screen_to_image_bbox(orig_screen_bbox)
        crop = self._cleaned_image_full.crop((x1, y1, x2, y2)).resize((width, height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(crop)

        old = self._erase_overlays.get(idx)
        if old:
            self.canvas.delete(old[1])
        item_id = self.canvas.create_image(sx1, sy1, anchor="nw", image=photo, tags=("erase_overlay",))
        self.canvas.tag_raise("bbox")
        self.canvas.tag_raise("handle")
        self._erase_overlays[idx] = (photo, item_id)

    def _update_drag_text_preview(self, idx: int, screen_bbox: tuple):
        """Disegna un'anteprima live del testo tradotto dentro il bbox in
        movimento, usando lo stesso algoritmo di fitting/wrap del render
        finale (font e a-capo reali), cosi' durante il trascinamento si vede
        subito se e come il testo entrera' nella nuova posizione."""
        now = time.time()
        if now - self._drag_preview_last_ts < 0.04:
            return
        self._drag_preview_last_ts = now

        det = self._review_data["detections"][idx]
        text = det.get("testo_tradotto", "").strip()
        fill_hex = det.get("manual_fill_color")
        text_color = det.get("manual_text_color") or (0, 0, 0)
        has_border = bool(det.get("manual_border"))
        is_ellipse = det.get("manual_balloon_shape") == "ellipse"
        if (not text or text == "-") and not fill_hex and not has_border:
            self._clear_drag_preview(idx)
            return

        sx1, sy1, sx2, sy2 = screen_bbox
        width = max(1, int(sx2 - sx1))
        height = max(1, int(sy2 - sy1))
        scale = self._review_scale or 1.0

        if self._render_cfg is None:
            try:
                self._render_cfg = load_cfg()["rendering"]
            except Exception:
                return
        render_cfg = self._render_cfg
        font_path = str(paths.resolve(det.get("font_path") or render_cfg["font_path"]))

        box_width_img = max(1, int(width / scale))
        box_height_img = max(1, int(height / scale))

        if self._cleaned_image_full is not None:
            x1, y1, x2, y2 = self._screen_to_image_bbox(screen_bbox)
            preview_img = self._cleaned_image_full.crop((x1, y1, x2, y2)).resize(
                (box_width_img, box_height_img), Image.LANCZOS
            ).convert("RGBA")
        else:
            preview_img = Image.new("RGBA", (box_width_img, box_height_img), (255, 255, 255, 235))
        draw = ImageDraw.Draw(preview_img)

        # Un balloon "ricreato" (fill/bordo manuali) non ha ancora nulla di
        # disegnato nel cleaned.png alla nuova posizione: senza questo, la
        # preview durante il trascinamento mostra solo il testo che fluttua
        # sullo sfondo, senza la sagoma del balloon sotto (vedi
        # _apply_manual_fill in render.py, applicato solo al render finale).
        draw_shape = draw.ellipse if is_ellipse else draw.rectangle
        box_edge = (0, 0, box_width_img - 1, box_height_img - 1)
        if fill_hex:
            draw_shape(box_edge, fill=fill_hex)
        if has_border:
            draw_shape((1, 1, box_width_img - 2, box_height_img - 2), outline="black", width=3)

        if text and text != "-":
            if is_ellipse:
                tw = box_width_img * render._INV_SQRT2
                th = box_height_img * render._INV_SQRT2
                tx = (box_width_img - tw) / 2
                ty = (box_height_img - th) / 2
            else:
                tx, ty, tw, th = 0, 0, box_width_img, box_height_img
            try:
                # Una size impostata a mano (Dim. nell'editor) va rispettata
                # anche qui: senza questo, trascinare un box con font
                # preimpostato mostrava un fitting automatico che ignorava
                # la scelta dell'utente, mentre il render finale la
                # rispettava gia' (vedi manual_size in _draw_text_in_bbox).
                manual_size = det.get("manual_font_size")
                min_size = manual_size or render_cfg["min_font_size"]
                max_size = manual_size or render_cfg["max_font_size"]
                line_spacing = det.get("manual_line_spacing") or render_cfg["line_spacing"]
                lines, font, _overflowed = render._fit_text_to_box(
                    draw, text, tw, th, font_path,
                    min_size, max_size, line_spacing,
                )
                render._draw_lines_centered(
                    draw, lines, font, tx, ty, tw, th,
                    line_spacing, text_color, align=det.get("manual_align") or "center",
                )
            except Exception:
                pass

        preview_img = preview_img.resize((width, height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(preview_img)

        old = self._drag_previews.get(idx)
        if old:
            self.canvas.delete(old[1])
        item_id = self.canvas.create_image(sx1, sy1, anchor="nw", image=photo, tags=("drag_preview",))
        self.canvas.tag_raise("bbox")
        self.canvas.tag_raise("handle")
        self._drag_previews[idx] = (photo, item_id)

    def _commit_bbox_changes(self, idx: int):
        for box in self._review_boxes:
            if box["idx"] == idx:
                sx1, sy1, sx2, sy2 = box["bbox_screen"]
                offset_x, offset_y = self._img_offset
                scale = self._review_scale
                x1 = int((sx1 - offset_x) / scale)
                y1 = int((sy1 - offset_y) / scale)
                x2 = int((sx2 - offset_x) / scale)
                y2 = int((sy2 - offset_y) / scale)
                img = Image.open(self._current_image_path)
                x1 = max(0, min(x1, img.width - 1))
                y1 = max(0, min(y1, img.height - 1))
                x2 = max(x1 + 1, min(x2, img.width))
                y2 = max(y1 + 1, min(y2, img.height))
                self._review_data["detections"][idx]["bbox"] = [x1, y1, x2, y2]
                self._review_data["detections"][idx]["_modified"] = True
                self._review_data["detections"][idx]["_bbox_modified"] = True
                # Bbox toccato a mano in Revisione: da qui in poi il render
                # deve rispettare queste dimensioni invece di ri-misurare da
                # solo l'estensione reale del balloon (che altrimenti
                # ignorerebbe un bbox allargato manualmente). Persiste nel
                # translated.json, non viene rimosso al salvataggio come
                # _modified/_bbox_modified.
                self._review_data["detections"][idx]["manual_text_box"] = True
                # balloon_split/manual_respect_shape dicono al render di
                # ignorare il rettangolo e seguire invece la maschera del
                # balloon (per non sconfinare nella strozzatura tra due lobi
                # fusi, vedi respect_shape in render.py). Una volta che
                # l'utente sposta o ridimensiona il box a mano, quella
                # maschera non corrisponde piu' alla posizione scelta: senza
                # spegnere questi flag il render continuerebbe a piazzare il
                # testo sulla vecchia sagoma, ignorando lo spostamento (bug:
                # "il testo torna al setup originale").
                self._review_data["detections"][idx]["balloon_split"] = False
                self._review_data["detections"][idx]["manual_respect_shape"] = False
                return

    def _on_canvas_double_click(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        for box in self._review_boxes:
            sx1, sy1, sx2, sy2 = box["bbox_screen"]
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                self._open_inline_editor(box["idx"], sx1, sy1, sx2, sy2)
                return

    MIN_EDITOR_WIDTH = 320
    MIN_EDITOR_HEIGHT = 580

    TAIL_DIRECTION_OPTIONS = {
        "Nessuno": "",
        "Su": "n",
        "Su-destra": "ne",
        "Destra": "e",
        "Giù-destra": "se",
        "Giù": "s",
        "Giù-sinistra": "sw",
        "Sinistra": "w",
        "Su-sinistra": "nw",
    }

    def _open_inline_editor(self, idx: int, sx1, sy1, sx2, sy2):
        self._close_inline_editor()
        det = self._review_data["detections"][idx]
        current_text = det.get("testo_tradotto", "")
        original_text = det.get("testo_originale", "")

        # Larghezza fissa e comoda (non piu' legata alla larghezza del
        # balloon, spesso stretta): l'editor ora affianca l'immagine invece
        # di sovrapporla, quindi puo' sempre usare tutto lo spazio che gli
        # serve per mostrare la formattazione senza righe tagliate.
        width = self.MIN_EDITOR_WIDTH
        height = max(sy2 - sy1, self.MIN_EDITOR_HEIGHT)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        height = min(height, max(200, canvas_h - 10))

        # Preferisce affiancare il popup alla pagina (a destra, o a sinistra
        # se a destra non c'e' posto) invece di piazzarlo sopra il balloon:
        # a zoom ridotto c'e' quasi sempre canvas vuoto ai lati della
        # pagina, e coprire il disegno rende piu' difficile valutare il
        # risultato mentre si modifica il testo. Ricade sul centrare sul
        # balloon (comportamento originale) solo se non c'e' spazio libero
        # ne' a destra ne' a sinistra (pagina che riempie tutto il canvas).
        margin = 14
        img_left, img_top = self._img_offset
        img_right = img_left + (self._review_photo.width() if self._review_photo else 0)
        if img_right + margin + width <= canvas_w:
            x = img_right + margin
        elif img_left - margin - width >= 0:
            x = img_left - margin - width
        else:
            x = (sx1 + sx2) / 2 - width / 2

        y = (sy1 + sy2) / 2 - height / 2
        x = max(0, min(x, canvas_w - width))
        y = max(0, min(y, canvas_h - height))

        editor_frame = Frame(self.canvas, bg="white", bd=2, relief="solid")
        editor_frame.place(x=x, y=y, width=width, height=height)

        # Troncato: un originale lungo su un popup stretto va a capo su
        # molte righe e, sommato alle righe fisse sotto (font/codino/bottoni),
        # puo' schiacciare a zero lo spazio rimasto per il text_widget
        # editabile, che lo eredita per ultimo (fill=BOTH, expand=True).
        original_preview = original_text or "(vuoto)"
        if len(original_preview) > 140:
            original_preview = original_preview[:140].rstrip() + "…"

        if det.get("ocr_empty_suspect") and not current_text.strip():
            Label(
                editor_frame,
                text="\u26a0 L'OCR non ha letto nulla ma nel balloon c'e' del testo:\n"
                      "trascrivilo qui, oppure scrivi - se e' un'insegna o un'onomatopea",
                bg="#ffe5e5", fg="#b00020", font=("Arial", 8, "bold"),
                anchor=W, justify=LEFT, wraplength=width - 10,
            ).pack(fill=X, side=TOP)
        elif det.get("_overflow") and current_text.strip():
            Label(
                editor_frame,
                text="\u2702 Nel render questo testo non ci stava ed e' stato troncato:\n"
                      "accorcialo, allarga il box o imposta una dimensione font piu' piccola",
                bg="#fff0e0", fg="#a35200", font=("Arial", 8, "bold"),
                anchor=W, justify=LEFT, wraplength=width - 10,
            ).pack(fill=X, side=TOP)

        original_frame = Frame(editor_frame, bg="#eeeeee")
        original_frame.pack(fill=X, side=TOP)
        Label(
            original_frame, text="Originale:", bg="#eeeeee", fg="gray30",
            font=("Arial", 8, "bold"), anchor=W
        ).pack(fill=X, padx=4, pady=(2, 0))
        Label(
            original_frame, text=original_preview, bg="#eeeeee", fg="gray30",
            font=("Arial", 9, "italic"), anchor=W, justify=LEFT, wraplength=max(1, int(width) - 8)
        ).pack(fill=X, padx=4, pady=(0, 2))

        text_widget = Text(editor_frame, wrap="word", font=("Arial", 10), bg="white", fg="black")

        # Bottoni e selettore font vengono ancorati al fondo (side=BOTTOM)
        # PRIMA di impacchettare la text area espandibile: cosi' ottengono
        # sempre lo spazio richiesto e non vengono tagliati fuori dal popup
        # quando il balloon (quindi il popup) e' piccolo.
        btn_frame = Frame(editor_frame, bg="white")
        btn_frame.pack(fill=X, side=BOTTOM)
        ttk.Button(btn_frame, text="✓ OK", command=lambda: self._confirm_inline_editor(idx, text_widget)).pack(side=LEFT, padx=2)
        ttk.Button(btn_frame, text="✗ Annulla", command=self._close_inline_editor).pack(side=LEFT, padx=2)

        fill_frame = Frame(editor_frame, bg="white")
        fill_frame.pack(fill=X, side=BOTTOM, padx=2)

        fill_color_holder = {"hex": det.get("manual_fill_color") or "#ffffff"}

        def _pick_fill_color():
            _rgb, hexval = colorchooser.askcolor(
                color=fill_color_holder["hex"], title="Colore di riempimento", parent=self.win
            )
            if hexval:
                fill_color_holder["hex"] = hexval
                color_btn.config(bg=hexval)

        color_btn = Button(
            fill_frame, text="  ", width=2, bg=fill_color_holder["hex"],
            command=_pick_fill_color, relief="solid", bd=1
        )
        color_btn.pack(side=LEFT, padx=(2, 4))

        fill_enabled_var = BooleanVar(value=bool(det.get("manual_fill_color")))
        ttk.Checkbutton(fill_frame, text="Riempi box", variable=fill_enabled_var).pack(side=LEFT)
        border_var = BooleanVar(value=bool(det.get("manual_border")))
        ttk.Checkbutton(fill_frame, text="Bordo nero", variable=border_var).pack(side=LEFT, padx=(10, 0))

        text_color_frame = Frame(editor_frame, bg="white")
        text_color_frame.pack(fill=X, side=BOTTOM, padx=2)

        text_color_holder = {"hex": det.get("manual_text_color") or "#000000"}

        def _pick_text_color():
            _rgb, hexval = colorchooser.askcolor(
                color=text_color_holder["hex"], title="Colore del testo", parent=self.win
            )
            if hexval:
                text_color_holder["hex"] = hexval
                text_color_btn.config(bg=hexval)

        text_color_btn = Button(
            text_color_frame, text="  ", width=2, bg=text_color_holder["hex"],
            command=_pick_text_color, relief="solid", bd=1
        )
        text_color_btn.pack(side=LEFT, padx=(2, 4))

        text_color_enabled_var = BooleanVar(value=bool(det.get("manual_text_color")))
        ttk.Checkbutton(
            text_color_frame, text="Colore testo manuale (altrimenti auto nero/bianco)",
            variable=text_color_enabled_var,
        ).pack(side=LEFT)

        # Riga propria a piena larghezza: nella riga di fill_frame sopra non
        # c'e' mai spazio a sufficienza sui balloon piccoli (popup largo
        # anche solo MIN_EDITOR_WIDTH=220px), quindi un bottone aggiunto li'
        # finirebbe tagliato fuori e invisibile.
        recreate_frame = Frame(editor_frame, bg="white")
        recreate_frame.pack(fill=X, side=BOTTOM, padx=2, pady=(2, 0))

        recreate_shape_var = BooleanVar(value=det.get("manual_balloon_shape") == "ellipse")

        tail_frame = Frame(editor_frame, bg="white")
        tail_frame.pack(fill=X, side=BOTTOM, padx=2)
        Label(tail_frame, text="Codino:", bg="white", fg="gray30", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(2, 4))
        tail_label_by_value = {v: k for k, v in self.TAIL_DIRECTION_OPTIONS.items()}
        tail_var = StringVar(value=tail_label_by_value.get(det.get("manual_balloon_tail") or "", "Nessuno"))
        ttk.Combobox(
            tail_frame, textvariable=tail_var, state="readonly",
            values=list(self.TAIL_DIRECTION_OPTIONS.keys()), width=14,
        ).pack(side=LEFT, fill=X, expand=True)

        def _recreate_balloon():
            # Preimposta i controlli con i valori tipici di un balloon
            # (bianco pieno + contorno nero + codino), da confermare con
            # OK: usato quando la pulizia automatica ha cancellato l'intera
            # sagoma del balloon (non solo il testo) e va ridisegnata a
            # mano. recreate_shape_var forza la forma a ellisse in fase di
            # render: senza, un box gia' ridimensionato a mano
            # (manual_text_box, tipico proprio di questo flusso: si allarga
            # il box per farlo combaciare con l'area del balloon sparito)
            # verrebbe riempito come rettangolo esatto invece che come
            # balloon ovale.
            fill_color_holder["hex"] = "#ffffff"
            color_btn.config(bg="#ffffff")
            fill_enabled_var.set(True)
            border_var.set(True)
            recreate_shape_var.set(True)
            if tail_var.get() == "Nessuno":
                tail_var.set("Giù")

        ttk.Button(recreate_frame, text="🔁 Ricrea balloon (bianco + bordo)", command=_recreate_balloon).pack(fill=X)

        shape_row = Frame(recreate_frame, bg="white")
        shape_row.pack(fill=X)
        ttk.Checkbutton(
            shape_row, text="Ellisse", variable=recreate_shape_var,
        ).pack(side=LEFT)
        ttk.Button(
            shape_row, text="✂ Dividi in due",
            command=lambda: self._split_inline_editor(idx, text_widget),
        ).pack(side=RIGHT)

        font_frame = Frame(editor_frame, bg="white")
        font_frame.pack(fill=X, side=BOTTOM, padx=2)
        auto_tag = " (auto)" if det.get("font_auto") else ""
        Label(
            font_frame, text=f"Font{auto_tag}:", bg="white", fg="gray30",
            font=("Arial", 8, "bold"), anchor=W
        ).pack(side=LEFT, padx=(2, 4))
        font_var = StringVar()
        available_fonts = self._get_available_fonts()
        current_font_path = det.get("font_path")
        current_font_name = next(
            (name for name, path in available_fonts.items() if path == current_font_path),
            next(iter(available_fonts), ""),
        )
        font_var.set(current_font_name)
        font_combo = ttk.Combobox(
            font_frame, textvariable=font_var, state="readonly",
            values=list(available_fonts.keys()), width=18,
        )
        font_combo.pack(side=LEFT, fill=X, expand=True, padx=(0, 2))
        font_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._set_balloon_font(idx, available_fonts.get(font_var.get())),
        )

        # Formattazione manuale, su piu' righe strette invece che una riga
        # sola larga: il popup ha larghezza fissa (quella del balloon, anche
        # solo MIN_EDITOR_WIDTH su un balloon piccolo) e non si allarga per
        # contenere i widget — stiparli tutti su una riga li taglia fuori
        # invece di andare a capo (limite di .pack in un Frame a larghezza
        # fissa). Se lasciati vuoti (font_size_var == "") il rendering torna
        # al fitting automatico (vedi _has_manual_style in render.py).
        apply_all_frame = Frame(editor_frame, bg="white")
        apply_all_frame.pack(fill=X, side=BOTTOM, padx=2, pady=(2, 0))
        ttk.Button(
            apply_all_frame, text="Imposta formattazione per tutti i balloon",
            command=lambda: self._apply_style_to_all(idx)
        ).pack(fill=X)

        outline_frame = Frame(editor_frame, bg="white")
        outline_frame.pack(fill=X, side=BOTTOM, padx=2)

        outline_color_holder = {"hex": det.get("manual_outline_color") or "#000000"}

        def _pick_outline_color():
            _rgb, hexval = colorchooser.askcolor(
                color=outline_color_holder["hex"], title="Colore contorno", parent=self.win
            )
            if hexval:
                outline_color_holder["hex"] = hexval
                outline_color_btn.config(bg=hexval)

        outline_enable_var = BooleanVar(value=bool(det.get("manual_outline_enable")))
        ttk.Checkbutton(outline_frame, text="Contorno", variable=outline_enable_var).pack(side=LEFT)
        outline_color_btn = Button(
            outline_frame, text="  ", width=2, bg=outline_color_holder["hex"],
            command=_pick_outline_color, relief="solid", bd=1
        )
        outline_color_btn.pack(side=LEFT, padx=(4, 4))
        Label(outline_frame, text="Sp.:", bg="white", fg="gray30", font=("Arial", 8, "bold")).pack(side=LEFT)
        outline_width_var = StringVar(value=str(det.get("manual_outline_width") or "1.0"))
        ttk.Entry(outline_frame, textvariable=outline_width_var, width=4).pack(side=LEFT, padx=(2, 0))

        style_frame = Frame(editor_frame, bg="white")
        style_frame.pack(fill=X, side=BOTTOM, padx=2, pady=(2, 0))
        Label(style_frame, text="Stile:", bg="white", fg="gray30", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(2, 4))
        bold_var = BooleanVar(value=bool(det.get("manual_bold")))
        ttk.Checkbutton(style_frame, text="Grassetto", variable=bold_var).pack(side=LEFT)
        italic_var = BooleanVar(value=bool(det.get("manual_italic")))
        ttk.Checkbutton(style_frame, text="Corsivo", variable=italic_var).pack(side=LEFT, padx=(6, 0))
        underline_var = BooleanVar(value=bool(det.get("manual_underline")))
        ttk.Checkbutton(style_frame, text="Sottolin.", variable=underline_var).pack(side=LEFT, padx=(6, 0))

        align_frame = Frame(editor_frame, bg="white")
        align_frame.pack(fill=X, side=BOTTOM, padx=2)
        Label(align_frame, text="Allineamento:", bg="white", fg="gray30", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(2, 4))
        align_var = StringVar(value=det.get("manual_align") or "center")
        for label, value in (("Sinistra", "left"), ("Centro", "center"), ("Destra", "right")):
            ttk.Radiobutton(align_frame, text=label, variable=align_var, value=value).pack(side=LEFT)

        # Vale per QUALSIASI balloon con sagoma reale (balloon_source
        # yolov8seg), non solo per i doppi/a clessidra: qualunque intervento
        # manuale (size, interlinea, stile, box ridimensionato) fa saltare il
        # fitting a maschera e passare al rettangolo pieno del bbox, che non
        # e' la curva del balloon — sui doppi sconfina nel "collo" tra i due
        # lobi, su un balloon singolo ovale sconfina agli angoli arrotondati
        # (vedi commento su manual_respect_shape in render.py).
        #
        # Per questo il flag parte ATTIVO sui balloon con sagoma reale: e' il
        # comportamento giusto nella stragrande maggioranza dei casi. Fa
        # eccezione il box gia' spostato/ridimensionato a mano
        # (manual_text_box): li' la maschera non corrisponde piu' alla
        # posizione scelta, e seguirla rimetterebbe il testo sulla vecchia
        # sagoma (vedi _commit_bbox_changes, che infatti spegne il flag).
        shape_fit_frame = Frame(editor_frame, bg="white")
        shape_fit_frame.pack(fill=X, side=BOTTOM, padx=2)
        _shape_fit_available = bool(
            det.get("balloon_source") == "yolov8seg" and det.get("mask_path")
        )
        _shape_fit_default = _shape_fit_available and not det.get("manual_text_box")
        respect_shape_var = BooleanVar(
            value=bool(det.get("manual_respect_shape")) or _shape_fit_default
        )
        ttk.Checkbutton(
            shape_fit_frame, text="Segui sagoma balloon",
            variable=respect_shape_var,
            state=(NORMAL if _shape_fit_available else DISABLED),
        ).pack(side=LEFT)
        if not _shape_fit_available:
            Label(
                shape_fit_frame, text="(nessuna sagoma per questo balloon)",
                bg="white", fg="gray50", font=("Arial", 8, "italic"),
            ).pack(side=LEFT, padx=(4, 0))

        size_frame = Frame(editor_frame, bg="white")
        size_frame.pack(fill=X, side=BOTTOM, padx=2)
        Label(size_frame, text="Dim.:", bg="white", fg="gray30", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(2, 2))
        font_size_var = StringVar(value=str(det.get("manual_font_size") or ""))
        ttk.Entry(size_frame, textvariable=font_size_var, width=4).pack(side=LEFT)
        # Campo vuoto = fitting automatico (vedi commento sopra): mostra qui
        # a titolo informativo la size scelta dall'ultimo render, cosi' non
        # sembra "vuoto/senza valore" mentre in realta' un font e' gia'
        # applicato. Non precompila l'Entry: farlo trasformerebbe
        # silenziosamente ogni balloon in un box a size manuale fissa.
        current_size = det.get("_rendered_font_size")
        if current_size and not det.get("manual_font_size"):
            Label(
                size_frame, text=f"(attuale: {current_size})", bg="white", fg="gray50",
                font=("Arial", 8, "italic"),
            ).pack(side=LEFT, padx=(4, 0))
        Label(size_frame, text="Interlinea:", bg="white", fg="gray30", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(8, 2))
        line_spacing_var = StringVar(value=str(det.get("manual_line_spacing") or ""))
        ttk.Entry(size_frame, textvariable=line_spacing_var, width=4).pack(side=LEFT)

        text_widget.pack(fill=BOTH, expand=True, padx=2, pady=2)
        text_widget.insert("1.0", current_text)
        text_widget.focus()
        text_widget.tag_add(SEL, "1.0", END)

        self._inline_editor = {
            "frame": editor_frame,
            "text_widget": text_widget,
            "idx": idx,
            "fill_enabled_var": fill_enabled_var,
            "border_var": border_var,
            "fill_color_holder": fill_color_holder,
            "recreate_shape_var": recreate_shape_var,
            "tail_var": tail_var,
            "text_color_holder": text_color_holder,
            "text_color_enabled_var": text_color_enabled_var,
            "font_size_var": font_size_var,
            "line_spacing_var": line_spacing_var,
            "align_var": align_var,
            "bold_var": bold_var,
            "italic_var": italic_var,
            "underline_var": underline_var,
            "outline_enable_var": outline_enable_var,
            "outline_color_holder": outline_color_holder,
            "outline_width_var": outline_width_var,
            "respect_shape_var": respect_shape_var,
        }
        self._select_box(idx)

        text_widget.bind("<Return>", lambda e: self._confirm_inline_editor(idx, text_widget))
        text_widget.bind("<Escape>", lambda e: self._close_inline_editor())

    def _get_available_fonts(self) -> dict:
        """Nome -> percorso dei font .ttf/.otf in fonts_dir, con cache."""
        if getattr(self, "_available_fonts_cache", None) is not None:
            return self._available_fonts_cache
        if self._render_cfg is None:
            try:
                self._render_cfg = load_cfg()["rendering"]
            except Exception:
                self._available_fonts_cache = {}
                return self._available_fonts_cache
        fonts_dir = paths.resolve(self._render_cfg.get("fonts_dir", "fonts"))
        fonts = {}
        if fonts_dir.exists():
            for f in sorted(list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf"))):
                fonts[f.stem] = str(f)
        self._available_fonts_cache = fonts
        return fonts

    def _set_balloon_font(self, idx: int, font_path: str):
        """Override manuale del font per un balloon: da qui in poi la
        suggestione automatica di ocr.py non lo tocca piu' (font_auto=False)."""
        if not font_path:
            return
        det = self._review_data["detections"][idx]
        det["font_path"] = font_path
        det["font_auto"] = False
        det["_modified"] = True
        self._mark_dirty()
        self._render_canvas()

    @staticmethod
    def _parse_positive_int(raw: str) -> int | None:
        try:
            v = int(float(raw.strip()))
            return v if v > 0 else None
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_positive_float(raw: str) -> float | None:
        try:
            v = float(raw.strip())
            return v if v > 0 else None
        except (ValueError, AttributeError):
            return None

    def _style_from_editor(self, editor: dict) -> dict:
        """Estrae dai campi correnti dell'editor (non ancora confermati) i
        soli campi di formattazione manuale, condivisi da OK e da 'Imposta
        per tutti'."""
        align = editor["align_var"].get()
        outline_enabled = editor["outline_enable_var"].get()
        return {
            "manual_font_size": self._parse_positive_int(editor["font_size_var"].get()),
            "manual_line_spacing": self._parse_positive_float(editor["line_spacing_var"].get()),
            "manual_align": align if align != "center" else None,
            "manual_bold": editor["bold_var"].get(),
            "manual_italic": editor["italic_var"].get(),
            "manual_underline": editor["underline_var"].get(),
            "manual_outline_enable": outline_enabled,
            "manual_outline_color": editor["outline_color_holder"]["hex"] if outline_enabled else None,
            "manual_outline_width": self._parse_positive_float(editor["outline_width_var"].get()) or 1.0,
            "manual_respect_shape": editor["respect_shape_var"].get(),
        }

    def _confirm_inline_editor(self, idx: int, text_widget: Text):
        new_text = text_widget.get("1.0", END).strip()
        det = self._review_data["detections"][idx]
        if new_text == "-":
            det["testo_originale"] = ""
            det["testo_tradotto"] = ""
        else:
            det["testo_tradotto"] = new_text
        editor = self._inline_editor
        if editor:
            det["manual_fill_color"] = (
                editor["fill_color_holder"]["hex"] if editor["fill_enabled_var"].get() else None
            )
            det["manual_border"] = editor["border_var"].get()
            det["manual_balloon_shape"] = "ellipse" if editor["recreate_shape_var"].get() else None
            det["manual_balloon_tail"] = self.TAIL_DIRECTION_OPTIONS.get(editor["tail_var"].get()) or None
            det["manual_text_color"] = (
                editor["text_color_holder"]["hex"] if editor["text_color_enabled_var"].get() else None
            )
            det.update(self._style_from_editor(editor))
        det["_modified"] = True
        # L'utente ha guardato il balloon e deciso (trascritto, oppure "-" per
        # lasciarlo com'e'): la segnalazione automatica ha esaurito il suo
        # compito e non deve ricomparire ad ogni ricaricamento.
        if det.get("ocr_empty_suspect"):
            det["ocr_empty_suspect"] = False
        self._close_inline_editor()
        self._mark_dirty()
        self._render_canvas()

    def _apply_style_to_all(self, idx: int):
        """'Imposta per tutti': copia SOLO la formattazione (font + i campi
        manual_* di stile) del balloon corrente su tutti gli altri, senza
        toccare testo/bbox/riempimento/bordo/forma — quelli restano
        specifici del singolo balloon."""
        editor = self._inline_editor
        if not editor or not self._review_data:
            return
        style = self._style_from_editor(editor)
        style["font_path"] = self._review_data["detections"][idx].get("font_path")
        style["font_auto"] = False
        for det in self._review_data["detections"]:
            det.update(style)
            det["_modified"] = True
        self._mark_dirty()
        self.status.config(text=t("Formattazione applicata a tutti i balloon della pagina."), foreground="black")

    def _find_balloon_split(self, det: dict) -> tuple[tuple[list[int], list[int]], float] | None:
        """Cerca la vera strozzatura tra i due lobi nella maschera del
        balloon (stesso algoritmo di render.py/balloon_shape.py usato per il
        fitting del testo a doppio lobo), per allineare il taglio di
        'Dividi in due' al collo reale invece del punto medio geometrico del
        bbox. Su un balloon molto asimmetrico (lobo piccolo + lobo grande,
        es. una didascalia "NO." fusa con un balloon di dialogo molto piu'
        grande) il punto medio taglia nel posto sbagliato e mette meta' del
        testo nel lobo piccolo, dove non c'entra. Ritorna None se non c'e'
        una maschera reale o nessuna strozzatura netta e' rilevabile: il
        chiamante ricade sul taglio a meta'."""
        mask_path = det.get("mask_path")
        if det.get("balloon_source") != "yolov8seg" or not mask_path:
            return None
        mask = render._load_mask_array(mask_path)
        if mask is None:
            return None

        # Analizza il profilo sulla sagoma NATIVA della maschera (il suo
        # bounding box reale), non sul bbox corrente del balloon: quest'
        # ultimo puo' essere stato ridimensionato a mano in Revisione, e un
        # bbox che non combacia piu' con l'estensione vera della maschera
        # fa saltare/spostare il profilo di larghezza, perdendo la
        # strozzatura reale. Il bbox corrente viene riusato solo dopo, per
        # ritagliare le due meta' risultanti.
        mys, mxs = np.where(mask > 0)
        if len(mys) == 0:
            return None
        nx1, ny1, nx2, ny2 = int(mxs.min()), int(mys.min()), int(mxs.max()) + 1, int(mys.max()) + 1
        sub = mask[ny1:ny2, nx1:nx2] > 0
        if sub.size == 0:
            return None
        x1, y1, x2, y2 = det["bbox"]

        row_widths = render._smooth_widths(sub.sum(axis=1).tolist())
        col_widths = render._smooth_widths(sub.sum(axis=0).tolist())

        def _trimmed_profile(widths: list[int]) -> tuple[int, list[int]] | None:
            # La punta appuntita di una coda/codino ha larghezza vicina a
            # zero, spesso PIU' bassa della vera strozzatura tra i due lobi:
            # senza scartarla, find_waist trova come minimo globale la coda
            # invece del collo reale (osservato su balloon con codino lungo
            # e assottigliato). Stessa soglia (12% del massimo) usata da
            # render.py._fit_text_to_mask per lo stesso motivo.
            max_w = max(widths, default=0)
            if max_w <= 0:
                return None
            threshold = max(4, int(max_w * 0.12))
            usable = [i for i, w in enumerate(widths) if w >= threshold]
            if not usable:
                return None
            top, bottom = usable[0], usable[-1]
            return top, widths[top:bottom + 1]

        candidates = []
        row_trim = _trimmed_profile(row_widths)
        if row_trim is not None:
            row_offset, row_trimmed = row_trim
            row_waist = render._find_waist(row_trimmed, 0)
            if row_waist is not None and max(row_trimmed, default=0) > 0:
                candidates.append(("row", row_offset + row_waist, row_trimmed[row_waist] / max(row_trimmed)))
        col_trim = _trimmed_profile(col_widths)
        if col_trim is not None:
            col_offset, col_trimmed = col_trim
            col_waist = render._find_waist(col_trimmed, 0)
            if col_waist is not None and max(col_trimmed, default=0) > 0:
                candidates.append(("col", col_offset + col_waist, col_trimmed[col_waist] / max(col_trimmed)))
        if not candidates:
            return None
        # Se entrambi gli assi trovano una strozzatura, usa quella piu'
        # pronunciata (rapporto larghezza-minima/larghezza-massima piu' basso).
        axis, cut, _ratio = min(candidates, key=lambda c: c[2])

        if axis == "row":
            cap_a = sum(row_widths[:cut]) or 1
            cap_b = sum(row_widths[cut:]) or 1
            # cut e' un indice nel riferimento della maschera nativa (ny1);
            # riportato in coordinate assolute e poi vincolato al bbox
            # corrente (che puo' essere piu' stretto/largo dopo un
            # ridimensionamento a mano), cosi' il taglio resta dentro l'area
            # che l'utente sta effettivamente editando.
            cut_y = max(y1, min(y2, ny1 + cut))
            bbox_a, bbox_b = [x1, y1, x2, cut_y], [x1, cut_y, x2, y2]
        else:
            cap_a = sum(col_widths[:cut]) or 1
            cap_b = sum(col_widths[cut:]) or 1
            cut_x = max(x1, min(x2, nx1 + cut))
            bbox_a, bbox_b = [x1, y1, cut_x, y2], [cut_x, y1, x2, y2]

        # Le parole si dividono in proporzione allo spazio reale di ciascun
        # lobo (somma delle larghezze di riga/colonna), non 50/50: un lobo
        # piccolo deve ricevere proporzionalmente meno testo.
        word_ratio = cap_a / (cap_a + cap_b)
        return (bbox_a, bbox_b), word_ratio

    def _split_inline_editor(self, idx: int, text_widget: Text):
        """Spezza il balloon in due box indipendenti, ciascuno con la sua
        quota di testo: utile per un balloon con forma irregolare (es. a
        clessidra) dove l'algoritmo di fitting su un unico box non riesce a
        posizionare bene il testo, e conviene invece controllare a mano
        posizione/dimensione di ciascuna meta'. Il taglio segue la vera
        strozzatura della maschera quando rilevabile (_find_balloon_split),
        altrimenti ricade sul punto medio del bbox con testo diviso a meta'."""
        det = self._review_data["detections"][idx]
        text = text_widget.get("1.0", END).strip()
        x1, y1, x2, y2 = det["bbox"]
        width, height = x2 - x1, y2 - y1

        split = self._find_balloon_split(det)
        if split is not None:
            (bbox_a, bbox_b), word_ratio = split
        else:
            # Taglia lungo l'asse piu' lungo: un box piu' alto che largo e'
            # quasi sempre due lobi impilati verticalmente (clessidra), uno
            # piu' largo che alto due lobi affiancati.
            if height >= width:
                mid = int(round((y1 + y2) / 2))
                bbox_a, bbox_b = [x1, y1, x2, mid], [x1, mid, x2, y2]
            else:
                mid = int(round((x1 + x2) / 2))
                bbox_a, bbox_b = [x1, y1, mid, y2], [mid, y1, x2, y2]
            word_ratio = 0.5

        words = text.split()
        if len(words) >= 2:
            split_at = round(len(words) * word_ratio)
            split_at = max(1, min(len(words) - 1, split_at))
            text_a, text_b = " ".join(words[:split_at]), " ".join(words[split_at:])
        else:
            text_a, text_b = text, ""

        existing_ids = [d.get("balloon_id") for d in self._review_data["detections"] if isinstance(d.get("balloon_id"), int)]
        new_id = (max(existing_ids) + 1) if existing_ids else len(self._review_data["detections"])

        new_det = dict(det)
        new_det["bbox"] = bbox_b
        new_det["testo_tradotto"] = text_b
        new_det["balloon_id"] = new_id
        new_det["manual_text_box"] = True
        # Il flag "segui sagoma" si riferisce alla maschera INTERA del
        # balloon fuso originale: dopo lo split ciascuna meta' ha un bbox
        # piu' piccolo che non corrisponde piu' a quella maschera, quindi va
        # spento per non far ripescare per errore la sagoma sbagliata
        # (vedi manual_respect_shape in render.py).
        new_det["manual_respect_shape"] = False
        new_det["_modified"] = True
        new_det["_overflow"] = False

        det["bbox"] = bbox_a
        det["testo_tradotto"] = text_a
        det["manual_text_box"] = True
        det["manual_respect_shape"] = False
        # Il codino (se presente) resta solo sulla seconda meta': altrimenti,
        # con "ricrea balloon" attivo, verrebbero disegnati due codini
        # separati sulle due ellissi.
        det["manual_balloon_tail"] = None
        det["_modified"] = True
        det["_bbox_modified"] = True

        self._review_data["detections"].insert(idx + 1, new_det)

        self._close_inline_editor()
        self._mark_dirty()
        self._render_canvas()

    def _close_inline_editor(self):
        if self._inline_editor:
            self._inline_editor["frame"].destroy()
            self._inline_editor = None

    def _mark_dirty(self):
        self._review_dirty = True
        self.btn_regenerate.config(state=NORMAL)
        self.status.config(
            text=t("⚠️ Modifiche in sospeso. Clicca 'Rigenera pagina' per applicarle."),
            foreground="orange"
        )

    def _save_and_regenerate(self):
        try:
            cfg = load_cfg()
            work_dir = Path(cfg["paths"]["work_dir"])
            output_dir = Path(cfg["paths"]["output_dir"])
            input_dir = Path(cfg["paths"]["input_dir"])
            comic = self._review_comic
            page_id = self._review_page_id

            comic_work_dir = work_dir / comic if comic else work_dir
            comic_output_dir = output_dir / comic if comic else output_dir
            comic_input_dir = input_dir / comic if comic else input_dir

            for det in self._review_data["detections"]:
                det.pop("_modified", None)
                det.pop("_bbox_modified", None)

            translated_path = comic_work_dir / page_id / "translated.json"

            # Se un balloon passa da "testo da tradurre" a onomatopea (o
            # viceversa) cambia l'insieme dei balloon da pulire con ComfyUI,
            # anche se il bbox non si e' spostato: senza questo controllo la
            # pipeline riuserebbe cleaned.png cosi' com'e', lasciando quel
            # balloon con la vecchia pulizia mai aggiornata. Spostare o
            # ridimensionare un box invece non richiede di ripassare da
            # ComfyUI: e' solo un cambio di posizionamento del testo dentro
            # un balloon gia' pulito.
            def _wants_mask(text):
                t = (text or "").strip()
                return bool(t) and t != "-"

            any_mask_eligibility_changed = False
            if translated_path.exists():
                try:
                    with open(translated_path, "r", encoding="utf-8") as f:
                        old_detections = json.load(f).get("detections", [])
                    # Confronto per balloon_id, non per posizione in lista: uno
                    # split (vedi "Dividi in due") aggiunge un balloon_id nuovo
                    # e sposta gli indici di quelli dopo, quindi un confronto
                    # posizionale (o per lunghezza) segnalerebbe un cambiamento
                    # anche senza che nessun balloon abbia davvero cambiato
                    # bisogno di pulizia, facendo ripartire ComfyUI a vuoto.
                    old_wants_by_id = {d.get("balloon_id"): _wants_mask(d.get("testo_tradotto")) for d in old_detections}
                    new_wants_by_id = {
                        d.get("balloon_id"): _wants_mask(d.get("testo_tradotto"))
                        for d in self._review_data["detections"]
                    }
                    common_ids = old_wants_by_id.keys() & new_wants_by_id.keys()
                    any_mask_eligibility_changed = any(
                        old_wants_by_id[bid] != new_wants_by_id[bid] for bid in common_ids
                    )
                except Exception:
                    any_mask_eligibility_changed = True

            with open(translated_path, "w", encoding="utf-8") as f:
                json.dump(self._review_data, f, ensure_ascii=False, indent=2)

            self.status.config(text=t(" Rigenerazione in corso..."), foreground="orange")
            self.win.update_idletasks()
            self.app._log(f"Rigenerazione pagina {page_id} ({comic or 'root'})...")

            cleaned_path = comic_work_dir / page_id / "cleaned.png"
            if cleaned_path.exists() and not any_mask_eligibility_changed:
                self.app._log(f"Uso pagina già pulita: {cleaned_path}")
            else:
                exts = {".png", ".jpg", ".jpeg", ".webp"}
                candidates = [p for p in comic_input_dir.iterdir() if p.stem == page_id and p.suffix.lower() in exts]
                if not candidates:
                    messagebox.showerror(t("Errore"), f"Immagine originale non trovata per {page_id}")
                    return
                page_path = candidates[0]
                if any_mask_eligibility_changed:
                    self.app._log("Testo/onomatopea modificati, rieseguo la pulizia (inpainting) per allinearla...")
                else:
                    self.app._log("Pagina pulita non trovata, eseguo pulizia (inpainting)...")
                cleaned_path = clean.run(page_path, translated_path, cfg, comic_work_dir)

            final_path = render.run(cleaned_path, translated_path, cfg, comic_output_dir)

            self.app._log(f"Pagina rigenerata: {final_path}")
            self._review_dirty = False
            self.btn_regenerate.config(state=DISABLED)
            self._load_page()
            self.status.config(text=f"✅ Pagina rigenerata: {page_id}", foreground="green")
        except Exception as e:
            messagebox.showerror(t("Errore"), f"Errore rigenerazione: {e}")
            self.status.config(text=f"Errore: {e}", foreground="red")

    def _on_close(self):
        if self._review_dirty:
            if not messagebox.askyesno(
                t("Modifiche non salvate"),
                t("Ci sono modifiche non rigenerate. Vuoi chiudere comunque, perdendole?")
            ):
                return
        self.win.destroy()

class PipelineGUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(t("Pipeline Traduzione Fumetti"))
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Segoe UI", 10))
        self.style.configure("TLabel", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        self.style.configure("Progress.TLabel", font=("Segoe UI", 10, "bold"))
        self.process = None
        self.running = False
        self.current_stage = None
        self.total_stages = 0
        self.completed_stages = 0
        self.current_stage_total_pages = 0
        self._review_window = None
        try:
            i18n.language_from_config(load_cfg())
        except Exception:
            pass  # config illeggibile: si resta sulla lingua di default
        self._build_ui()
        # Le etichette sono letterali italiani nel codice: si traduce l'albero
        # dei widget finito, invece di avvolgere ~200 stringhe una per una.
        i18n.translate_tk_tree(self.root)
        self.root.title(t("Pipeline Traduzione Fumetti"))
        self._refresh_status()

    def _build_ui(self):
        header = ttk.Label(self.root, text="Pipeline Traduzione Fumetti", style="Header.TLabel")
        header.pack(pady=10)
        self.progress_frame = ttk.LabelFrame(self.root, text="Progressione", padding=10)
        self.progress_frame.pack(fill=X, padx=10, pady=5)
        self.lbl_stage = ttk.Label(self.progress_frame, text="In attesa...", style="Progress.TLabel")
        self.lbl_stage.pack(anchor=W)
        self.progress_total = ttk.Progressbar(self.progress_frame, orient=HORIZONTAL, length=400, mode="determinate")
        self.progress_total.pack(fill=X, pady=5)
        self.lbl_progress_total = ttk.Label(self.progress_frame, text="0%")
        self.lbl_progress_total.pack(anchor=E)
        self.lbl_stage_detail = ttk.Label(self.progress_frame, text="")
        self.lbl_stage_detail.pack(anchor=W)
        self.progress_stage = ttk.Progressbar(self.progress_frame, orient=HORIZONTAL, length=400, mode="determinate")
        self.progress_stage.pack(fill=X, pady=5)
        self.lbl_progress_stage = ttk.Label(self.progress_frame, text="0/0 pagine")
        self.lbl_progress_stage.pack(anchor=E)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.tab_pipeline = Frame(self.notebook)
        self.notebook.add(self.tab_pipeline, text="Pipeline")
        self._build_pipeline_tab()
        self.tab_pretreat = Frame(self.notebook)
        self.notebook.add(self.tab_pretreat, text="Pretrattamento")
        self._build_pretreat_tab()
        self.tab_config = Frame(self.notebook)
        self.notebook.add(self.tab_config, text="Configurazione")
        self._build_config_tab()
        self.tab_edit = Frame(self.notebook)
        self.notebook.add(self.tab_edit, text="Correzioni")
        self._build_edit_tab()
        self.tab_review = Frame(self.notebook)
        self.notebook.add(self.tab_review, text="Revisione")
        self._build_review_tab()
        self.tab_cache = Frame(self.notebook)
        self.notebook.add(self.tab_cache, text="Cache")
        self._build_cache_tab()
        self._build_console()

    def _build_pipeline_tab(self):
        outer = Frame(self.tab_pipeline)
        outer.pack(fill=BOTH, expand=True)
        canvas = Canvas(outer)
        scrollbar = Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        frame = Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=frame, anchor="nw")
        def on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)
        frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        def on_mousewheel(e):
            canvas.yview_scroll(-1 if e.num == 4 else 1, "units")
        canvas.bind("<Button-4>", on_mousewheel)
        canvas.bind("<Button-5>", on_mousewheel)
        canvas.bind("<Enter>", lambda e: canvas.focus_set())
        self.status_frame = ttk.LabelFrame(frame, text="Stato Attuale", padding=10)
        self.status_frame.pack(fill=X, pady=5)
        self.lbl_status = ttk.Label(self.status_frame, text="Caricamento...")
        self.lbl_status.pack(anchor=W)
        self.lbl_translator = ttk.Label(self.status_frame, text="Traduttore: -")
        self.lbl_translator.pack(anchor=W)
        self.lbl_lm_models = ttk.Label(self.status_frame, text="llama-server: -")
        self.lbl_lm_models.pack(anchor=W)
        stage_frame = ttk.LabelFrame(frame, text="Stage da eseguire", padding=10)
        stage_frame.pack(fill=X, pady=10)
        self.stage_var = StringVar(value="full")
        stages = [
            ("Pipeline COMPLETA (OCR -> Traduzione -> Pulizia -> Render)", "full"),
            ("Solo OCR (Detection + Riconoscimento)", "ocr"),
            ("Solo Traduzione", "auto_translate"),
            ("Solo Pulizia", "clean"),
            ("Solo Render (Scrittura testo)", "render"),
        ]
        for text, value in stages:
            ttk.Radiobutton(stage_frame, text=text, variable=self.stage_var, value=value).pack(anchor=W, pady=2)
        self.skip_translate_var = BooleanVar(value=False)
        ttk.Checkbutton(
            stage_frame,
            text="Salta traduzione (usa il testo OCR cosi' com'e', fumetto gia' in italiano)",
            variable=self.skip_translate_var,
        ).pack(anchor=W, pady=(5, 0))
        opts_frame = ttk.LabelFrame(frame, text="Opzioni", padding=10)
        opts_frame.pack(fill=X, pady=5)
        limit_frame = Frame(opts_frame)
        limit_frame.pack(fill=X, pady=2)
        ttk.Label(limit_frame, text="Limite pagine:").pack(side=LEFT)
        self.limit_var = StringVar(value="")
        ttk.Entry(limit_frame, textvariable=self.limit_var, width=10).pack(side=LEFT, padx=5)
        ttk.Label(limit_frame, text="(vuoto = tutte, N = pagina N, N-M = intervallo pagine N-M)").pack(side=LEFT)
        comic_frame = Frame(opts_frame)
        comic_frame.pack(fill=X, pady=2)
        ttk.Label(comic_frame, text="Fumetto:").pack(side=LEFT)
        self.comic_var = StringVar(value="tutti")
        self.combo_comics = ttk.Combobox(comic_frame, textvariable=self.comic_var, state="readonly", width=30)
        self.combo_comics.pack(side=LEFT, padx=5)
        backend_frame = Frame(opts_frame)
        backend_frame.pack(fill=X, pady=2)
        ttk.Label(backend_frame, text="Backend OCR:").pack(side=LEFT)
        self.ocr_backend_var = StringVar(value="qwen")
        self.combo_ocr_backend = ttk.Combobox(
            backend_frame, textvariable=self.ocr_backend_var, state="readonly", width=20,
            values=["qwen", "paddleocr_vl"],
        )
        self.combo_ocr_backend.pack(side=LEFT, padx=5)
        btn_frame = Frame(frame)
        btn_frame.pack(fill=X, pady=15)
        self.btn_run = ttk.Button(btn_frame, text="Avvia", command=self._run_pipeline)
        self.btn_run.pack(side=LEFT, padx=5)
        self.btn_stop = ttk.Button(btn_frame, text="Ferma", command=self._stop_pipeline, state=DISABLED)
        self.btn_stop.pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Aggiorna stato", command=self._refresh_status).pack(side=LEFT, padx=5)

    def _build_pretreat_tab(self):
        frame = self.tab_pretreat
        pretreat_frame = ttk.LabelFrame(frame, text="Pretrattamento pagine (ComfyUI)", padding=10)
        pretreat_frame.pack(fill=X, padx=10, pady=10)
        ttk.Label(
            pretreat_frame,
            text="Passa le pagine grezze attraverso ComfyUI/Flux-Klein con un prompt libero,\n"
                 "PRIMA di OCR - utile per upscale/pulizia rumore/altro. Non tocca mai gli\n"
                 "originali in input_pages/. Il prompt resta salvato per la volta dopo.",
            foreground="gray",
        ).pack(anchor=W, pady=(0, 5))

        # --- Cartella di partenza ---
        in_row = Frame(pretreat_frame)
        in_row.pack(fill=X, pady=2)
        ttk.Label(in_row, text="Cartella input:").pack(side=LEFT)
        self.pretreat_in_var = StringVar(value=self._default_pretreat_input_dir())
        ttk.Entry(in_row, textvariable=self.pretreat_in_var).pack(
            side=LEFT, fill=X, expand=True, padx=5)
        ttk.Button(in_row, text="Sfoglia...", command=self._browse_pretreat_input_dir).pack(side=LEFT)

        # --- Fumetto da trattare ---
        comic_row = Frame(pretreat_frame)
        comic_row.pack(fill=X, pady=2)
        ttk.Label(comic_row, text="Fumetto:").pack(side=LEFT)
        self.pretreat_comic_var = StringVar(value="tutti")
        self.combo_pretreat_comics = ttk.Combobox(
            comic_row, textvariable=self.pretreat_comic_var, state="readonly", width=35)
        self.combo_pretreat_comics.pack(side=LEFT, padx=5)
        ttk.Button(comic_row, text="Aggiorna",
                   command=self._refresh_pretreat_comics).pack(side=LEFT)

        # --- Pagine: tutte / singola / intervallo ---
        pages_frame = ttk.LabelFrame(pretreat_frame, text="Pagine", padding=8)
        pages_frame.pack(fill=X, pady=(8, 2))
        self.pretreat_range_mode = StringVar(value="all")
        row_all = Frame(pages_frame)
        row_all.pack(fill=X, anchor=W)
        ttk.Radiobutton(row_all, text="Tutto il fumetto", variable=self.pretreat_range_mode,
                        value="all", command=self._update_pretreat_range_state).pack(side=LEFT)
        row_single = Frame(pages_frame)
        row_single.pack(fill=X, anchor=W, pady=2)
        ttk.Radiobutton(row_single, text="Pagina singola:", variable=self.pretreat_range_mode,
                        value="single", command=self._update_pretreat_range_state).pack(side=LEFT)
        self.pretreat_page_var = StringVar(value="1")
        self.entry_pretreat_page = ttk.Entry(row_single, textvariable=self.pretreat_page_var, width=6)
        self.entry_pretreat_page.pack(side=LEFT, padx=5)
        row_range = Frame(pages_frame)
        row_range.pack(fill=X, anchor=W, pady=2)
        ttk.Radiobutton(row_range, text="Intervallo, da:", variable=self.pretreat_range_mode,
                        value="range", command=self._update_pretreat_range_state).pack(side=LEFT)
        self.pretreat_start_var = StringVar(value="1")
        self.entry_pretreat_start = ttk.Entry(row_range, textvariable=self.pretreat_start_var, width=6)
        self.entry_pretreat_start.pack(side=LEFT, padx=5)
        ttk.Label(row_range, text="a:").pack(side=LEFT)
        self.pretreat_end_var = StringVar(value="")
        self.entry_pretreat_end = ttk.Entry(row_range, textvariable=self.pretreat_end_var, width=6)
        self.entry_pretreat_end.pack(side=LEFT, padx=5)
        ttk.Label(row_range, text="(vuoto = fino all'ultima pagina)",
                  foreground="gray").pack(side=LEFT)

        # --- Risoluzione di lavoro ---
        res_row = Frame(pretreat_frame)
        res_row.pack(fill=X, pady=(8, 2))
        self.pretreat_fullres_var = BooleanVar(value=False)
        ttk.Checkbutton(
            res_row,
            text="Piena risoluzione (bypassa il resize del workflow)",
            variable=self.pretreat_fullres_var,
            command=self._update_pretreat_res_state,
        ).pack(side=LEFT)
        ttk.Label(res_row, text="Megapixel:").pack(side=LEFT, padx=(15, 0))
        self.pretreat_mp_var = StringVar(value=self._workflow_megapixels())
        self.entry_pretreat_mp = ttk.Entry(res_row, textvariable=self.pretreat_mp_var, width=6)
        self.entry_pretreat_mp.pack(side=LEFT, padx=5)
        ttk.Label(res_row, text="(vuoto = come salvato nel workflow)",
                  foreground="gray").pack(side=LEFT)
        ttk.Label(
            pretreat_frame,
            text="A piena risoluzione Flux vede la pagina intera senza rimpicciolirla, ma serve\n"
                 "molta piu' VRAM ed e' fuori dalla risoluzione su cui il modello e' addestrato:\n"
                 "se va in OOM o peggiora, alza i megapixel invece di togliere del tutto il resize.",
            foreground="gray",
        ).pack(anchor=W)

        # --- Destinazione ---
        out_row = Frame(pretreat_frame)
        out_row.pack(fill=X, pady=(8, 2))
        ttk.Label(out_row, text="Salva in:").pack(side=LEFT)
        self.pretreat_out_var = StringVar(value=self._default_pretreat_dir())
        ttk.Entry(out_row, textvariable=self.pretreat_out_var).pack(
            side=LEFT, fill=X, expand=True, padx=5)
        ttk.Button(out_row, text="Sfoglia...", command=self._browse_pretreat_dir).pack(side=LEFT)
        ttk.Label(
            pretreat_frame,
            text="Le pagine finiscono in <destinazione>/<fumetto>/. "
                 "Default: pretreated_pages/ accanto alla cartella input.",
            foreground="gray",
        ).pack(anchor=W)

        ttk.Label(pretreat_frame, text="Prompt:").pack(anchor=W, pady=(8, 0))
        pretreat_text_container = Frame(pretreat_frame)
        pretreat_text_container.pack(fill=X)
        self.pretreat_prompt_text = Text(pretreat_text_container, height=4, wrap="word",
            font=("Consolas", 9), bg="#2d2d2d", fg="#d4d4d4",
            insertbackground="white")
        pretreat_scroll = Scrollbar(pretreat_text_container, command=self.pretreat_prompt_text.yview)
        self.pretreat_prompt_text.configure(yscrollcommand=pretreat_scroll.set)
        self.pretreat_prompt_text.pack(side=LEFT, fill=X, expand=True)
        pretreat_scroll.pack(side=RIGHT, fill=Y)
        try:
            self.pretreat_prompt_text.insert("1.0", load_cfg().get("flux_pretreat", {}).get("last_prompt", ""))
        except Exception:
            pass
        ttk.Button(pretreat_frame, text="▶ Avvia pretrattamento", command=self._run_pretreat).pack(pady=5)
        self._update_pretreat_range_state()
        self._update_pretreat_res_state()

    def _default_pretreat_input_dir(self) -> str:
        try:
            return str(Path(load_cfg()["paths"]["input_dir"]))
        except Exception:
            return ""

    def _browse_pretreat_input_dir(self):
        initial = self.pretreat_in_var.get().strip() or self._default_pretreat_input_dir()
        chosen = filedialog.askdirectory(
            title=t("Cartella di partenza del pretrattamento"),
            initialdir=initial or None)
        if not chosen:
            return
        self.pretreat_in_var.set(chosen)
        # La destinazione di default segue la sorgente: cartella sorella,
        # mai dentro l'input.
        self.pretreat_out_var.set(str(Path(chosen).parent / "pretreated_pages"))
        self._refresh_pretreat_comics()

    def _refresh_pretreat_comics(self):
        """Ripopola la lista fumetti dalla cartella input scelta qui, che puo'
        essere diversa da paths.input_dir usata dalla tab Pipeline."""
        base = Path(self.pretreat_in_var.get().strip() or self._default_pretreat_input_dir())
        values = ["tutti"]
        try:
            subdirs = sorted(d.name for d in base.iterdir() if d.is_dir())
            if subdirs:
                values += subdirs
            else:
                exts = {".png", ".jpg", ".jpeg", ".webp"}
                if any(p.suffix.lower() in exts for p in base.iterdir()):
                    values.append("(root)")
        except Exception as e:
            self._log(f"Cartella input pretrattamento non leggibile: {e}")
        self.combo_pretreat_comics.config(values=values)
        if self.pretreat_comic_var.get() not in values:
            self.pretreat_comic_var.set("tutti")

    def _default_pretreat_dir(self) -> str:
        # Cartella sorella di input_pages/ (stessa scelta di flux_pretreat.py):
        # mai dentro input_pages, cosi' non puo' sovrascrivere gli originali.
        try:
            base_input = Path(load_cfg()["paths"]["input_dir"])
            return str(base_input.parent / "pretreated_pages")
        except Exception:
            return ""

    def _workflow_megapixels(self) -> str:
        """Valore di megapixel gia' salvato nel nodo di scala del workflow,
        cosi' il campo parte da quello che ComfyUI userebbe davvero."""
        try:
            with open(FLUX_WORKFLOW_FILE, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            for node in workflow.values():
                mp = node.get("inputs", {}).get("megapixels")
                if mp is not None:
                    return str(mp)
        except Exception:
            pass
        return ""

    def _update_pretreat_res_state(self):
        self.entry_pretreat_mp.config(
            state=DISABLED if self.pretreat_fullres_var.get() else NORMAL)

    def _update_pretreat_range_state(self):
        mode = self.pretreat_range_mode.get()
        self.entry_pretreat_page.config(state=NORMAL if mode == "single" else DISABLED)
        state_range = NORMAL if mode == "range" else DISABLED
        self.entry_pretreat_start.config(state=state_range)
        self.entry_pretreat_end.config(state=state_range)

    def _browse_pretreat_dir(self):
        initial = self.pretreat_out_var.get().strip() or self._default_pretreat_dir()
        chosen = filedialog.askdirectory(
            title=t("Cartella di destinazione del pretrattamento"),
            initialdir=initial or None)
        if chosen:
            self.pretreat_out_var.set(chosen)

    def _build_config_tab(self):
        outer = Frame(self.tab_config)
        outer.pack(fill=BOTH, expand=True)
        canvas = Canvas(outer)
        scrollbar = Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        frame = Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=frame, anchor="nw")
        def on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)
        frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        def on_mousewheel(e):
            canvas.yview_scroll(-1 if e.num == 4 else 1, "units")
        canvas.bind("<Button-4>", on_mousewheel)
        canvas.bind("<Button-5>", on_mousewheel)
        canvas.bind("<Enter>", lambda e: canvas.focus_set())
        dir_frame = ttk.LabelFrame(frame, text="Directory di lavoro", padding=10)
        dir_frame.pack(fill=X, pady=5)
        try:
            cfg = load_cfg()
            current_input = cfg["paths"]["input_dir"]
            current_root = str(Path(current_input).parent)
        except Exception:
            current_root = str(Path.cwd())
            cfg = {}
        self.work_root_var = StringVar(value=current_root)
        root_row = Frame(dir_frame)
        root_row.pack(fill=X, pady=2)
        ttk.Label(root_row, text="Root:").pack(side=LEFT)
        ttk.Entry(root_row, textvariable=self.work_root_var, width=45).pack(side=LEFT, padx=5)
        ttk.Button(root_row, text="Sfoglia…", command=self._browse_work_root).pack(side=LEFT)
        ttk.Button(
            root_row, text="Applica",
            command=lambda: self._apply_work_root(self.work_root_var.get())
        ).pack(side=LEFT, padx=5)
        self.lbl_input_path = ttk.Label(
            dir_frame,
            text="input  → " + cfg.get("paths", {}).get("input_dir", ""),
            foreground="gray"
        )
        self.lbl_output_path = ttk.Label(
            dir_frame,
            text="output → " + cfg.get("paths", {}).get("output_dir", ""),
            foreground="gray"
        )
        self.lbl_work_path = ttk.Label(
            dir_frame,
            text="work   → " + cfg.get("paths", {}).get("work_dir", ""),
            foreground="gray"
        )
        self.lbl_input_path.pack(anchor=W)
        self.lbl_output_path.pack(anchor=W)
        self.lbl_work_path.pack(anchor=W)
        lingua_frame = ttk.LabelFrame(frame, text="Lingua interfaccia", padding=10)
        lingua_frame.pack(fill=X, pady=10)
        ttk.Label(
            lingua_frame,
            text="Lingua dell'interfaccia. I messaggi di log della pipeline restano in italiano.",
            foreground="gray",
        ).pack(anchor=W, pady=(0, 5))
        lingua_row = Frame(lingua_frame)
        lingua_row.pack(fill=X)
        ttk.Label(lingua_row, text="Lingua:").pack(side=LEFT)
        self.lingua_var = StringVar(value=i18n.get_language())
        self.combo_lingua = ttk.Combobox(
            lingua_row, textvariable=self.lingua_var, state="readonly", width=14,
            values=[i18n.LINGUE[c] for c in i18n.LINGUE],
        )
        self.combo_lingua.pack(side=LEFT, padx=5)
        ttk.Button(lingua_row, text="Salva lingua", command=self._save_language).pack(side=LEFT, padx=5)

        lm_frame = ttk.LabelFrame(frame, text="Modelli GGUF (llama-server)", padding=10)
        lm_frame.pack(fill=X, pady=10)
        ttk.Label(
            lm_frame,
            text="Path dei .gguf che llama-server carica per lo stage OCR e per la traduzione.\n"
                 "Un modello per volta: il server viene avviato all'inizio dello stage e chiuso\n"
                 "alla fine, cosi' la VRAM torna libera per la pulizia.",
            foreground="gray",
        ).pack(anchor=W, pady=(0, 5))
        for etichetta, attr in (("OCR (qwen):", "lm_ocr_var"),
                                ("OCR (paddleocr_vl):", "lm_paddle_var"),
                                ("Proiettore OCR (mmproj):", "lm_mmproj_var"),
                                ("Traduzione:", "lm_trans_var")):
            row = Frame(lm_frame)
            row.pack(fill=X, pady=2)
            ttk.Label(row, text=etichetta, width=24, anchor=W).pack(side=LEFT)
            var = StringVar()
            setattr(self, attr, var)
            ttk.Entry(row, textvariable=var).pack(side=LEFT, fill=X, expand=True, padx=5)
            ttk.Button(row, text="...", width=3,
                       command=lambda v=var: self._browse_gguf(v)).pack(side=LEFT)
        port_row = Frame(lm_frame)
        port_row.pack(fill=X, pady=2)
        ttk.Label(port_row, text="Porta llama-server:", width=24, anchor=W).pack(side=LEFT)
        self.llama_port_var = StringVar()
        ttk.Entry(port_row, textvariable=self.llama_port_var, width=8).pack(side=LEFT, padx=5)
        prompt_frame = ttk.LabelFrame(lm_frame, text="System Prompt traduzione", padding=5)
        prompt_frame.pack(fill=X, pady=(8, 0))
        self.lm_prompt_text = Text(prompt_frame, height=5, wrap="word",
            font=("Consolas", 9), bg="#2d2d2d", fg="#d4d4d4",
            insertbackground="white")
        prompt_scroll = Scrollbar(prompt_frame, command=self.lm_prompt_text.yview)
        self.lm_prompt_text.configure(yscrollcommand=prompt_scroll.set)
        self.lm_prompt_text.pack(side=LEFT, fill=X, expand=True)
        prompt_scroll.pack(side=RIGHT, fill=Y)
        ttk.Button(lm_frame, text="Salva modelli", command=self._save_lm_models).pack(pady=5)
        font_frame = ttk.LabelFrame(frame, text="Dimensione font fissa", padding=10)
        font_frame.pack(fill=X, pady=10)
        ttk.Label(
            font_frame,
            text="Se impostata, forza questa dimensione (px) su tutti i balloon normali di\n"
                 "ogni pagina, al posto del calcolo automatico. Lascia vuoto per l'automatico.",
            foreground="gray",
        ).pack(anchor=W, pady=(0, 5))
        fixed_size_row = Frame(font_frame)
        fixed_size_row.pack(fill=X)
        ttk.Label(fixed_size_row, text="Dimensione (px):").pack(side=LEFT)
        self.fixed_font_size_var = StringVar()
        ttk.Entry(fixed_size_row, textvariable=self.fixed_font_size_var, width=6).pack(side=LEFT, padx=5)
        ttk.Button(fixed_size_row, text="Salva", command=self._save_fixed_font_size).pack(side=LEFT, padx=5)
        ttk.Button(frame, text="Visualizza config.yaml", command=self._view_config).pack(pady=10)
        self._load_config_values()

    def _browse_work_root(self):
        chosen = filedialog.askdirectory(title=t("Seleziona directory di lavoro"))
        if not chosen:
            return
        self.work_root_var.set(chosen)
        self._apply_work_root(chosen)

    def _apply_work_root(self, root_dir: str):
        try:
            cfg = load_cfg()
            base = Path(root_dir)
            cfg["paths"]["input_dir"]  = str(base / "input_pages")
            cfg["paths"]["output_dir"] = str(base / "output_pages")
            cfg["paths"]["work_dir"]   = str(base / "work")
            save_cfg(cfg)
            self._log(f"Directory di lavoro impostata: {root_dir}")
            self.lbl_input_path.config( text="input  → " + cfg["paths"]["input_dir"])
            self.lbl_output_path.config(text="output → " + cfg["paths"]["output_dir"])
            self.lbl_work_path.config(  text="work   → " + cfg["paths"]["work_dir"])
            self.work_root_var.set(root_dir)
            self._refresh_status()
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _build_edit_tab(self):
        frame = Frame(self.tab_edit)
        frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        top_frame = Frame(frame)
        top_frame.pack(fill=X, pady=5)
        ttk.Label(top_frame, text="Fumetto:").pack(side=LEFT)
        self.edit_comic_var = StringVar()
        self.combo_edit_comics = ttk.Combobox(top_frame, textvariable=self.edit_comic_var, state="readonly", width=35)
        self.combo_edit_comics.pack(side=LEFT, padx=5)
        self.combo_edit_comics.bind("<<ComboboxSelected>>", lambda e: self._refresh_edit_pages())
        ttk.Button(top_frame, text="Aggiorna", command=self._refresh_edit_comics).pack(side=LEFT, padx=5)
        export_frame = ttk.LabelFrame(frame, text="Esporta / Importa testo TXT", padding=10)
        export_frame.pack(fill=X, pady=5)
        row1 = Frame(export_frame)
        row1.pack(fill=X)
        ttk.Label(row1, text="Pagine:").pack(side=LEFT)
        self.edit_export_range_var = StringVar(value="")
        ttk.Entry(row1, textvariable=self.edit_export_range_var, width=10).pack(side=LEFT, padx=5)
        ttk.Label(row1, text="(vuoto = tutte, N = pagina N, N-M = intervallo)", foreground="gray").pack(side=LEFT)
        row2 = Frame(export_frame)
        row2.pack(fill=X, pady=(5, 0))
        ttk.Label(row2, text="Contenuto:").pack(side=LEFT)
        self.edit_export_mode_var = StringVar(value="both")
        self.combo_export_mode = ttk.Combobox(
            row2, textvariable=self.edit_export_mode_var, state="readonly", width=22,
            values=["Originale + Tradotto", "Solo originale", "Solo tradotto"],
        )
        self.combo_export_mode.current(0)
        self.combo_export_mode.pack(side=LEFT, padx=5)
        ttk.Button(row2, text="Esporta in TXT...", command=self._export_edit_text).pack(side=LEFT, padx=10)
        ttk.Button(row2, text="Importa da TXT...", command=self._import_edit_text).pack(side=LEFT, padx=5)
        page_frame = Frame(frame)
        page_frame.pack(fill=X, pady=5)
        ttk.Label(page_frame, text="Pagina:").pack(side=LEFT)
        self.edit_page_var = StringVar()
        self.combo_edit_pages = ttk.Combobox(page_frame, textvariable=self.edit_page_var, state="readonly", width=10)
        self.combo_edit_pages.pack(side=LEFT, padx=5)
        self.combo_edit_pages.bind("<<ComboboxSelected>>", lambda e: self._load_preview())
        ttk.Button(page_frame, text="Carica", command=self._load_preview).pack(side=LEFT, padx=5)
        ttk.Button(page_frame, text="Salva modifiche", command=self._save_edits).pack(side=LEFT, padx=5)
        editor_frame = ttk.LabelFrame(frame, text="Editor traduzioni (doppio-click per modificare)", padding=5)
        editor_frame.pack(fill=BOTH, expand=True, pady=5)
        columns = ("id", "originale", "tradotto", "status")
        self.edit_tree = ttk.Treeview(editor_frame, columns=columns, show="headings", height=12)
        self.edit_tree.heading("id", text="ID")
        self.edit_tree.heading("originale", text="Testo Originale")
        self.edit_tree.heading("tradotto", text="Traduzione")
        self.edit_tree.heading("status", text="Stato")
        self.edit_tree.column("id", width=40, anchor=CENTER)
        self.edit_tree.column("originale", width=400)
        self.edit_tree.column("tradotto", width=400)
        self.edit_tree.column("status", width=80, anchor=CENTER)
        tree_scroll = Scrollbar(editor_frame, orient="vertical", command=self.edit_tree.yview)
        self.edit_tree.configure(yscrollcommand=tree_scroll.set)
        self.edit_tree.pack(side=LEFT, fill=BOTH, expand=True)
        tree_scroll.pack(side=RIGHT, fill=Y)
        self.edit_tree.bind("<Double-1>", self._on_tree_double_click)
        self.edit_popup = None
        self.editing_item = None
        self.editing_column = None
        btn_frame = Frame(frame)
        btn_frame.pack(fill=X, pady=5)
        ttk.Button(btn_frame, text="Apri editor CLI", command=self._run_editor).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Marca come onomatopea (-)", command=self._mark_onomatopea).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Ripristina originale", command=self._restore_original).pack(side=LEFT, padx=5)
        self.edit_status = ttk.Label(frame, text="", foreground="green")
        self.edit_status.pack(pady=2)
        self._current_edit_data = None
        self._modified = False

    def _build_review_tab(self):
        frame = Frame(self.tab_review)
        frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        ttk.Label(
            frame,
            text="La revisione visuale del fumetto si apre in una finestra dedicata\n"
            "a schermo pieno, per avere piu' spazio per leggere e correggere.",
            justify=CENTER
        ).pack(pady=20)
        ttk.Button(frame, text="Apri finestra di Revisione", command=self._open_review_window).pack(pady=10)

    def _build_cache_tab(self):
        frame = Frame(self.tab_cache)
        frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        self.cache_stats = ttk.Label(frame, text="Statistiche cache: -")
        self.cache_stats.pack(pady=10)
        btn_frame = Frame(frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Aggiorna stats", command=self._refresh_cache_stats).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Svuota cache", command=self._clear_cache).pack(side=LEFT, padx=5)
        list_frame = ttk.LabelFrame(frame, text="Entries in cache", padding=10)
        list_frame.pack(fill=BOTH, expand=True, pady=5)
        self.cache_listbox = Listbox(list_frame)
        self.cache_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        cache_scroll = Scrollbar(list_frame, command=self.cache_listbox.yview)
        cache_scroll.pack(side=RIGHT, fill=Y)
        self.cache_listbox.config(yscrollcommand=cache_scroll.set)
        self._refresh_cache_stats()

    def _build_console(self):
        console_frame = ttk.LabelFrame(self.root, text="Output", padding=5)
        console_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.console = Text(console_frame, wrap="word", height=8, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.console.pack(side=LEFT, fill=BOTH, expand=True)
        console_scroll = Scrollbar(console_frame, command=self.console.yview)
        console_scroll.pack(side=RIGHT, fill=Y)
        self.console.config(yscrollcommand=console_scroll.set)
        self.console.config(state=DISABLED)

    def _reset_progress(self):
        self.completed_stages = 0
        self.total_stages = 0
        self._current_page_index = 0
        self.progress_total["value"] = 0
        self.progress_stage["value"] = 0
        self.lbl_progress_total.config(text=t("0%"))
        self.lbl_progress_stage.config(text=t("0/0 pagine"))
        self.lbl_stage.config(text=t("In attesa..."))
        self.lbl_stage_detail.config(text="")

    def _set_stage_progress(self, stage_name: str, current: int, total: int):
        if total > 0:
            pct = (current / total) * 100
            self.progress_stage["value"] = pct
            self.lbl_progress_stage.config(text=f"{current}/{total} pagine")
            self.lbl_stage_detail.config(text=f"Stage: {stage_name}")
        else:
            self.progress_stage["value"] = 0
            self.lbl_progress_stage.config(text=t("Inizializzazione..."))

    def _update_total_progress(self):
        if self.total_stages > 0:
            pct = (self.completed_stages / self.total_stages) * 100
            self.progress_total["value"] = pct
            self.lbl_progress_total.config(text=f"{pct:.0f}%")

    def _advance_stage(self):
        self.completed_stages += 1
        self._update_total_progress()

    def _parse_progress_from_output(self, line: str) -> dict | None:
        match = re.search(r"===\s*(OCR|Traduzione|Pulizia|Render|Pretrattamento):\s*(.+?)\s*===", line)
        if match:
            self._current_page_index = getattr(self, '_current_page_index', 0) + 1
            stage_name = match.group(1)
            self.root.after(0, lambda s=stage_name: self.lbl_stage.config(
                text=f"Stage: {s} - pagina {self._current_page_index}/{self.current_stage_total_pages}"
            ))
            if self.current_stage_total_pages > 0:
                pct = (self._current_page_index / self.current_stage_total_pages) * 100
                self.root.after(0, lambda p=pct: (
                    self.progress_stage.configure(value=p),
                    self.lbl_progress_stage.config(
                        text=f"{self._current_page_index}/{self.current_stage_total_pages} pagine"
                    )
                ))
            return {"page_index": self._current_page_index}
        match = re.search(r"FUMETTO:.*?(\d+)\s*pagine", line)
        if match:
            self._current_page_index = 0
            total = int(match.group(1))
            self.root.after(0, lambda t=total: self.progress_stage.configure(value=0))
            return {"total_pages": total}
        return None

    def _log(self, message: str, tag: str = "info"):
        self.console.config(state=NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.insert(END, f"[{timestamp}] {message}\n")
        self.console.see(END)
        self.console.config(state=DISABLED)
        parsed = self._parse_progress_from_output(message)
        if parsed:
            if "total_pages" in parsed:
                self.current_stage_total_pages = parsed["total_pages"]
            elif "page_index" in parsed and self.current_stage_total_pages > 0:
                self._set_stage_progress(self.current_stage, parsed["page_index"], self.current_stage_total_pages)

    def _check_llama_ready(self, stage: str, skip_translate: bool) -> str | None:
        """Messaggio d'errore se llama-server non e' utilizzabile per questo
        stage, None se e' tutto a posto."""
        if not shutil.which("llama-server"):
            return ("Comando 'llama-server' non trovato nel PATH.\n"
                    "Installa llama.cpp (winget install ggml.llamacpp) o aggiungi "
                    "la cartella dei binari al PATH.")
        try:
            cfg = load_cfg()
        except Exception as e:
            return f"config.yaml non leggibile: {e}"

        ls = cfg.get("llama_server", {})
        serve_ocr = stage in ("ocr", "full", "auto_translate")
        serve_trad = stage == "translate_lm" or (
            stage in ("full", "auto_translate") and not skip_translate)

        richiesti = []
        if serve_ocr:
            chiave = ("paddleocr_model_path"
                      if cfg.get("ocr_backend", "qwen") == "paddleocr_vl" else "ocr_model_path")
            richiesti.append(("modello OCR", ls.get(chiave), chiave))
        if serve_trad:
            richiesti.append(("modello di traduzione", ls.get("translate_model_path"),
                              "translate_model_path"))

        for etichetta, path, chiave in richiesti:
            if not path:
                return (f"Manca il {etichetta} in config.yaml "
                        f"(llama_server.{chiave}).\nImpostalo nella tab Configurazione.")
            if not Path(path).exists():
                return f"{etichetta.capitalize()} non trovato:\n{path}"
        return None

    def _refresh_status(self):
        try:
            cfg = load_cfg()
            self.lbl_translator.config(text=t("Traduttore: modello locale (llama-server)"))
            if llama_server_is_up(cfg):
                modello = llama_server_loaded_model(cfg) or "?"
                self.lbl_lm_models.config(text=f"llama-server: attivo - {modello}")
            elif shutil.which("llama-server"):
                self.lbl_lm_models.config(
                    text=t("llama-server: pronto (viene avviato all'inizio dello stage)"))
            else:
                self.lbl_lm_models.config(text=t("llama-server: NON TROVATO nel PATH"))
            comics = collect_comics(cfg)
            comic_values = ["tutti"] + [c or "(root)" for c in comics]
            self.combo_comics.config(values=comic_values)
            if self.comic_var.get() not in comic_values:
                self.comic_var.set("tutti")
            self.combo_edit_comics.config(values=[c or "(root)" for c in comics])
            self._refresh_pretreat_comics()
            self.ocr_backend_var.set(cfg.get("ocr_backend", "qwen"))
            self.lbl_status.config(text=f"Config: OK | Fumetti trovati: {len(comics)}")
        except Exception as e:
            self.lbl_status.config(text=f"Errore: {e}")

    def _run_pipeline(self):
        if self.running:
            return
        skip_translate = self.skip_translate_var.get()
        stage = self.stage_var.get()
        # main.py avvia e chiude llama-server da solo: qui basta accorgersi in
        # anticipo se manca il binario o il GGUF, invece di lasciar fallire lo
        # stage a meta' del fumetto.
        if stage in ("ocr", "translate_lm", "full", "auto_translate"):
            problema = self._check_llama_ready(stage, skip_translate)
            if problema:
                messagebox.showerror(t("Errore"), problema)
                return
        stage = self.stage_var.get()
        limit_str = self.limit_var.get().strip()
        comic = self.comic_var.get()
        if comic == "tutti":
            comic = None
        elif comic == "(root)":
            comic = ""
        start = None
        end = None
        if limit_str:
            if '-' in limit_str:
                parts = limit_str.split('-')
                try:
                    start = (int(parts[0]) - 1) if parts[0] else 0
                    end = int(parts[1]) if parts[1] else None
                except ValueError:
                    messagebox.showerror(t("Errore"), t("Formato intervallo non valido"))
                    return
            else:
                try:
                    n = int(limit_str)
                    start = n - 1
                    end = n
                except ValueError:
                    messagebox.showerror(t("Errore"), t("Numero non valido"))
                    return
        self._reset_progress()
        if stage == "full":
            self._run_full_pipeline(start, end, comic, skip_translate)
            return
        if stage == "auto_translate":
            stage = "translate_skip" if skip_translate else "translate_lm"
        self.total_stages = 1
        self.current_stage = stage
        self.current_stage_total_pages = 0
        try:
            cfg = load_cfg()
            self.current_stage_total_pages = count_pages(cfg, comic)
        except Exception:
            pass
        self._execute_command(stage, start, end, comic)

    def _run_full_pipeline(self, start, end, comic, skip_translate=False):
        translate_stage = "translate_skip" if skip_translate else "translate_lm"
        stages = ["ocr", translate_stage, "clean", "render"]
        self.total_stages = len(stages)
        self._log(f"Pipeline completa: {' -> '.join(stages)}")
        try:
            cfg = load_cfg()
            total_pages = count_pages(cfg, comic)
        except Exception:
            total_pages = 0
        def run_stages():
            self.running = True
            self.btn_run.config(state=DISABLED)
            self.btn_stop.config(state=NORMAL)
            try:
                cfg = load_cfg()
                if comic:
                    comics_to_run = [comic]
                else:
                    comics_to_run = collect_comics(cfg) or [""]
            except Exception:
                comics_to_run = [comic or ""]
            self.total_stages = len(stages) * len(comics_to_run)
            self.completed_stages = 0
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            for cidx, cur_comic in enumerate(comics_to_run):
                if not self.running:
                    self._log("Pipeline interrotta")
                    break
                label = cur_comic or "(root)"
                self._log(f"{'='*40}")
                self._log(f"Fumetto {cidx+1}/{len(comics_to_run)}: {label}")
                self._log(f"{'='*40}")
                for i, stage in enumerate(stages):
                    if not self.running:
                        self._log("Pipeline interrotta")
                        break
                    self.current_stage = stage
                    try:
                        cfg = load_cfg()
                        self.current_stage_total_pages = count_pages(cfg, cur_comic if cur_comic else None)
                    except Exception:
                        self.current_stage_total_pages = 0
                    self._current_page_index = 0
                    self._update_total_progress()
                    self.root.after(0, lambda s=stage, idx=i, c=label: self.lbl_stage.config(
                        text=f"[{c}] Stage {idx+1}/{len(stages)}: {s}"))
                    self._set_stage_progress(stage, 0, self.current_stage_total_pages)
                    cmd = build_command(stage, limit=None,
                        comic=cur_comic if cur_comic else None,
                        start=start, end=end,
                        ocr_backend=self.ocr_backend_var.get())
                    self._log(f"Comando: {' '.join(cmd)}")
                    try:
                        self.process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            env=env,
                        )
                        for line in self.process.stdout:
                            self.root.after(0, lambda l=line: self._log(l.strip()))
                        self.process.wait()
                        rc = self.process.returncode
                        if rc != 0:
                            self.root.after(0, lambda s=stage, c=label: self._log(
                                f"Errore nello stage {s} per {c} (codice {rc}), passo al fumetto successivo"))
                            break
                        self.completed_stages += 1
                        self._update_total_progress()
                        self.root.after(0, lambda s=stage, c=label: self._log(
                            f"✓ [{c}] Stage {s} completato"))
                    except Exception as e:
                        self.root.after(0, lambda err=e: self._log(f"Errore: {err}"))
                        break
                else:
                    self._log(f"✓ Fumetto {label} completato")
            self.root.after(0, self._pipeline_finished)
        threading.Thread(target=run_stages, daemon=True).start()

    def _execute_command(self, stage, start, end, comic):
        cmd = build_command(stage, limit=None, comic=comic, start=start, end=end,
            ocr_backend=self.ocr_backend_var.get())
        self._log(f"Esecuzione: {' '.join(cmd)}")
        self.running = True
        self.btn_run.config(state=DISABLED)
        self.btn_stop.config(state=NORMAL)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        def run_in_thread():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                for line in self.process.stdout:
                    self.root.after(0, lambda l=line: self._log(l.strip()))
                self.process.wait()
                rc = self.process.returncode
                if rc == 0:
                    self._advance_stage()
                    self.root.after(0, lambda: self._log("Completato!"))
                else:
                    self.root.after(0, lambda: self._log(f"Errore (codice {rc})"))
            except Exception as e:
                self.root.after(0, lambda msg=str(e): self._log(f"Errore: {msg}"))
            finally:
                self.running = False
                self.process = None
                self.root.after(0, self._pipeline_finished)
        threading.Thread(target=run_in_thread, daemon=True).start()

    def _run_pretreat(self):
        if self.running:
            return
        prompt = self.pretreat_prompt_text.get("1.0", END).strip()
        if not prompt:
            messagebox.showerror(t("Errore"), t("Scrivi un prompt prima di avviare il pretrattamento"))
            return
        try:
            cfg = load_cfg()
            cfg.setdefault("flux_pretreat", {})["last_prompt"] = prompt
            save_cfg(cfg)
        except Exception as e:
            self._log(f"Attenzione: prompt non salvato in config ({e})")

        comic = self.pretreat_comic_var.get()
        if comic == "tutti":
            comic = None
        elif comic == "(root)":
            comic = ""

        # Pagine 1-based nella GUI, slice 0-based per flux_pretreat.py.
        mode = self.pretreat_range_mode.get()
        start = end = None
        if mode == "single":
            try:
                n = int(self.pretreat_page_var.get().strip())
                if n < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror(t("Errore"), t("Numero di pagina non valido"))
                return
            start, end = n - 1, n
        elif mode == "range":
            start_str = self.pretreat_start_var.get().strip()
            end_str = self.pretreat_end_var.get().strip()
            try:
                start = (int(start_str) - 1) if start_str else 0
                end = int(end_str) if end_str else None
                if start < 0 or (end is not None and end <= start):
                    raise ValueError
            except ValueError:
                messagebox.showerror(t("Errore"), t("Intervallo di pagine non valido"))
                return

        # 0 = bypass del nodo di scala, >0 = megapixel di lavoro,
        # None = si lascia il workflow com'e'.
        if self.pretreat_fullres_var.get():
            megapixels = 0
        else:
            mp_str = self.pretreat_mp_var.get().strip()
            if mp_str:
                try:
                    megapixels = float(mp_str.replace(",", "."))
                    if megapixels <= 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror(
                        t("Errore"),
                        t("Megapixel non validi: usa un numero maggiore di 0, "
                        "oppure spunta 'Piena risoluzione'."))
                    return
            else:
                megapixels = None

        input_dir = self.pretreat_in_var.get().strip() or None
        output_dir = self.pretreat_out_var.get().strip() or None
        if input_dir and not Path(input_dir).is_dir():
            messagebox.showerror(t("Errore"), f"Cartella input non trovata:\n{input_dir}")
            return
        if input_dir and output_dir and Path(input_dir).resolve() == Path(output_dir).resolve():
            messagebox.showerror(t("Errore"), t("Input e destinazione coincidono: sovrascriverebbe gli originali."))
            return

        self._reset_progress()
        self.total_stages = 1
        self.current_stage = "pretreat"
        # Il conteggio pagine segue la cartella scelta qui, non paths.input_dir.
        try:
            cfg_pages = {"paths": {"input_dir": input_dir or self._default_pretreat_input_dir()}}
            self.current_stage_total_pages = count_pages(cfg_pages, comic)
        except Exception:
            self.current_stage_total_pages = 0
        cmd = build_command("pretreat", comic=comic, start=start, end=end, prompt=prompt,
                            input_dir=input_dir, output_dir=output_dir,
                            megapixels=megapixels)
        self._log(f"Esecuzione: {' '.join(cmd)}")
        self.running = True
        self.btn_run.config(state=DISABLED)
        self.btn_stop.config(state=NORMAL)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        def run_in_thread():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                for line in self.process.stdout:
                    self.root.after(0, lambda l=line: self._log(l.strip()))
                self.process.wait()
                rc = self.process.returncode
                if rc == 0:
                    self.root.after(0, lambda: self._log("Pretrattamento completato!"))
                else:
                    self.root.after(0, lambda: self._log(f"Errore (codice {rc})"))
            except Exception as e:
                self.root.after(0, lambda msg=str(e): self._log(f"Errore: {msg}"))
            finally:
                self.running = False
                self.process = None
                self.root.after(0, self._pipeline_finished)
        threading.Thread(target=run_in_thread, daemon=True).start()

    def _pipeline_finished(self):
        self.running = False
        self.process = None
        self.btn_run.config(state=NORMAL)
        self.btn_stop.config(state=DISABLED)
        self.lbl_stage.config(text=t("Completato!"))
        self.lbl_stage_detail.config(text="")
        self._refresh_status()

    def _stop_pipeline(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self._log("Pipeline fermata")
            self.running = False
            self.btn_run.config(state=NORMAL)
            self.btn_stop.config(state=DISABLED)

    def _load_config_values(self):
        try:
            cfg = load_cfg()
            self.lingua_var.set(i18n.LINGUE.get(i18n.get_language(), "Italiano"))
            ls = cfg.get("llama_server", {})
            self.lm_ocr_var.set(ls.get("ocr_model_path", ""))
            self.lm_paddle_var.set(ls.get("paddleocr_model_path", ""))
            self.lm_mmproj_var.set(ls.get("ocr_mmproj_path", ""))
            self.lm_trans_var.set(ls.get("translate_model_path", ""))
            self.llama_port_var.set(str(ls.get("port", 8081)))
            default_prompt = ("You are a professional comic book translator. "
                "Keep the original tone, register and profanity, do not soften anything.")
            prompt = cfg.get("local_llm", {}).get("translate_system_prompt", default_prompt)
            self.lm_prompt_text.delete("1.0", END)
            self.lm_prompt_text.insert("1.0", prompt)
            fixed_size = cfg.get("rendering", {}).get("fixed_font_size")
            self.fixed_font_size_var.set(str(fixed_size) if fixed_size else "")
        except Exception as e:
            self._log(f"Errore caricamento config: {e}")

    def _save_language(self):
        """La lingua e' letta all'avvio (le etichette sono tradotte una volta
        sull'albero dei widget), quindi si salva e si chiede un riavvio invece
        di ricostruire tutta l'interfaccia a caldo."""
        try:
            scelta = self.lingua_var.get()
            codice = next((c for c, nome in i18n.LINGUE.items() if nome == scelta), scelta)
            cfg = load_cfg()
            cfg.setdefault("ui", {})["language"] = codice
            save_cfg(cfg)
            messagebox.showinfo(
                t("Info"),
                t("Lingua salvata. Riavvia il programma per applicarla ovunque."))
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _browse_gguf(self, var: StringVar):
        chosen = filedialog.askopenfilename(
            title=t("Seleziona un modello GGUF"),
            initialdir=str(Path(var.get()).parent) if var.get() else None,
            filetypes=[("Modelli GGUF", "*.gguf"), ("Tutti i file", "*.*")],
        )
        if chosen:
            var.set(chosen)

    def _save_lm_models(self):
        try:
            cfg = load_cfg()
            ls = cfg.setdefault("llama_server", {})
            ls["ocr_model_path"] = self.lm_ocr_var.get().strip()
            ls["paddleocr_model_path"] = self.lm_paddle_var.get().strip()
            ls["ocr_mmproj_path"] = self.lm_mmproj_var.get().strip()
            ls["translate_model_path"] = self.lm_trans_var.get().strip()
            porta = self.llama_port_var.get().strip()
            if porta:
                try:
                    ls["port"] = int(porta)
                except ValueError:
                    messagebox.showerror(t("Errore"), t("Porta non valida"))
                    return
            cfg.setdefault("local_llm", {})["translate_system_prompt"] =                 self.lm_prompt_text.get("1.0", END).strip()
            cfg["local_llm"]["base_url"] = f"http://127.0.0.1:{ls.get('port', 8081)}/v1"
            save_cfg(cfg)
            self._log("Modelli llama-server salvati")
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _save_fixed_font_size(self):
        try:
            raw = self.fixed_font_size_var.get().strip()
            cfg = load_cfg()
            if not raw:
                cfg.setdefault("rendering", {})["fixed_font_size"] = None
                save_cfg(cfg)
                self._log("Dimensione font fissa disattivata (automatico)")
                return
            size = int(raw)
            if size <= 0:
                raise ValueError("deve essere positiva")
            cfg.setdefault("rendering", {})["fixed_font_size"] = size
            save_cfg(cfg)
            self._log(f"Dimensione font fissa impostata: {size}px")
        except ValueError:
            messagebox.showerror(t("Errore"), t("Inserisci un numero intero positivo, o lascia vuoto"))
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _view_config(self):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            top = Toplevel(self.root)
            top.title(t("config.yaml"))
            text = Text(top, wrap="word", width=80, height=40)
            text.pack(fill=BOTH, expand=True)
            text.insert(END, content)
            text.config(state=DISABLED)
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _refresh_edit_comics(self):
        try:
            cfg = load_cfg()
            comics = collect_comics(cfg)
            values = [c or "(root)" for c in comics]
            self.combo_edit_comics.config(values=values)
            if values:
                self.edit_comic_var.set(values[0])
                self._refresh_edit_pages()
        except Exception as e:
            self._log(f"Errore: {e}")

    def _refresh_edit_pages(self):
        try:
            cfg = load_cfg()
            work_dir = Path(cfg["paths"]["work_dir"])
            comic = self.edit_comic_var.get()
            if comic == "(root)":
                comic = ""
            pages = []
            if comic:
                base = work_dir / comic
                if base.exists():
                    pages = sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "translated.json").exists())
            else:
                pages = sorted(p.name for p in work_dir.iterdir() if p.is_dir() and (p / "translated.json").exists())
            self.combo_edit_pages.config(values=pages)
            if pages:
                self.edit_page_var.set(pages[0])
                self._load_preview()
        except Exception as e:
            self._log(f"Errore: {e}")

    def _export_edit_text(self):
        try:
            cfg = load_cfg()
            work_dir = Path(cfg["paths"]["work_dir"])
            comic = self.edit_comic_var.get()
            if comic == "(root)":
                comic = ""
            base = (work_dir / comic) if comic else work_dir
            if not base.exists():
                messagebox.showwarning(t("Attenzione"), t("Fumetto non valido"))
                return
            pages = sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "translated.json").exists())
            if not pages:
                messagebox.showwarning(t("Attenzione"), t("Nessuna pagina trovata"))
                return
            range_str = self.edit_export_range_var.get().strip()
            if range_str:
                if '-' in range_str:
                    parts = range_str.split('-')
                    try:
                        start = (int(parts[0]) - 1) if parts[0] else 0
                        end = int(parts[1]) if parts[1] else len(pages)
                    except ValueError:
                        messagebox.showerror(t("Errore"), t("Formato intervallo non valido"))
                        return
                else:
                    try:
                        n = int(range_str)
                        start = n - 1
                        end = n
                    except ValueError:
                        messagebox.showerror(t("Errore"), t("Numero non valido"))
                        return
                start = max(0, start)
                selected = pages[start:end]
            else:
                selected = pages
            if not selected:
                messagebox.showwarning(t("Attenzione"), t("Nessuna pagina nell'intervallo indicato"))
                return
            mode = self.edit_export_mode_var.get()
            want_orig = mode in ("Originale + Tradotto", "Solo originale")
            want_trans = mode in ("Originale + Tradotto", "Solo tradotto")
            suffix = {"Originale + Tradotto": "originale_tradotto", "Solo originale": "originale", "Solo tradotto": "tradotto"}[mode]
            prompt_mode = {"Originale + Tradotto": "both", "Solo originale": "orig", "Solo tradotto": "trans"}[mode]
            out_path = filedialog.asksaveasfilename(
                title=t("Salva testo come"),
                defaultextension=".txt",
                filetypes=[("File di testo", "*.txt")],
                initialfile=f"{comic or 'testo'}_{suffix}.txt",
            )
            if not out_path:
                return
            src_lang, dst_lang = export_prompt.langs_from_cfg(cfg)
            # Il prompt sta prima del primo marcatore === pagina ===, dove
            # l'import lo ignora: il file resta reimportabile com'e'.
            lines = [export_prompt.build(prompt_mode, comic, base, src_lang, dst_lang)]
            for page in selected:
                path = base / page / "translated.json"
                if not path.exists():
                    path = base / page / "ocr.json"
                if not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                lines.append(f"=== {page} ===")
                for det in data.get("detections", []):
                    bid = det.get("balloon_id", "?")
                    orig = det.get("testo_originale", "").replace("\n", " ")
                    trans = det.get("testo_tradotto", "").replace("\n", " ")
                    if want_orig:
                        lines.append(f"[{bid}] ORIGINALE: {orig}")
                    if want_trans:
                        lines.append(f"[{bid}] TRADOTTO: {trans}")
                lines.append("")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.edit_status.config(text=f"Esportato: {out_path} | {len(selected)} pagine")
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _import_edit_text(self):
        try:
            in_path = filedialog.askopenfilename(
                title=t("Importa testo da"),
                filetypes=[("File di testo", "*.txt")],
            )
            if not in_path:
                return
            cfg = load_cfg()
            work_dir = Path(cfg["paths"]["work_dir"])
            comic = self.edit_comic_var.get()
            if comic == "(root)":
                comic = ""
            base = (work_dir / comic) if comic else work_dir
            with open(in_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
            page_re = re.compile(r"===\s*(.+?)\s*===")
            line_re = re.compile(r"^\[(.+?)\]\s+(ORIGINALE|TRADOTTO):\s?(.*)$")
            updates = {}
            current_page = None

            def apply_segment(segment):
                m = line_re.match(segment)
                if m and current_page:
                    bid, kind, text = m.group(1), m.group(2), m.group(3)
                    page_updates = updates.setdefault(current_page, {})
                    balloon_updates = page_updates.setdefault(bid, {})
                    if kind == "ORIGINALE":
                        balloon_updates["testo_originale"] = text
                    else:
                        balloon_updates["testo_tradotto"] = text

            for raw in raw_lines:
                line = raw.rstrip("\n")
                remaining = line
                while True:
                    m = page_re.search(remaining)
                    if not m:
                        if remaining.strip():
                            apply_segment(remaining.strip())
                        break
                    before = remaining[:m.start()].strip()
                    if before:
                        apply_segment(before)
                    current_page = m.group(1)
                    remaining = remaining[m.end():].strip()
                    if not remaining:
                        break
            if not updates:
                messagebox.showwarning(t("Attenzione"), t("Nessun dato riconosciuto nel file"))
                return
            import translate_local
            pages_done = 0
            balloons_done = 0
            cached = 0
            for page, balloon_updates in updates.items():
                path = base / page / "translated.json"
                if not path.exists():
                    path = base / page / "ocr.json"
                if not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
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
                    # La traduzione importata entra in cache: e' una scelta
                    # umana e deve sopravvivere a un rilancio dello stage di
                    # traduzione, che altrimenti la ritradurrebbe da zero.
                    # Si usa il testo originale FINALE (l'import puo' aver
                    # corretto anche quello) e la chiave e' la stessa dello
                    # stage, glossario incluso.
                    if "testo_tradotto" in upd:
                        try:
                            if translate_local.remember(
                                path.parent, cfg,
                                det.get("testo_originale", ""), det.get("testo_tradotto", ""),
                            ):
                                cached += 1
                        except Exception as e:
                            log_msg = f"cache non aggiornata per balloon {bid} di {page}: {e}"
                            print(log_msg)
                if changed:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    pages_done += 1
            self.edit_status.config(text=f"Importato: {pages_done} pagine modificate, {balloons_done} balloon aggiornati, {cached} in cache")
            self._load_preview()
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _run_editor(self):
        comic = self.edit_comic_var.get()
        page = self.edit_page_var.get()
        if not page:
            messagebox.showwarning(t("Attenzione"), t("Seleziona una pagina"))
            return
        if comic == "(root)":
            comic = ""
        try:
            cfg = load_cfg()
            work_dir = Path(cfg["paths"]["work_dir"])
            cmd = [sys.executable, EDIT_SCRIPT, "--work-dir", str(work_dir), "--page", page]
            if comic:
                cmd.extend(["--comic", comic])
            self._log(f"Avvio editor: {' '.join(cmd)}")
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _load_preview(self):
        try:
            cfg = load_cfg()
            work_dir = Path(cfg["paths"]["work_dir"])
            comic = self.edit_comic_var.get()
            page = self.edit_page_var.get()
            if not page:
                return
            if comic == "(root)":
                comic = ""
            path = work_dir / comic / page / "translated.json" if comic else work_dir / page / "translated.json"
            if not path.exists():
                path = work_dir / comic / page / "ocr.json" if comic else work_dir / page / "ocr.json"
            if not path.exists():
                messagebox.showwarning(t("Attenzione"), f"Nessun file trovato per {page}")
                return
            with open(path, "r", encoding="utf-8") as f:
                self._current_edit_data = json.load(f)
            self.edit_tree.delete(*self.edit_tree.get_children())
            for det in self._current_edit_data.get("detections", []):
                bid = det.get("balloon_id", "?")
                orig = det.get("testo_originale", "")
                trans = det.get("testo_tradotto", "")
                if not orig and not trans:
                    status = "Scartato"
                elif not trans:
                    status = "Da tradurre"
                elif trans == "-":
                    status = "Onomatopea"
                else:
                    status = "OK"
                self.edit_tree.insert("", END, values=(bid, orig, trans, status), tags=(str(bid),))
            self._modified = False
            self.edit_status.config(text=f"Caricato: {path.name} | {len(self._current_edit_data.get('detections', []))} balloon")
        except Exception as e:
            messagebox.showerror(t("Errore"), str(e))

    def _on_tree_double_click(self, event):
        if not self._current_edit_data:
            return
        region = self.edit_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        column = self.edit_tree.identify_column(event.x)
        item = self.edit_tree.identify_row(event.y)
        if not item or column != "#3":
            return
        values = self.edit_tree.item(item, "values")
        current_text = values[2] if len(values) > 2 else ""
        x, y, width, height = self.edit_tree.bbox(item, column)
        self.editing_item = item
        self.editing_column = column
        self.edit_popup = Entry(self.edit_tree, width=width // 7)
        self.edit_popup.place(x=x, y=y, width=width, height=height)
        self.edit_popup.insert(0, current_text)
        self.edit_popup.select_range(0, END)
        self.edit_popup.focus()
        self.edit_popup.bind("<Return>", lambda e: self._confirm_edit())
        self.edit_popup.bind("<Escape>", lambda e: self._cancel_edit())
        self.edit_popup.bind("<FocusOut>", lambda e: self._confirm_edit())

    def _confirm_edit(self):
        if not self.edit_popup or not self.editing_item:
            return
        new_text = self.edit_popup.get().strip()
        values = list(self.edit_tree.item(self.editing_item, "values"))
        old_text = values[2] if len(values) > 2 else ""
        if new_text != old_text:
            values[2] = new_text
            if new_text == "-":
                values[3] = "Onomatopea"
            elif new_text:
                values[3] = "Modificato"
            else:
                values[3] = "Da tradurre"
            self.edit_tree.item(self.editing_item, values=values)
            self._modified = True
            self.edit_status.config(text=t("Modifiche non salvate"), foreground="orange")
        self._cancel_edit()

    def _cancel_edit(self):
        if self.edit_popup:
            self.edit_popup.destroy()
            self.edit_popup = None
            self.editing_item = None
            self.editing_column = None

    def _save_edits(self):
        if not self._current_edit_data or not self._modified:
            messagebox.showinfo(t("Info"), t("Nessuna modifica da salvare"))
            return
        try:
            tree_items = self.edit_tree.get_children()
            detections = self._current_edit_data.get("detections", [])
            for i, item in enumerate(tree_items):
                values = self.edit_tree.item(item, "values")
                if i < len(detections):
                    new_trans = values[2] if len(values) > 2 else ""
                    detections[i]["testo_tradotto"] = new_trans
                    if new_trans == "-":
                        detections[i]["testo_originale"] = ""
                        detections[i]["testo_tradotto"] = ""
            cfg = load_cfg()
            work_dir = Path(cfg["paths"]["work_dir"])
            comic = self.edit_comic_var.get()
            page = self.edit_page_var.get()
            if comic == "(root)":
                comic = ""
            out_path = work_dir / comic / page / "translated.json" if comic else work_dir / page / "translated.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self._current_edit_data, f, ensure_ascii=False, indent=2)
            self._modified = False
            self.edit_status.config(text=f"Salvato: {out_path}", foreground="green")
            self._log(f"Traduzioni salvate: {out_path}")
        except Exception as e:
            messagebox.showerror(t("Errore"), f"Errore salvataggio: {e}")

    def _mark_onomatopea(self):
        selection = self.edit_tree.selection()
        if not selection:
            messagebox.showwarning(t("Attenzione"), t("Seleziona un balloon"))
            return
        item = selection[0]
        values = list(self.edit_tree.item(item, "values"))
        values[2] = "-"
        values[3] = "Onomatopea"
        self.edit_tree.item(item, values=values)
        self._modified = True
        self.edit_status.config(text=t("Modifiche non salvate"), foreground="orange")

    def _restore_original(self):
        selection = self.edit_tree.selection()
        if not selection:
            messagebox.showwarning(t("Attenzione"), t("Seleziona un balloon"))
            return
        item = selection[0]
        values = list(self.edit_tree.item(item, "values"))
        original = values[1] if len(values) > 1 else ""
        values[2] = original
        values[3] = "Ripristinato"
        self.edit_tree.item(item, values=values)
        self._modified = True
        self.edit_status.config(text=t("Modifiche non salvate"), foreground="orange")

    def _open_review_window(self):
        if self._review_window is not None and self._review_window.win.winfo_exists():
            self._review_window.win.lift()
            self._review_window.win.focus_force()
            return
        self._review_window = ReviewWindow(self)

    def _refresh_cache_stats(self):
        try:
            stats = tc.stats()
            self.cache_stats.config(text=f"Entries: {stats['entries']} | File: {stats['file']}")
            self.cache_listbox.delete(0, END)
            keys = tc.all_keys() if hasattr(tc, "all_keys") else []
            if keys:
                for key in keys[:100]:
                    self.cache_listbox.insert(END, f"{key[:16]}...")
            else:
                self.cache_listbox.insert(END, "(cache vuota)")
        except Exception as e:
            self.cache_stats.config(text=f"Errore: {e}")

    def _clear_cache(self):
        if messagebox.askyesno(t("Conferma"), t("Svuotare tutta la cache?")):
            try:
                if hasattr(tc, "clear"):
                    tc.clear()
                self._log("Cache svuotata")
                self._refresh_cache_stats()
            except Exception as e:
                messagebox.showerror(t("Errore"), str(e))

if __name__ == "__main__":
    root = Tk()
    app = PipelineGUI(root)
    root.mainloop()
