"""
Localizzazione dell'interfaccia (italiano / inglese).

La chiave del dizionario e' la stringa italiana cosi' com'e' scritta nel
codice: in italiano `t()` la restituisce identica (nessun costo, nessun
rischio di chiave sbagliata), in inglese cerca la traduzione e, se manca,
ricade sull'italiano invece di mostrare una chiave criptica. Cosi' una
stringa nuova aggiunta a una GUI funziona subito, e resta solo da tradurre.

Riguarda la sola interfaccia. I messaggi di log della pipeline restano in
italiano in entrambe le lingue: sono diagnostica, non testo d'interfaccia.
"""
import re

LINGUE = {"it": "Italiano", "en": "English"}
LINGUA_DEFAULT = "it"

_lingua = LINGUA_DEFAULT


def set_language(codice: str) -> str:
    """Imposta la lingua corrente. Un codice sconosciuto ricade sul default
    invece di far fallire l'avvio della GUI."""
    global _lingua
    _lingua = codice if codice in LINGUE else LINGUA_DEFAULT
    return _lingua


def get_language() -> str:
    return _lingua


def language_from_config(cfg: dict) -> str:
    return set_language((cfg.get("ui") or {}).get("language", LINGUA_DEFAULT))


def t(testo: str) -> str:
    """Traduce una stringa d'interfaccia nella lingua corrente."""
    if _lingua == "it" or not testo:
        return testo
    tradotta = _EN.get(testo)
    if tradotta is not None:
        return tradotta
    # Etichette costruite come "Prefisso: valore": si prova a tradurre il
    # solo prefisso, cosi' "Fumetto: X" funziona senza una voce per ogni X.
    testo_pulito = testo.strip()
    if testo_pulito in _EN:
        return testo.replace(testo_pulito, _EN[testo_pulito])
    return testo


# ── Tkinter ───────────────────────────────────────────────────────────────

def translate_tk_tree(widget) -> None:
    """Traduce il testo gia' impostato sui widget Tk di un albero.

    Le GUI costruiscono le loro etichette una sola volta all'avvio: invece di
    avvolgere ~200 letterali con t(), si passa sull'albero finito e si
    traduce quello che c'e'. Le stringhe decise a runtime (messaggi di
    errore, stati) restano avvolte in t() nel punto in cui nascono.
    """
    if _lingua == "it":
        return
    try:
        import tkinter.ttk as ttk
    except Exception:
        ttk = None

    def visita(w):
        try:
            if "text" in w.keys():
                attuale = w.cget("text")
                if isinstance(attuale, str) and attuale:
                    nuovo = t(attuale)
                    if nuovo != attuale:
                        w.configure(text=nuovo)
        except Exception:
            pass  # widget senza opzione text, o gia' distrutto
        try:
            if "values" in w.keys():
                valori = w.cget("values")
                if isinstance(valori, (list, tuple)) and valori:
                    tradotti = [t(str(v)) for v in valori]
                    if tradotti != [str(v) for v in valori]:
                        w.configure(values=tradotti)
        except Exception:
            pass
        if ttk is not None and isinstance(w, ttk.Notebook):
            try:
                for i in range(w.index("end")):
                    w.tab(i, text=t(w.tab(i, "text")))
            except Exception:
                pass
        for figlio in w.winfo_children():
            visita(figlio)

    visita(widget)


# ── Template HTML della web app ───────────────────────────────────────────

# Si traduce solo il testo fra i tag e gli attributi title/placeholder: mai
# per sostituzione libera sull'HTML, che colpirebbe anche id, classi e
# codice JavaScript (una chiave corta come "N" o "Output" farebbe danni).
_RE_TESTO = re.compile(r'>(\s*)([^<>{}\n][^<>{}]*?)(\s*)<')
_RE_ATTR = re.compile(r'\b(title|placeholder)="([^"{}]+)"')

_cache_html: dict[tuple[int, str], str] = {}


def translate_html(html: str) -> str:
    """Traduce un template HTML nella lingua corrente (risultato in cache:
    il template e' una costante di modulo, si traduce una volta per lingua)."""
    if _lingua == "it":
        return html
    chiave = (id(html), _lingua)
    if chiave in _cache_html:
        return _cache_html[chiave]

    def sost_testo(m):
        return ">" + m.group(1) + t(m.group(2)) + m.group(3) + "<"

    def sost_attr(m):
        return f'{m.group(1)}="{t(m.group(2))}"'

    tradotto = _RE_ATTR.sub(sost_attr, _RE_TESTO.sub(sost_testo, html))
    _cache_html[chiave] = tradotto
    return tradotto


# ── Dizionario ────────────────────────────────────────────────────────────

_EN = {
    # Titoli e tab
    "Pipeline Traduzione Fumetti": "Comic Translation Pipeline",
    "🎌 Pipeline Traduzione Fumetti": "🎌 Comic Translation Pipeline",
    "Pipeline Fumetti": "Comic Pipeline",
    "Pipeline": "Pipeline",
    "Pretrattamento": "Pretreatment",
    "Configurazione": "Settings",
    "Config": "Settings",
    "Correzioni": "Corrections",
    "Revisione": "Review",
    "Cache": "Cache",
    "Revisione fumetto - Interattiva": "Comic review - interactive",

    # Stage
    "Stage da eseguire": "Stage to run",
    "Stage": "Stage",
    "Pipeline COMPLETA": "FULL pipeline",
    "Solo OCR": "OCR only",
    "Solo Traduzione": "Translation only",
    "Solo Pulizia": "Cleaning only",
    "Solo Render (testo)": "Render only (text)",
    "Solo Render (Scrittura testo)": "Render only (text drawing)",
    "Salta traduzione (usa il testo OCR cosi' com'e', fumetto gia' in italiano)":
        "Skip translation (use the OCR text as is, comic already in the target language)",

    # Opzioni pipeline
    "Opzioni": "Options",
    "Limite pagine:": "Page limit:",
    "(vuoto = tutte, N = pagina N, N-M = intervallo pagine N-M)":
        "(empty = all, N = page N, N-M = pages N to M)",
    "(vuoto = tutte, N = pagina N, N-M = intervallo)":
        "(empty = all, N = page N, N-M = range)",
    "Fumetto:": "Comic:",
    "Backend OCR:": "OCR backend:",
    "OCR via:": "OCR via:",
    "Modello di visione usato dallo stage OCR": "Vision model used by the OCR stage",
    "Forza ritraduzione (ignora cache)": "Force re-translation (ignore cache)",
    "Ignora la cache traduzioni e richiama sempre l'AI, anche per pagine già tradotte":
        "Ignore the translation cache and always call the model, even for pages already translated",
    "Qwen3-VL-8B": "Qwen3-VL-8B",
    "PaddleOCR-VL-1.6 (veloce)": "PaddleOCR-VL-1.6 (fast)",
    "Intervallo pagine (es. 5 o 3-10):": "Page range (e.g. 5 or 3-10):",
    "Pagine (es. 5 o 3-10):": "Pages (e.g. 5 or 3-10):",
    "tutti": "all",
    "tutte": "all",
    "Avvia": "Run",
    "▶ Avvia": "▶ Run",
    "Ferma": "Stop",
    "■ Ferma": "■ Stop",
    "Aggiorna stato": "Refresh status",
    "Aggiorna": "Refresh",
    "Aggiorna stats": "Refresh stats",

    # Stato / progressione
    "Stato Attuale": "Current status",
    "Stato": "Status",
    "Progressione": "Progress",
    "In attesa...": "Waiting...",
    "In attesa": "Waiting",
    "Inizializzazione...": "Initialising...",
    "Caricamento...": "Loading...",
    "Caricamento…": "Loading…",
    "Completato!": "Done!",
    "0/0 pagine": "0/0 pages",
    "Traduttore: -": "Translator: -",
    "Traduttore: modello locale (llama-server)": "Translator: local model (llama-server)",
    "llama-server: -": "llama-server: -",
    "llama-server: NON TROVATO nel PATH": "llama-server: NOT FOUND on PATH",
    "llama-server: pronto (viene avviato all'inizio dello stage)":
        "llama-server: ready (started at the beginning of the stage)",
    "Output": "Output",

    # Pretrattamento
    "Pretrattamento pagine (ComfyUI)": "Page pretreatment (ComfyUI)",
    "▶ Avvia pretrattamento": "▶ Run pretreatment",
    "Passa le pagine grezze attraverso ComfyUI/Flux-Klein con un prompt libero,\n"
    "PRIMA di OCR - utile per upscale/pulizia rumore/altro. Non tocca mai gli\n"
    "originali in input_pages/. Il prompt resta salvato per la volta dopo.":
        "Runs the raw pages through ComfyUI/Flux-Klein with a free-text prompt,\n"
        "BEFORE OCR - useful for upscaling, denoising and the like. It never touches\n"
        "the originals in input_pages/. The prompt is remembered for next time.",
    "Cartella input:": "Input folder:",
    "Cartella di partenza del pretrattamento": "Pretreatment source folder",
    "Cartella di destinazione del pretrattamento": "Pretreatment destination folder",
    "Salva in:": "Save to:",
    "Le pagine finiscono in <destinazione>/<fumetto>/. "
    "Default: pretreated_pages/ accanto alla cartella input.":
        "Pages are written to <destination>/<comic>/. "
        "Default: pretreated_pages/ next to the input folder.",
    "Pagine": "Pages",
    "Tutto il fumetto": "Whole comic",
    "Pagina singola:": "Single page:",
    "Intervallo, da:": "Range, from:",
    "a:": "to:",
    "(vuoto = fino all'ultima pagina)": "(empty = through the last page)",
    "Piena risoluzione (bypassa il resize del workflow)":
        "Full resolution (bypass the workflow's resize)",
    "Megapixel:": "Megapixels:",
    "(vuoto = come salvato nel workflow)": "(empty = as saved in the workflow)",
    "A piena risoluzione Flux vede la pagina intera senza rimpicciolirla, ma serve\n"
    "molta piu' VRAM ed e' fuori dalla risoluzione su cui il modello e' addestrato:\n"
    "se va in OOM o peggiora, alza i megapixel invece di togliere del tutto il resize.":
        "At full resolution Flux sees the whole page without shrinking it, but it needs\n"
        "far more VRAM and is outside the resolution the model was trained on: if it runs\n"
        "out of memory or the result gets worse, raise the megapixels instead of removing\n"
        "the resize altogether.",
    "Prompt:": "Prompt:",
    "Scrivi un prompt prima di avviare il pretrattamento":
        "Write a prompt before starting the pretreatment",
    "Es: upscale and sharpen the image, keep composition and colors unchanged...":
        "E.g. upscale and sharpen the image, keep composition and colors unchanged...",

    # Configurazione
    "Directory di lavoro": "Working directory",
    "Root:": "Root:",
    "Seleziona directory di lavoro": "Select working directory",
    "Sfoglia...": "Browse...",
    "Sfoglia…": "Browse…",
    "Modelli GGUF (llama-server)": "GGUF models (llama-server)",
    "Path dei .gguf che llama-server carica per lo stage OCR e per la traduzione.\n"
    "Un modello per volta: il server viene avviato all'inizio dello stage e chiuso\n"
    "alla fine, cosi' la VRAM torna libera per la pulizia.":
        "Paths of the .gguf files llama-server loads for the OCR stage and for translation.\n"
        "One model at a time: the server starts at the beginning of the stage and is shut\n"
        "down at the end, so the VRAM is free again for cleaning.",
    "Path dei .gguf che llama-server carica per OCR e traduzione. Un modello per volta:\n"
    "    il server parte all'inizio dello stage e viene chiuso alla fine, cosi' la VRAM\n"
    "    torna libera per la pulizia.":
        "Paths of the .gguf files llama-server loads for OCR and translation. One model at\n"
        "    a time: the server starts at the beginning of the stage and is shut down at the\n"
        "    end, so the VRAM is free again for cleaning.",
    "Modello OCR (qwen)": "OCR model (qwen)",
    "Modello OCR (paddleocr_vl)": "OCR model (paddleocr_vl)",
    "Proiettore multimodale (mmproj)": "Multimodal projector (mmproj)",
    "Modello traduzione": "Translation model",
    "Porta llama-server:": "llama-server port:",
    "Porta llama-server": "llama-server port",
    "Seleziona un modello GGUF": "Select a GGUF model",
    "Salva modelli": "Save models",
    "System Prompt traduzione": "Translation system prompt",
    "Dimensione font fissa": "Fixed font size",
    "Se impostata, forza questa dimensione (px) su tutti i balloon normali di\n"
    "ogni pagina, al posto del calcolo automatico. Lascia vuoto per l'automatico.":
        "If set, forces this size (px) on every normal balloon of every page,\n"
        "instead of the automatic calculation. Leave empty for automatic.",
    "Se impostata, forza questa dimensione (px) su tutti i balloon normali di\n"
    "    ogni pagina, al posto del calcolo automatico. Lascia vuoto per l'automatico.":
        "If set, forces this size (px) on every normal balloon of every page,\n"
        "    instead of the automatic calculation. Leave empty for automatic.",
    "Dimensione (px):": "Size (px):",
    "Salva dimensione": "Save size",
    "(vuoto = automatico)": "(empty = automatic)",
    "automatico": "automatic",
    "Visualizza config.yaml": "View config.yaml",
    "config.yaml": "config.yaml",
    "Lingua interfaccia": "Interface language",
    "Lingua:": "Language:",
    "Lingua dell'interfaccia. I messaggi di log della pipeline restano in italiano.":
        "Language of the interface. The pipeline log messages stay in Italian.",
    "Salva lingua": "Save language",
    "Lingua salvata. Riavvia il programma per applicarla ovunque.":
        "Language saved. Restart the program to apply it everywhere.",
    "Lingua salvata: riavvia la pagina per applicarla.":
        "Language saved: reload the page to apply it.",

    # Correzioni
    "Esporta / Importa testo TXT": "Export / import text (TXT)",
    "Esporta / Importa testo (TXT)": "Export / import text (TXT)",
    "Esporta tutte le pagine del fumetto selezionato (o un intervallo) in un unico file\n"
    "    TXT per correggerlo con un editor esterno, poi reimportalo per riportare le modifiche\n"
    "    nei translated.json. L'import scrive direttamente sui file: se la pagina caricata sopra\n"
    "    e' tra quelle importate, ricaricala per vedere le modifiche.":
        "Exports every page of the selected comic (or a range) into a single TXT file so you\n"
        "    can correct it in an external editor, then import it back to write the changes into\n"
        "    the translated.json files. The import writes to the files directly: if the page loaded\n"
        "    above is among those imported, reload it to see the changes.",
    "Pagine:": "Pages:",
    "Contenuto:": "Content:",
    "Originale + Tradotto": "Original + translated",
    "Solo originale": "Original only",
    "Solo tradotto": "Translation only",
    "Originale": "Original",
    "Traduzione": "Translation",
    "Esporta in TXT...": "Export to TXT...",
    "Esporta in TXT…": "Export to TXT…",
    "Importa da TXT...": "Import from TXT...",
    "Importa da TXT…": "Import from TXT…",
    "Salva testo come": "Save text as",
    "Importa testo da": "Import text from",
    "Pagina:": "Page:",
    "Carica": "Load",
    "Salva modifiche": "Save changes",
    "Salva": "Save",
    "Editor traduzioni (doppio-click per modificare)":
        "Translation editor (double-click to edit)",
    "ID": "ID",
    "Testo Originale": "Original text",
    "Originale:": "Original:",
    "Apri editor CLI": "Open CLI editor",
    "Marca come onomatopea (-)": "Mark as sound effect (-)",
    "Marca come onomatopea (svuota la traduzione)":
        "Mark as sound effect (clears the translation)",
    "Ripristina originale": "Restore original",
    "Ripristina il testo originale come traduzione":
        "Restore the original text as the translation",
    "Azioni": "Actions",

    # Revisione
    "Apri finestra di Revisione": "Open review window",
    "La revisione visuale del fumetto si apre in una finestra dedicata\n"
    "a schermo pieno, per avere piu' spazio per leggere e correggere.":
        "The visual review opens in its own full-screen window,\n"
        "for more room to read and correct.",
    "🔍 Controlla fumetto": "🔍 Check comic",
    "Controlla tutte le pagine del fumetto: testi rimasti in lingua originale, "
    "da trascrivere, troncati, stage mancanti":
        "Checks every page of the comic: text left in the source language, "
        "text to transcribe, truncated text, missing stages",
    "Controllo fumetto": "Comic check",
    "Controllo del fumetto in corso...": "Checking the comic...",
    "Doppio click su una riga per aprire la pagina sul balloon indicato.":
        "Double-click a row to open the page at that balloon.",
    "Nessun problema": "No problems",
    "— nessun problema in questa pagina": "— no problems on this page",
    "⚠ Prossimo problema": "⚠ Next problem",
    "prossimo problema ·": "next problem ·",
    "Salta al prossimo balloon da trascrivere o con testo troncato (tasto N)":
        "Jump to the next balloon to transcribe or with truncated text (key N)",
    "cambia pagina": "change page",
    "< Precedente": "< Previous",
    "‹ Precedente": "‹ Previous",
    "Successiva >": "Next >",
    "Successiva ›": "Next ›",
    "Adatta": "Fit",
    "Chiudi": "Close",
    "⟳ Rigenera pagina": "⟳ Regenerate page",
    " Rigenera pagina": " Regenerate page",
    " Rigenerazione in corso...": " Regenerating...",
    "⏳ Rigenerazione in corso...": "⏳ Regenerating...",
    "➕ Nuovo box": "➕ New box",
    "✖ Annulla nuovo box": "✖ Cancel new box",
    "🗑 Elimina box": "🗑 Delete box",
    "Elimina box": "Delete box",
    "✂ Dividi in due": "✂ Split in two",
    "🔁 Ricrea balloon (bianco + bordo)": "🔁 Rebuild balloon (white + outline)",
    "🔁 Ricrea balloon": "🔁 Rebuild balloon",
    "Reimposta riempimento bianco + bordo nero: usa questo se la pulizia automatica "
    "ha cancellato l'intero balloon, non solo il testo":
        "Resets to white fill + black outline: use this if the automatic cleaning "
        "wiped the whole balloon, not just the text",
    "Riempi box": "Fill box",
    "Bordo nero": "Black outline",
    "Contorno": "Outline",
    "Colore contorno": "Outline colour",
    "Colore di riempimento": "Fill colour",
    "Colore del testo": "Text colour",
    "Colore testo manuale (altrimenti auto nero/bianco)":
        "Manual text colour (otherwise automatic black/white)",
    "Colore testo manuale": "Manual text colour",
    "(altrimenti auto nero/bianco)": "(otherwise automatic black/white)",
    "Ellisse": "Ellipse",
    "Forma a ellisse (testo nel rettangolo inscritto, non nel bbox intero)":
        "Ellipse shape (text in the inscribed rectangle, not the whole bbox)",
    "Segui sagoma balloon (doppio/clessidra)": "Follow balloon shape (double/hourglass)",
    "Codino:": "Tail:",
    "Nessuno": "None",
    "Su": "Up",
    "Giù": "Down",
    "Sinistra": "Left",
    "Destra": "Right",
    "Su-destra": "Up-right",
    "Su-sinistra": "Up-left",
    "Giù-destra": "Down-right",
    "Giù-sinistra": "Down-left",
    "Stile:": "Style:",
    "Grassetto": "Bold",
    "Corsivo": "Italic",
    "Sottolin.": "Underl.",
    "Allineamento:": "Alignment:",
    "Interlinea:": "Line spacing:",
    "Dim.:": "Size:",
    "Sp.:": "Sp.:",
    "Applica": "Apply",
    "Imposta formattazione per tutti i balloon": "Apply formatting to every balloon",
    "Formattazione applicata a tutti i balloon della pagina.":
        "Formatting applied to every balloon on the page.",
    "Seleziona un balloon": "Select a balloon",
    "Seleziona una pagina": "Select a page",
    "Trascina sul canvas per disegnare il nuovo box.":
        "Drag on the canvas to draw the new box.",
    "Box troppo piccolo, riprova trascinando un'area piu' ampia.":
        "Box too small, try again dragging a larger area.",
    "Eliminare questo balloon dalla pagina?": "Delete this balloon from the page?",
    "Pagina non disponibile": "Page not available",
    "Nessuna pagina renderizzata trovata.": "No rendered page found.",
    "⚠ L'OCR non ha letto nulla ma nel balloon c'e' del testo:\n"
    "trascrivilo qui, oppure scrivi - se e' un'insegna o un'onomatopea":
        "⚠ OCR read nothing but there is text in the balloon:\n"
        "type it here, or write - if it is a sign or a sound effect",
    "✂ Nel render questo testo non ci stava ed e' stato troncato:\n"
    "accorcialo, allarga il box o imposta una dimensione font piu' piccola":
        "✂ This text did not fit when rendered and was truncated:\n"
        "shorten it, widen the box, or set a smaller font size",
    "⚠️ Modifiche in sospeso. Clicca 'Rigenera pagina' per applicarle.":
        "⚠️ Pending changes. Click 'Regenerate page' to apply them.",
    '⚠ Modifiche in sospeso. Clicca "Rigenera pagina" per applicarle.':
        '⚠ Pending changes. Click "Regenerate page" to apply them.',
    "Ci sono modifiche non rigenerate. Vuoi chiudere comunque, perdendole?":
        "There are changes that have not been regenerated. Close anyway and lose them?",
    "Ci sono modifiche non rigenerate. Vuoi scartarle e continuare?":
        "There are changes that have not been regenerated. Discard them and continue?",
    "Modifiche non salvate": "Unsaved changes",
    "Nessuna modifica da salvare": "No changes to save",
    "🖱️ Clicca e trascina un balloon per spostarlo | Trascina gli handle per ridimensionare "
    "| Doppio-click per modificare il testo | ➕ Nuovo box per aggiungerne uno, poi Canc/🗑 "
    "per eliminare quello selezionato | 🔴 Rosso ⚠ = testo mancante, da trascrivere | 🟠 "
    "Arancione ✂ = testo troncato, da accorciare | 🔎 Rotella o frecce ↑↓ per zoomare | ✋ "
    "Trascina area vuota (o tasto centrale) per scorrere":
        "🖱️ Click and drag a balloon to move it | Drag the handles to resize | Double-click "
        "to edit the text | ➕ New box to add one, then Del/🗑 to remove the selected one | "
        "🔴 Red ⚠ = missing text, to transcribe | 🟠 Orange ✂ = truncated text, to shorten | "
        "🔎 Wheel or ↑↓ arrows to zoom | ✋ Drag an empty area (or middle button) to pan",
    "🖱️ Trascina un balloon per spostarlo · trascina gli angoli/lati per ridimensionare "
    "· doppio-click per modificare testo e font ·":
        "🖱️ Drag a balloon to move it · drag the corners/edges to resize "
        "· double-click to edit text and font ·",
    "PagSu/PagGiù": "PgUp/PgDn",

    # Cache
    "Statistiche cache: -": "Cache statistics: -",
    "Entries in cache": "Cache entries",
    "Svuota cache": "Clear cache",
    "Svuotare tutta la cache?": "Clear the whole cache?",
    "Pulisci": "Clear",

    # Messaggi comuni
    "Errore": "Error",
    "Attenzione": "Warning",
    "Info": "Info",
    "Conferma": "Confirm",
    "Fumetto non valido": "Invalid comic",
    "Nessuna pagina trovata": "No page found",
    "Nessuna pagina nell'intervallo indicato": "No page in the given range",
    "Nessun dato riconosciuto nel file": "No data recognised in the file",
    "Formato intervallo non valido": "Invalid range format",
    "Intervallo di pagine non valido": "Invalid page range",
    "Numero non valido": "Invalid number",
    "Numero di pagina non valido": "Invalid page number",
    "Porta non valida": "Invalid port",
    "Megapixel non validi: usa un numero maggiore di 0, oppure spunta 'Piena risoluzione'.":
        "Invalid megapixels: use a number greater than 0, or tick 'Full resolution'.",
    "Inserisci un numero intero positivo, o lascia vuoto":
        "Enter a positive whole number, or leave empty",
    "Input e destinazione coincidono: sovrascriverebbe gli originali.":
        "Input and destination are the same: it would overwrite the originals.",
    "✓ OK": "✓ OK",
    "✗ Annulla": "✗ Cancel",
    "...": "...",
    "🔁": "🔁",
    "🔍+": "🔍+",
    "🔍−": "🔍−",
    "N": "N",
    "0%": "0%",
    "100%": "100%",
    # Stringhe del template web, con la loro indentazione originale
    "Passa le pagine grezze (fumetto/intervallo scelti sopra) attraverso ComfyUI/Flux-Klein\n"
    "    con un prompt libero, PRIMA di OCR \u2014 utile per upscale/pulizia rumore/altro. Scrive in una\n"
    "    cartella separata (pretreated_pages/), non tocca mai gli originali in input_pages/. Il\n"
    "    prompt qui sotto resta salvato per la prossima volta.":
        "Runs the raw pages (comic/range selected above) through ComfyUI/Flux-Klein\n"
        "    with a free-text prompt, BEFORE OCR \u2014 useful for upscaling, denoising and the like.\n"
        "    It writes to a separate folder (pretreated_pages/) and never touches the originals\n"
        "    in input_pages/. The prompt below is remembered for next time.",
    "\u26a0 L'OCR non ha letto nulla,\n"
    "         ma nel ritaglio c'e' del testo: trascrivilo qui (o scrivi":
        "\u26a0 OCR read nothing,\n"
        "         but there is text in the crop: type it here (or write",
    "se e'\n         un'insegna/onomatopea da lasciare com'e').":
        "if it is\n         a sign or sound effect to leave as is).",
    "\u2702 Nel render questo testo\n"
    "         non ci stava ed e' stato troncato: accorcialo, oppure allarga il box o\n"
    "         imposta una dimensione font piu' piccola qui sotto.":
        "\u2702 This text did not fit when\n"
        "         rendered and was truncated: shorten it, or widen the box, or set a smaller\n"
        "         font size below.",
    "Pipeline COMPLETA (OCR -> Traduzione -> Pulizia -> Render)":
        "FULL pipeline (OCR -> translation -> cleaning -> render)",
    "Traduzione:": "Translation:",
    "OCR (qwen):": "OCR (qwen):",
    "OCR (paddleocr_vl):": "OCR (paddleocr_vl):",
    "Proiettore OCR (mmproj):": "OCR projector (mmproj):",
    # Messaggi costruiti lato Python dalla web app
    "Modelli salvati": "Models saved",
    "Lingua non valida": "Invalid language",
    "Dimensione font fissa disattivata (automatico)": "Fixed font size turned off (automatic)",
    "Cache svuotata. Entries: 0": "Cache cleared. Entries: 0",
    "Pipeline già in esecuzione": "Pipeline already running",
    "Pipeline già in esecuzione": "Pipeline already running",
    "Prompt pretrattamento vuoto": "Pretreatment prompt is empty",
    "Nessun file caricato": "No file uploaded",
    "Prompt vuoto, non salvato": "Empty prompt, not saved",
    "Parametro non valido": "Invalid parameter",
}
