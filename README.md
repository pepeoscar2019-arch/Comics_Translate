# Fumetti — pipeline di traduzione fumetti, interamente locale

Traduce le pagine di un fumetto da una lingua all'altra senza mandare nulla in
rete: rileva i balloon, ne legge il testo, lo traduce, cancella l'originale
dalla pagina e ci scrive sopra la traduzione impaginata.

Tutti i modelli girano sulla tua macchina — nessuna API, nessuna chiave, nessun
servizio esterno.

> **Interfaccia in italiano o inglese**, dalla tab Configurazione.
> I messaggi di log della pipeline restano in italiano.

---

## Come funziona

La pipeline è divisa in stage, eseguibili singolarmente o in sequenza:

| Stage | Cosa fa | Cosa usa |
|---|---|---|
| `ocr` | rileva i balloon e ne trascrive il testo | comic-text-detector + YOLO, poi un modello di visione su llama-server |
| `translate` | traduce le battute, con cache e glossario per fumetto | llama-server |
| `clean` | cancella il testo originale dai balloon | riempimento deterministico, su CPU |
| `render` | scrive la traduzione impaginata nel balloon | Pillow |

Due dettagli che spiegano parecchie scelte del codice:

- **llama-server viene avviato all'inizio di uno stage e chiuso alla fine.** È
  la chiusura del processo a liberare la VRAM per lo stage successivo: non c'è
  nessun passaggio di "scarica il modello".
- **La pulizia dei balloon non usa un modello generativo.** Riempie l'area del
  balloon con il colore di sfondo dominante, preservando il contorno. È
  deterministica, riproducibile e non ha bisogno di VRAM. Backend basati su
  inpainting sono stati provati e rimossi: rigeneravano l'intera pagina e
  introducevano derive visibili.

Oltre agli stage ci sono un **pretrattamento** facoltativo (passa le pagine
grezze attraverso ComfyUI prima dell'OCR, per upscale o pulizia del rumore) e
una **Revisione** interattiva per correggere a mano testo, posizione e
formattazione dei balloon prima del render finale.

## Interfacce

Le stesse funzioni sono disponibili in tre modi:

```bash
python main.py --stage full --comic "Nome del fumetto"   # riga di comando
python pipeline_gui_tk.py                                # GUI desktop (Tkinter)
python web_app.py                                        # web app locale
```

---

## Requisiti

- **Python 3.12+** (sviluppato su 3.14)
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** con `llama-server` nel
  PATH. Su Windows: `winget install ggml.llamacpp`
- Una GPU aiuta molto ma non è obbligatoria: `llama-server` sceglie da solo
  quanti layer tenere in VRAM.
- **ComfyUI** — solo se vuoi il pretrattamento. Tutto il resto funziona senza.

### Modelli da scaricare

Nessun modello è incluso nel repository. Servono file GGUF, per esempio:

| Ruolo | Modello usato nello sviluppo |
|---|---|
| OCR (visione) | `Qwen3-VL-8B-Instruct` (+ il suo `mmproj`) oppure `PaddleOCR-VL-1.6` |
| Traduzione | `translategemma-27b-it` (quantizzazione Q4_K_S) |

Vanno bene anche altri modelli: i percorsi si impostano in `config.yaml` o
dalla tab Configurazione.

---

## Installazione

Il progetto usa **tre ambienti virtuali separati**, perché il rilevamento dei
balloon ha bisogno di versioni di torch che non convivono con il resto.
`detect.py` invoca gli altri due interpreti come processi separati.

```bash
git clone <questo-repo> fumetti
cd fumetti

# 1. ambiente principale (pipeline, GUI, web app)
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. rilevamento del testo — richiede comic-text-detector
git clone https://github.com/dmMaze/comic-text-detector
python -m venv venv_ctd
venv_ctd/Scripts/python.exe -m pip install -r requirements-ctd.txt

# 3. segmentazione dei balloon
python -m venv venv_bubbleseg
venv_bubbleseg/Scripts/python.exe -m pip install -r requirements-bubbleseg.txt
```

Su Windows c'è anche `setup_windows.ps1`, che fa gli stessi passaggi.

Servono poi i **pesi dei modelli di rilevamento**, non inclusi nel repository:

**`comic-text-detector/data/comictextdetector.pt.onnx`** — rilevamento dei
riquadri di testo. Si scarica dai rilasci del
[repository originale](https://github.com/dmMaze/comic-text-detector).

**`bubble_seg/model.pt`** — segmentazione dei balloon, YOLOv8m-seg addestrato
sulla singola classe `speech bubble`:

```bash
# richiede git-lfs
git clone https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble
cp yolov8m_seg-speech-bubble/model.pt bubble_seg/model.pt
```

Oppure scaricando il solo file da
[huggingface.co/kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
(`model.pt`, 54.809.749 byte,
SHA-256 `200141c4…50ed610`). È distribuito sotto **GPL-3.0**: usarlo come
modello, come fa questa pipeline, non impone nulla al codice qui dentro, ma
tienilo presente se lo ridistribuisci.

Va bene qualunque altro modello di segmentazione dei balloon in formato
ultralytics: il percorso si imposta in `config.yaml`, sotto `bubble_seg`.

### Configurazione

```bash
cp config.example.yaml config.yaml
```

Poi apri `config.yaml` e imposta i percorsi: cartelle di lavoro (`paths`) e file
GGUF (`llama_server`). Il file di esempio commenta ogni sezione.

`config.yaml` è in `.gitignore`: le impostazioni locali non finiscono nel
repository.

---

## Uso

Metti le pagine in una sottocartella di `input_dir`, una per fumetto:

```
input_pages/
  Il mio fumetto - Capitolo 1/
    001.jpg
    002.jpg
```

Poi:

```bash
# tutto il fumetto, tutti gli stage
venv/Scripts/python.exe main.py --stage full --comic "Il mio fumetto - Capitolo 1"

# solo alcune pagine
venv/Scripts/python.exe main.py --stage ocr --comic "..." --start 0 --end 5

# controlla cosa manca prima di considerarlo finito
venv/Scripts/python.exe main.py --stage audit --comic "..."
```

Le pagine finite finiscono in `output_dir`, lo stato intermedio in `work_dir`
(un `translated.json` per pagina, modificabile a mano).

### Rifinire il risultato

Il render automatico non azzecca tutto. La tab **Revisione** apre la pagina
finita con i balloon come riquadri modificabili: sposta, ridimensiona, correggi
il testo, cambia font e allineamento, poi rigenera la singola pagina. Segnala
da sola i balloon dove l'OCR non ha letto nulla e quelli dove il testo è stato
troncato.

La tab **Correzioni** esporta tutte le battute di un fumetto in un file TXT da
correggere con un editor esterno, e le reimporta. Le correzioni fatte a mano
finiscono nella cache di traduzione, così non vengono sovrascritte se rilanci
lo stage.

---

## Struttura

```
main.py               orchestrazione degli stage
detect.py             rilevamento balloon (invoca i due venv separati)
ocr.py                trascrizione via modello di visione
translate_local.py    traduzione, cache, glossario
clean.py              cancellazione del testo dal balloon
render.py             impaginazione e disegno del testo tradotto
balloon_shape.py      geometria dei balloon (fusi, con codino, ecc.)
fit_check.py          riscrittura piu' corta quando il testo non entra
audit.py              controllo di cosa resta da sistemare
i18n.py               localizzazione dell'interfaccia
pipeline_gui_tk.py    GUI desktop
web_app.py            web app locale
flux_pretreat.py      pretrattamento facoltativo via ComfyUI
```

---

## Software di terze parti

| Componente | Licenza | Come viene usato |
|---|---|---|
| [comic-text-detector](https://github.com/dmMaze/comic-text-detector) | GPL-3.0 | clonato a parte, eseguito come processo separato nel suo venv |
| [ultralytics](https://github.com/ultralytics/ultralytics) (YOLO) | AGPL-3.0 | installato in un venv separato, eseguito come processo separato |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | installato a parte, usato come server HTTP |
| ComfyUI | GPL-3.0 | facoltativo, usato come server HTTP per il solo pretrattamento |
| [yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble) | GPL-3.0 | pesi del modello di segmentazione, scaricati a parte |

Nessuno di questi è incluso nel repository, e nessuno è importato dal codice di
questo progetto: sono programmi separati, invocati via subprocess o via HTTP.
Se li integri diversamente, verifica gli obblighi delle rispettive licenze.

I font in `fonts/` hanno licenze proprie: controllale prima di ridistribuirli.

## Nota d'uso

Questo è uno strumento per tradurre fumetti in locale, pensato per materiale di
cui hai il diritto di fare una traduzione — opere di pubblico dominio, lavori
propri, o materiale per cui hai il permesso. Non incoraggia né facilita la
redistribuzione di opere protette da copyright.

## Licenza

Vedi [LICENSE](LICENSE).
