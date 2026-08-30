# Comics_Translate — a fully local comic translation pipeline

*[Versione italiana](README.it.md)*

Translates comic pages from one language to another without sending anything
over the network: it finds the speech balloons, reads the text inside them,
translates it, erases the original from the page and typesets the translation
back into the balloon.

Every model runs on your own machine — no API, no keys, no external service.

> **The interface is available in English or Italian**, from the Settings tab.
> The pipeline's log messages stay in Italian.

---

## How it works

The pipeline is split into stages, which can be run one at a time or in
sequence:

| Stage | What it does | What it uses |
|---|---|---|
| `ocr` | finds the balloons and transcribes their text | comic-text-detector + YOLO, then a vision model on llama-server |
| `translate` | translates the lines, with a cache and a per-comic glossary | llama-server |
| `clean` | erases the original text from the balloons | deterministic fill, on CPU |
| `render` | typesets the translation into the balloon | Pillow |

Two details explain a good deal of the code:

- **llama-server is started at the beginning of a stage and shut down at the
  end.** Killing the process is what frees the VRAM for the next stage, so
  there is no "unload the model" step anywhere.
- **Balloon cleaning does not use a generative model.** It fills the balloon
  area with the dominant background colour while preserving the outline. It is
  deterministic, reproducible and needs no VRAM. Inpainting-based backends were
  tried and removed: they regenerated the whole page and introduced visible
  drift.

Besides the stages there is an optional **pretreatment** (runs the raw pages
through ComfyUI before OCR, for upscaling or denoising) and an interactive
**Review** window to fix text, position and formatting of individual balloons
before the final render.

## Interfaces

The same functionality is available three ways:

```bash
python main.py --stage full --comic "Comic name"   # command line
python pipeline_gui_tk.py                          # desktop GUI (Tkinter)
python web_app.py                                  # local web app
```

---

## Requirements

- **Python 3.12+** (developed on 3.14)
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** with `llama-server` on
  your PATH. On Windows: `winget install ggml.llamacpp`
- A GPU helps a lot but is not required: `llama-server` decides on its own how
  many layers fit in VRAM.
- **ComfyUI** — only if you want the pretreatment step. Everything else works
  without it.

### Models to download

No model is bundled with this repository. You need GGUF files, for example:

| Role | Model used during development |
|---|---|
| OCR (vision) | `Qwen3-VL-8B-Instruct` (plus its `mmproj`) or `PaddleOCR-VL-1.6` |
| Translation | `translategemma-27b-it` (Q4_K_S quantisation) |

Other models work too: the paths are set in `config.yaml`, or from the Settings
tab.

---

## Installation

The project uses **three separate virtual environments**, because balloon
detection needs versions of torch that do not coexist with the rest.
`detect.py` invokes the other two interpreters as separate processes.

```bash
git clone https://github.com/<your-user>/Comics_Translate
cd Comics_Translate

# 1. main environment (pipeline, GUI, web app)
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. text detection — needs comic-text-detector
git clone https://github.com/dmMaze/comic-text-detector
python -m venv venv_ctd
venv_ctd/Scripts/python.exe -m pip install -r requirements-ctd.txt

# 3. balloon segmentation
python -m venv venv_bubbleseg
venv_bubbleseg/Scripts/python.exe -m pip install -r requirements-bubbleseg.txt
```

On Windows `setup_windows.ps1` runs the same steps.

You then need the **detection model weights**, not included in this repository:

**`comic-text-detector/data/comictextdetector.pt.onnx`** — text region
detection. Download it from the releases of the
[original repository](https://github.com/dmMaze/comic-text-detector).

**`bubble_seg/model.pt`** — balloon segmentation, a YOLOv8m-seg model trained
on the single class `speech bubble`:

```bash
# requires git-lfs
git clone https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble
cp yolov8m_seg-speech-bubble/model.pt bubble_seg/model.pt
```

Or download just the file from
[huggingface.co/kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
(`model.pt`, 54,809,749 bytes, SHA-256 `200141c4…50ed610`). It is distributed
under **GPL-3.0**: using it as a model, as this pipeline does, imposes nothing
on the code here, but keep it in mind if you redistribute it.

Any other ultralytics-format balloon segmentation model works: set the path in
`config.yaml`, under `bubble_seg`.

### Configuration

```bash
cp config.example.yaml config.yaml
```

Then open `config.yaml` and set the paths: working directories (`paths`) and
GGUF files (`llama_server`). The example file documents every section.

`config.yaml` is in `.gitignore`, so your local settings never reach the
repository.

---

## Usage

Put the pages in a subfolder of `input_dir`, one folder per comic:

```
input_pages/
  My comic - Chapter 1/
    001.jpg
    002.jpg
```

Then:

```bash
# whole comic, every stage
venv/Scripts/python.exe main.py --stage full --comic "My comic - Chapter 1"

# only some pages
venv/Scripts/python.exe main.py --stage ocr --comic "..." --start 0 --end 5

# check what is still missing before calling it done
venv/Scripts/python.exe main.py --stage audit --comic "..."
```

Finished pages land in `output_dir`, intermediate state in `work_dir` (one
`translated.json` per page, editable by hand).

### Polishing the result

Automatic rendering does not get everything right. The **Review** tab opens the
finished page with the balloons as editable boxes: move them, resize them, fix
the text, change font and alignment, then regenerate that single page. It
flags on its own the balloons where OCR read nothing and those where the text
was truncated.

The **Corrections** tab exports every line of a comic to a TXT file you can fix
in an external editor, and imports it back. Hand-made corrections go into the
translation cache, so re-running the stage does not overwrite them.

---

## Layout

```
main.py               stage orchestration
detect.py             balloon detection (invokes the two separate venvs)
ocr.py                transcription via the vision model
translate_local.py    translation, cache, glossary
clean.py              erasing the text from the balloon
render.py             typesetting and drawing the translated text
balloon_shape.py      balloon geometry (fused balloons, tails, ...)
fit_check.py          shorter rewrite when the text does not fit
audit.py              report of what is still to fix
i18n.py               interface localisation
pipeline_gui_tk.py    desktop GUI
web_app.py            local web app
flux_pretreat.py      optional ComfyUI pretreatment
```

---

## Third-party software

| Component | Licence | How it is used |
|---|---|---|
| [comic-text-detector](https://github.com/dmMaze/comic-text-detector) | GPL-3.0 | cloned separately, run as a separate process in its own venv |
| [ultralytics](https://github.com/ultralytics/ultralytics) (YOLO) | AGPL-3.0 | installed in a separate venv, run as a separate process |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | installed separately, used as an HTTP server |
| ComfyUI | GPL-3.0 | optional, used as an HTTP server for the pretreatment only |
| [yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble) | GPL-3.0 | segmentation model weights, downloaded separately |

None of these is included in this repository, and none is imported by this
project's code: they are separate programs, invoked over subprocess or HTTP. If
you integrate them differently, check the obligations of their licences.

The fonts in `fonts/` have their own licences: check them before redistributing.

## A note on use

This is a tool for translating comics locally, meant for material you have the
right to translate — public domain works, your own work, or material you have
permission for. It does not encourage or facilitate redistributing copyrighted
works.

## Licence

See [LICENSE](LICENSE).
