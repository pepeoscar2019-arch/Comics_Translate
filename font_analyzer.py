"""
Analisi euristica dei font nei balloon dei fumetti.
Rileva caratteristiche visive del testo originale e suggerisce il font più simile
 dalla libreria disponibile.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from typing import Optional, Tuple, List
import json

log = logging.getLogger("pipeline.font_analyzer")


# ============================================================
# CARATTERISTICHE FONT ANALIZZABILI
# ============================================================

class FontFeatures:
    """Caratteristiche euristiche rilevate dal testo originale."""

    def __init__(self):
        self.is_bold: bool = False          # Testo in grassetto
        self.is_italic: bool = False        # Testo in corsivo
        self.is_all_caps: bool = False      # Tutto maiuscolo
        self.is_handwritten: bool = False   # Stile manoscritto
        self.avg_stroke_width: float = 1.0  # Spessore medio del tratto
        self.text_color: Tuple[int, int, int] = (0, 0, 0)  # Colore del testo
        self.bg_color: Tuple[int, int, int] = (255, 255, 255)  # Colore sfondo
        self.estimated_size: int = 20       # Dimensione stimata in px

    def to_dict(self) -> dict:
        return {
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "is_all_caps": self.is_all_caps,
            "is_handwritten": self.is_handwritten,
            "avg_stroke_width": self.avg_stroke_width,
            "text_color": list(self.text_color),
            "bg_color": list(self.bg_color),
            "estimated_size": self.estimated_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FontFeatures":
        f = cls()
        f.is_bold = d.get("is_bold", False)
        f.is_italic = d.get("is_italic", False)
        f.is_all_caps = d.get("is_all_caps", False)
        f.is_handwritten = d.get("is_handwritten", False)
        f.avg_stroke_width = d.get("avg_stroke_width", 1.0)
        f.text_color = tuple(d.get("text_color", [0, 0, 0]))
        f.bg_color = tuple(d.get("bg_color", [255, 255, 255]))
        f.estimated_size = d.get("estimated_size", 20)
        return f


# ============================================================
# ANALISI IMMAGINE
# ============================================================

def _crop_balloon(image_bgr: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Estrae il crop del balloon dall'immagine."""
    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return image_bgr[y1:y2, x1:x2]


def _detect_text_color(crop: np.ndarray) -> Tuple[int, int, int]:
    """
    Rileva il colore dominante del testo nel balloon.
    Usa k-means clustering per separare testo e sfondo.
    """
    # Converti in RGB e reshape
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    # K-means con 2 cluster (testo e sfondo)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Determina quale cluster è il testo (quello con meno pixel, tipicamente)
    counts = np.bincount(labels.flatten())
    text_cluster = 0 if counts[0] < counts[1] else 1

    color = tuple(centers[text_cluster].astype(int))
    return color


def _detect_bg_color(crop: np.ndarray) -> Tuple[int, int, int]:
    """Rileva il colore dominante dello sfondo."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    bg_cluster = 0 if counts[0] > counts[1] else 1

    color = tuple(centers[bg_cluster].astype(int))
    return color


def _estimate_stroke_width(crop: np.ndarray, text_color: Tuple[int, int, int]) -> float:
    """
    Stima lo spessore del tratto del font analizzando i bordi del testo.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Crea una maschera del testo (pixel simili al colore del testo)
    # Convertiamo text_color in BGR per confronto
    bgr_color = (text_color[2], text_color[1], text_color[0])

    # Maschera basata sulla distanza di colore
    diff = np.abs(crop.astype(np.int16) - np.array(bgr_color, dtype=np.int16))
    color_distance = np.sum(diff, axis=2)

    # Soglia: pixel entro una certa distanza dal colore del testo
    mask = (color_distance < 150).astype(np.uint8) * 255

    # Erosione e dilatazione per pulire
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Distance transform per stimare lo spessore
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Lo spessore è circa il doppio della distanza massima dal bordo
    max_dist = np.max(dist) if np.max(dist) > 0 else 1
    stroke_width = max_dist * 2

    return min(stroke_width, 20.0)  # Limita a valori ragionevoli


def _detect_is_bold(crop: np.ndarray, stroke_width: float) -> bool:
    """
    Determina se il font è bold basandosi sullo spessore del tratto.
    """
    # Se lo spessore medio è > 3.5px, consideriamo bold
    return stroke_width > 3.5


def _detect_is_all_caps(text: str) -> bool:
    """Verifica se il testo è tutto in maiuscolo."""
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def _detect_is_handwritten(crop: np.ndarray) -> bool:
    """
    Euristico: i font manoscritti hanno bordi più irregolari.
    Analizza la varianza dei bordi.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Calcola la varianza della direzione dei bordi (irregolarità)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Alta varianza = bordi irregolari = possibile manoscritto
    edge_pixels = magnitude[edges > 0]
    if len(edge_pixels) < 10:
        return False

    # Normalizza per dimensione
    irregularity = np.std(edge_pixels) / (np.mean(edge_pixels) + 1e-6)

    return irregularity > 0.8


def _estimate_font_size(crop: np.ndarray, text: str) -> int:
    """
    Stima la dimensione del font in pixel basandosi sull'altezza dei caratteri.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Threshold per isolare il testo
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Trova i contorni dei caratteri
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 20

    # Altezza media dei contorni (caratteri)
    heights = [cv2.boundingRect(cnt)[3] for cnt in contours if cv2.boundingRect(cnt)[3] > 5]
    if not heights:
        return 20

    avg_height = np.mean(heights)
    return int(avg_height)


# ============================================================
# FUNZIONE PRINCIPALE DI ANALISI
# ============================================================

def analyze_font(image_bgr: np.ndarray, bbox: List[int], text: str) -> FontFeatures:
    """
    Analizza il font originale nel balloon e restituisce le caratteristiche rilevate.

    Args:
        image_bgr: Immagine della pagina in formato BGR (OpenCV)
        bbox: Bounding box [x1, y1, x2, y2] del balloon
        text: Testo estratto dal balloon (per analisi maiuscole/minuscole)

    Returns:
        FontFeatures con le caratteristiche rilevate
    """
    crop = _crop_balloon(image_bgr, bbox)
    if crop.size == 0:
        log.warning(f"Crop vuoto per bbox {bbox}")
        return FontFeatures()

    features = FontFeatures()

    # Rileva colori
    features.text_color = _detect_text_color(crop)
    features.bg_color = _detect_bg_color(crop)

    # Stima spessore e bold
    features.avg_stroke_width = _estimate_stroke_width(crop, features.text_color)
    features.is_bold = _detect_is_bold(crop, features.avg_stroke_width)

    # Analisi testo
    features.is_all_caps = _detect_is_all_caps(text)

    # Euristico manoscritto
    features.is_handwritten = _detect_is_handwritten(crop)

    # Dimensione stimata
    features.estimated_size = _estimate_font_size(crop, text)

    log.info(
        f"  Font analizzato: bold={features.is_bold}, caps={features.is_all_caps}, "
        f"handwritten={features.is_handwritten}, stroke={features.avg_stroke_width:.1f}px, "
        f"size={features.estimated_size}px, color={features.text_color}"
    )

    return features


# ============================================================
# MATCHING FONT
# ============================================================

# Font di fumetto/manga noti, il cui filename non segue le convenzioni
# generiche (bold/italic/hand/script/...) intercettate dall'euristica
# sottostante. Chiave = stem del file (lowercase, senza estensione).
# "avoid": penalizza il font nel matching automatico anche quando le
# caratteristiche coincidono (es. Comic Sans: i lettori di fumetti/manga
# lo riconoscono e lo detestano, va usato solo se non c'e' alternativa).
# "preferred": font standard preferito dall'utente per il dialogo non
# manoscritto: a parita' di punteggio con altri candidati vince questo.
_KNOWN_COMIC_FONTS = {
    "comic": {"is_handwritten": False, "avoid": True},
    "comicbd": {"is_handwritten": False, "is_bold": True, "avoid": True},
    "comici": {"is_handwritten": False, "is_italic": True, "avoid": True},
    "comicz": {"is_handwritten": False, "is_bold": True, "is_italic": True, "avoid": True},
    "comic_sans_ms": {"is_handwritten": False, "avoid": True},
    "animeace2_reg": {"is_handwritten": False, "preferred": True},
    "animeace2_bld": {"is_handwritten": False, "is_bold": True, "preferred": True},
    "animeace2_ital": {"is_handwritten": False, "is_italic": True, "preferred": True},
    "kalam-regular": {"is_handwritten": True},
    "kalam-bold": {"is_handwritten": True, "is_bold": True},
    "itim-regular": {"is_handwritten": True},
    "fuzzybubbles-regular": {"is_handwritten": True},
    "fuzzybubbles-bold": {"is_handwritten": True, "is_bold": True},
    "yatraone-regular": {"is_handwritten": True},
    "bangers-regular": {"is_handwritten": True, "is_bold": True},
    "rubikglitch-regular": {"is_handwritten": True},
}


class FontLibrary:
    """
    Libreria di font disponibili con i loro metadati.
    Scansiona una cartella di font TTF/OTF estraendo le caratteristiche.
    """

    def __init__(self, fonts_dir: str = "fonts"):
        self.fonts_dir = Path(fonts_dir)
        self.available_fonts: List[dict] = []
        self._scan_fonts()

    def _scan_fonts(self):
        """Scansiona la cartella fonts e cataloga i font disponibili."""
        if not self.fonts_dir.exists():
            log.warning(f"Cartella font non trovata: {self.fonts_dir}")
            return

        for font_file in self.fonts_dir.glob("*.ttf"):
            self._analyze_font_file(font_file)
        for font_file in self.fonts_dir.glob("*.otf"):
            self._analyze_font_file(font_file)

        log.info(f"Libreria font: {len(self.available_fonts)} font trovati")

    def _analyze_font_file(self, font_path: Path):
        """Analizza un file font per estrarne le caratteristiche dal nome."""
        name = font_path.stem.lower()

        # Estrai informazioni dal nome del file
        is_bold = "bold" in name or "heavy" in name or "black" in name
        is_italic = "italic" in name or "oblique" in name
        is_handwritten = any(w in name for w in ["comic", "hand", "script", "cursive", "brush", "marker"])
        is_sans = "sans" in name or "arial" in name or "helvetica" in name or "verdana" in name
        is_serif = "serif" in name or "times" in name or "georgia" in name or "garamond" in name

        font_info = {
            "path": str(font_path),
            "name": font_path.stem,
            "is_bold": is_bold,
            "is_italic": is_italic,
            "is_handwritten": is_handwritten,
            "is_sans": is_sans,
            "is_serif": is_serif,
            "avoid": False,
            "preferred": False,
        }
        font_info.update(_KNOWN_COMIC_FONTS.get(name, {}))
        self.available_fonts.append(font_info)

    def find_best_match(self, features: FontFeatures) -> str:
        """
        Trova il font più simile alle caratteristiche richieste.

        Returns:
            Percorso del font migliore, o None se non trovato.
        """
        if not self.available_fonts:
            log.warning("Nessun font disponibile nella libreria")
            return None

        best_score = -1
        best_font = None

        for font_info in self.available_fonts:
            score = 0

            # Match bold
            if features.is_bold == font_info["is_bold"]:
                score += 3

            # Match handwritten
            if features.is_handwritten == font_info["is_handwritten"]:
                score += 4  # Peso alto perché lo stile è molto distintivo

            # Match all-caps (preferisci font che supportano bene le maiuscole)
            if features.is_all_caps and not font_info["is_handwritten"]:
                score += 1

            # Penalità per mismatch
            if features.is_bold and not font_info["is_bold"]:
                score -= 2
            if features.is_handwritten and not font_info["is_handwritten"]:
                score -= 3

            # Font da evitare (es. Comic Sans): scelto solo se non c'e'
            # nessuna alternativa migliore in libreria.
            if font_info.get("avoid"):
                score -= 10

            # Font standard preferito (Anime Ace): a parita' di punteggio
            # con altri candidati per il dialogo non manoscritto, vince lui.
            if font_info.get("preferred") and not features.is_handwritten:
                score += 2

            log.debug(f"  Font '{font_info['name']}': score={score}")

            if score > best_score:
                best_score = score
                best_font = font_info

        if best_font:
            log.info(f"Font selezionato: {best_font['name']} (score={best_score})")
            return best_font["path"]

        # Fallback: primo font disponibile
        fallback = self.available_fonts[0]["path"]
        log.warning(f"Nessun match trovato, uso fallback: {fallback}")
        return fallback


# ============================================================
# FUNZIONE DI UTILITÀ
# ============================================================

def get_font_for_balloon(
    image_bgr: np.ndarray,
    bbox: List[int],
    text: str,
    fonts_dir: str = "fonts",
    default_font: str = None
) -> Tuple[str, FontFeatures]:
    """
    Funzione principale: analizza il font originale e restituisce il percorso
    del font più simile dalla libreria.

    Args:
        image_bgr: Immagine pagina BGR
        bbox: Bounding box del balloon
        text: Testo del balloon
        fonts_dir: Cartella con i font TTF/OTF
        default_font: Font di fallback se la libreria è vuota

    Returns:
        (percorso_font, features)
    """
    features = analyze_font(image_bgr, bbox, text)

    library = FontLibrary(fonts_dir)
    font_path = library.find_best_match(features)

    if font_path is None and default_font:
        font_path = default_font
        log.info(f"Uso font di default: {default_font}")

    return font_path, features
