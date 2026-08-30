import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_model_cache = None


def _get_model(model_path: str) -> YOLO:
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(model_path)
    return _model_cache


def run(image_path: Path, output_json_path: Path, masks_dir: Path, model_path: str, conf: float = 0.25):
    """
    Segmenta i veri speech bubble (forma completa, non solo il testo) con
    kitsumed/yolov8m_seg-speech-bubble. A differenza di comic-text-detector,
    la maschera segue il contorno reale del balloon (bordo+interno), utile
    per i casi in cui il bbox del testo e' molto piu' piccolo del balloon
    che lo contiene (es. dialoghi brevi in balloon grandi).
    """
    model = _get_model(model_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere immagine: {image_path}")
    h, w = img.shape[:2]

    masks_dir.mkdir(parents=True, exist_ok=True)

    # retina_masks=True e' indispensabile: senza, r.masks.data resta nello
    # spazio LETTERBOX dell'input del modello (es. 640x512 per una pagina
    # 880x1168, con bande di padding ai lati per arrivare a un multiplo di
    # 32). Ridimensionare quella maschera direttamente a (w, h) allunga
    # anche le bande di padding, spostando e comprimendo la sagoma rispetto
    # al balloon vero: su una pagina verticale lo scarto arriva a ~20-25 px
    # in orizzontale (maschera spostata verso destra a sinistra della
    # pagina e viceversa). La pulizia riempiva quindi in parte il disegno
    # accanto al balloon lasciando scoperte le lettere sul lato opposto,
    # che riaffioravano sotto la traduzione. Con retina_masks le maschere
    # arrivano gia' alla risoluzione dell'immagine originale.
    results = model.predict(str(image_path), device="cpu", conf=conf, verbose=False, retina_masks=True)
    r = results[0]

    detections = []
    if r.masks is not None and r.boxes is not None:
        masks_data = r.masks.data.cpu().numpy()
        for i, (mask, box, confidence) in enumerate(zip(masks_data, r.boxes.xyxy, r.boxes.conf)):
            # Normalmente gia' a piena risoluzione grazie a retina_masks; il
            # resize resta come rete di sicurezza se una versione di
            # ultralytics dovesse restituire comunque un'altra dimensione.
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask > 0.5).astype(np.uint8) * 255

            ys, xs = np.where(mask_bin > 0)
            if len(xs) == 0:
                continue
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

            mask_path = masks_dir / f"bubble_{i:03d}.png"
            cv2.imwrite(str(mask_path), mask_bin)

            detections.append({
                "bubble_id": i,
                "bbox": bbox,
                "confidence": float(confidence),
                "mask_path": str(mask_path),
            })

    output = {
        "page_path": str(image_path),
        "page_id": image_path.stem,
        "image_size": [w, h],
        "bubbles": detections,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(detections)} balloon (forma reale) rilevati -> {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--model", default=str(Path(__file__).parent / "model.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    run(Path(args.image), Path(args.output_json), Path(args.masks_dir), args.model, args.conf)
