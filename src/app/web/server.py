"""
Urban Green Score — serveur web.

FastAPI sert la page (static/index.html) et expose une API unique :
    POST /api/predict   (multipart: file=<image>, mode=cloud|local)

- mode=cloud : l'image est transmise à l'API Gateway (-> SageMaker).
               Le serveur fait l'appel, donc pas de souci de CORS côté navigateur.
- mode=local : inférence directe avec le UNet local (torch chargé à la demande).

Lancement (depuis la racine du repo) :
    uvicorn server:app --app-dir app/web --reload
ou :
    python app/web/server.py
"""

import base64
import io
import os
import sys
from functools import lru_cache

import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

# --- Imports du projet (même logique que le Streamlit : ce fichier est à 2 niveaux de la racine)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)

from config import NUM_CLASSES, IGNORE_INDEX  # noqa: E402
from preprocessing.utils import resize_image  # noqa: E402

# --- Configuration (surchargeable par variables d'environnement)
API_GATEWAY_URL = os.getenv(
    "API_GATEWAY_URL",
    "https://5jhgyaenq3.execute-api.eu-west-3.amazonaws.com/predict",
)
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    os.path.join(ROOT_DIR, "artifacts", "models", "best_model.pth"),
)
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))

CLASSES = [
    {"id": 0, "name": "no-data", "color": [0, 0, 0]},
    {"id": 1, "name": "background", "color": [200, 200, 200]},
    {"id": 2, "name": "building", "color": [128, 128, 128]},
    {"id": 3, "name": "road", "color": [255, 255, 0]},
    {"id": 4, "name": "water", "color": [0, 0, 255]},
    {"id": 5, "name": "barren", "color": [139, 69, 19]},
    {"id": 6, "name": "forest", "color": [0, 255, 0]},
    {"id": 7, "name": "agriculture", "color": [0, 128, 0]},
]
PALETTE = np.array([c["color"] for c in CLASSES], dtype=np.uint8)

app = FastAPI(title="Urban Green Score")


# ---------------------------------------------------------------- helpers
def colorize_mask(mask_np: np.ndarray) -> Image.Image:
    idx = np.clip(mask_np.astype(np.int64), 0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[idx])


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_jsonable(obj):
    """Convertit tensors / numpy en types Python sérialisables."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):  # torch.Tensor, np.ndarray, np.generic
        return obj.tolist()
    return obj


# ---------------------------------------------------------------- inférence locale
@lru_cache(maxsize=1)
def load_local_model(model_path: str):
    import torch
    from training.model import UNet

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def predict_local(image_bytes: bytes):
    import torch
    from scoring.green_score import compute_green_score

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = resize_image(image)

    x = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)

    model, device = load_local_model(LOCAL_MODEL_PATH)
    with torch.no_grad():
        output = model(x.to(device))
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()

    green_score, proportions = compute_green_score(pred_mask, ignore_index=IGNORE_INDEX)
    mask_np = pred_mask.numpy().astype(np.uint8)
    return image, mask_np, green_score, proportions


# ---------------------------------------------------------------- inférence cloud
def predict_cloud(image_bytes: bytes):
    # On normalise en PNG puisque l'API attend du image/png
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    response = requests.post(
        API_GATEWAY_URL,
        headers={"Content-Type": "image/png"},
        data=buf.getvalue(),
        timeout=API_TIMEOUT,
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"API Gateway a répondu {response.status_code} : {response.text[:300]}",
        )

    result = response.json()
    mask_np = np.array(
        Image.open(io.BytesIO(base64.b64decode(result["mask_png_base64"])))
    ).astype(np.uint8)
    image = image.resize((mask_np.shape[1], mask_np.shape[0]))
    return image, mask_np, result["green_score"], result["class_proportions"]


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/config")
def config():
    return {
        "api_url": API_GATEWAY_URL,
        "model_path": LOCAL_MODEL_PATH,
        "local_model_available": os.path.isfile(LOCAL_MODEL_PATH),
        "classes": CLASSES,
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...), mode: str = Form("cloud")):
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    try:
        if mode == "local":
            image, mask_np, green_score, proportions = predict_local(image_bytes)
        elif mode == "cloud":
            image, mask_np, green_score, proportions = predict_cloud(image_bytes)
        else:
            raise HTTPException(status_code=400, detail=f"Mode inconnu : {mode}")
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"API Gateway injoignable : {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__} : {e}")

    return {
        "mode": mode,
        "green_score": float(to_jsonable(green_score)),
        "class_proportions": to_jsonable(proportions),
        "mask_shape": list(mask_np.shape),
        "image_png_base64": png_b64(image),
        "mask_png_base64": png_b64(colorize_mask(mask_np)),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))