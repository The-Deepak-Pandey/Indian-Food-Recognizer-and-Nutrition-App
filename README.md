# Indian Food Recognizer + DP Naturals Web App

End-to-end Indian-dish recognition pipeline with a local Flask UI. Trained DINOv2-L (LoRA) classifier over 80 Indian dishes, multi-task auxiliary heads (diet/course), nutrition + allergen-badge lookup, and a simple upload-and-predict web app.

---

## Folder structure

Below is the folder structure specifically for the final working model - few files are not in the repository, due to limitation issues, but this is how the final architecture and implementation of it with a Flask app looks like -

```
home/
├── README.md                          ← this file
├── dp_final.ipynb                     ← end-to-end training/eval notebook
│
├── data/
│   ├── raw/                           ← Kaggle dataset (input)
│   │   ├── indian_food.csv            ← 80-class metadata (diet, course, region, ingredients)
│   │   └── indian-food-images/        ← ~4k images, 80 sub-folders (one per dish)
│   ├── processed/                     ← derived from raw/ by notebook cells 4–5
│   │   ├── splits.json                ← train/val/test split (80/10/10, seed 42)
│   │   └── class_to_aux.json          ← per-class diet/course/ingredients/region
│   └── nutrition/
│       └── nutrition_db.json          ← 80-dish DB: kcal/carb/protein/fat ranges
│                                        + peanuts/dairy/gluten allergen badges
│                                        (red / yellow / green codes)
│
├── models/
│   ├── dinov2_large_best.pt           ← trained checkpoint (DINOv2-L + LoRA r=16,
│   │                                    multi-task head; 86.25% val top-1, ~1.2 GB)
│   └── classes.json                   ← ordered list of 80 class names
│
├── results/                           ← evaluation artifacts
│   ├── confusion_matrix.npy           ← 80×80 raw counts
│   ├── confusion_matrix.png           ← rendered heatmap
│   ├── history.json                   ← train_loss / val_top1 per epoch
│   └── summary.json                   ← test top-1/5, ECE, temperature, params
│
├── hf_cache/                          ← HuggingFace cache (auto-created)
│
└── dp_naturals/                       ← local web app
    ├── app.py                         ← Flask server (routes: /, /predict, /uploads,
    │                                    /history, /health)
    ├── model_api.py                   ← inference layer (FoodModel + DishRecognizer)
    ├── requirements.txt               ← flask, torch, torchvision, timm, transformers,
    │                                    peft, Pillow
    ├── history.json                   ← rolling log of last 20 predictions
    ├── templates/
    │   └── index.html                 ← upload UI
    ├── static/
    │   ├── style.css                  ← styling (allergen color codes, layout)
    │   └── script.js                  ← drag-drop upload + result rendering
    └── uploads/                       ← user-uploaded images (auto-created)
```

> **Note:** A small leftover `data/data/` folder may exist from an earlier path-config bug — it is empty and safe to delete.

---

## Pipeline overview

### 1. Notebook (`dp_final.ipynb`)
Run cells top-to-bottom. The training cell **auto-skips** if `models/dinov2_large_best.pt` already exists, so by default it loads the pre-trained checkpoint and:
1. Evaluates on the test set (top-1 / top-5 / per-class report).
2. Calibrates predictions via temperature scaling.
3. **Builds `data/nutrition/nutrition_db.json`** — the canonical nutrition + allergen DB used by the web app.
4. Runs a single-image inference demo.

To force re-training, set `SKIP_TRAIN_IF_CKPT = False` in the train cell.

### 2. Web app (`dp_naturals/`)
Local-only Flask UI for upload-and-predict. The browser shows the dish prediction, confidence, nutrition cards, and color-coded allergen badges (peanuts / dairy / gluten as red / yellow / green).

**Run:**
```bash
cd dp_naturals
pip install -r requirements.txt
DP_DEVICE=cpu python3 app.py     # CPU is safer on small/shared GPUs
# open http://127.0.0.1:5000
```

**Env-var overrides** (optional):
| Var | Default | Purpose |
|---|---|---|
| `DP_MODEL_CKPT` | `../models/dinov2_large_best.pt` | path to checkpoint |
| `DP_NUTRITION_DB` | `../data/nutrition/nutrition_db.json` | path to nutrition DB |
| `DP_DEVICE` | auto (CPU if GPU has < 3 GB free) | `cpu`, `cuda:0`, etc. |
| `DP_T_VAL` | `1.0` | softmax temperature |
| `DP_OOD_THRESHOLD` | `0.30` | MSP cutoff for "not Indian food" message |

---

## Model summary

| Field | Value |
|---|---|
| Backbone | DINOv2-L (ViT-L/14, frozen base + LoRA r=16) |
| Input size | 518 × 518 |
| Classes | 80 |
| Val top-1 | 86.25 % |
| Trainable params | ~5 M (out of ~300 M total) |
| Heads | dish (80) · diet (3) · course (4) |

## Allergen badges (in `nutrition_db.json`)

Per dish, three tracked allergens with one of:
- **red** — contains allergen (high risk)
- **yellow** — may contain / uncertain
- **green** — safe / not present

```json
"rabri": {
  "allergens": { "peanuts": "yellow", "dairy": "red", "gluten": "green" }
}
```

---

## Quick start

1. Run the notebook → confirms model loads, regenerates `nutrition_db.json`.
2. Launch the web app → upload a dish image → see prediction + nutrition + allergen badges.
