# Skin-Scan OSS Starter

Production-grade skin analysis API that generates 7 category maps from facial images: redness, oiliness, texture, pores, blemishes, hydration, and pigmentation.

## Quickstart

### Local

```bash
# Python 3.11+
uv venv && uv pip install -e .
cp .env.example .env

# Run API
uv run uvicorn src.app.main:app --reload --port 8000

# One-off CLI scan
uv run python -m src.cli.scan_image --input path/to/face.jpg --out out_dir/
```

### Docker

```bash
docker build -t skin-scan .
docker run -p 8000:8000 skin-scan
```

### Make targets

```bash
make setup      # Create venv, install dependencies
make dev        # Run API with reload
make test       # Run pytest
make scan SCAN=path/to/img.jpg  # CLI scan
make lint       # Run ruff + pyright
```

## API Endpoints

### POST /scan

Upload an image and receive skin analysis results.

**Request:** multipart/form-data with `image` field

**Response:**
```json
{
  "scores": {
    "redness": 0.81,
    "oiliness": 0.74,
    "texture": 0.62,
    "pores": 0.48,
    "blemishes": 0.36,
    "hydration": 0.29,
    "pigment": 0.97
  },
  "overlays": {
    "redness": "data:image/png;base64,...",
    "oiliness": "...",
    "texture": "...",
    "pores": "...",
    "blemishes": "...",
    "hydration": "...",
    "pigment": "..."
  },
  "regions": ["forehead", "nose", "cheeks", "chin"]
}
```

### GET /health

Health check endpoint.

## Pipeline Overview

```
input image
→ preprocess (color constancy, white balance, face crop, gamma)
→ landmarks (MediaPipe FaceMesh - 468 points)
→ region masks (forehead, cheeks, nose, chin)
→ per-map analysis:
   • redness: CIE LAB a* channel
   • oiliness: specular highlight detection
   • texture: local binary patterns / variance
   • pores: blob detection on high-pass filter
   • blemishes: rule-based + optional CNN
   • hydration: texture-based proxy
   • pigment: normalized brown mask
→ normalize to [0,1], compute region-weighted scores
→ overlay RGBA heatmaps
→ JSON + base64 overlays out
```

## Map Algorithms

### Redness
Convert to CIE LAB, use a* channel. Z-score within face mask, clamp to [0,1].

### Oiliness
Detect specular highlights via HSV thresholding (high V, low S) with gradient confirmation.

### Texture
Local Binary Patterns or Laplacian variance over patches. High variance = roughness.

### Pores
High-pass filter (DoG) + Laplacian of Gaussian blob detector (radius 2-5px).

### Blemishes
Rule-based: oiliness + pore blobs + brightness delta. Optional CNN classifier.

### Hydration
Proxy via texture low-frequency dryness + reduced specular response.

### Pigment
Brown mask via RGB thresholds and LAB b* after shading correction.

## Structure

```
skin-scan/
├─ src/
│  ├─ app/              # FastAPI application
│  ├─ pipeline/         # Image analysis pipeline
│  │  ├─ maps/          # Individual map algorithms
│  │  ├─ face_mesh.py   # MediaPipe face detection
│  │  ├─ preprocess.py  # Image preprocessing
│  │  ├─ compose.py     # Pipeline orchestration
│  │  └─ visualize.py   # Heatmap generation
│  ├─ ml/               # Optional ML models
│  └─ cli/              # Command-line interface
├─ tests/
└─ web/                 # Demo frontend
```

## Disclaimer

This is a cosmetic analysis tool for educational and research purposes. **This is not medical software.** Do not use for diagnostic purposes. Always consult qualified healthcare professionals for medical advice.

## License

MIT
