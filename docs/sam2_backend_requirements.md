# SAM2 Backend Server Requirements Document

## 1. Background

### 1.1 What is SAM2?

**Segment Anything Model 2 (SAM2)** is Meta's foundation model for image and video segmentation. It can segment any object in an image given:
- **Point prompts**: Click locations indicating foreground (include) or background (exclude)
- **Box prompts**: Bounding boxes around objects
- **Mask prompts**: Previous masks for refinement

SAM2 improves on SAM1 with better accuracy, video support, and memory efficiency.

**Key Resources:**
- [SAM2 GitHub Repository](https://github.com/facebookresearch/segment-anything-2)
- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
- [Model Weights Download](https://github.com/facebookresearch/segment-anything-2#model-checkpoints)

### 1.2 Our Use Case

SuperSplat is a Gaussian Splat editor. Users need to select regions of 3D Gaussian splats for editing (delete, hide, transform). Currently, users draw shapes manually. SAM2 will enable "smart selection" - click a few points and AI identifies the object boundaries.

**Workflow:**
1. User clicks points on a 2D render of the Gaussian splat scene
2. Frontend captures the rendered image + point coordinates
3. Backend runs SAM2 to generate a segmentation mask
4. Frontend maps the 2D mask back to 3D Gaussian splats

### 1.3 Architecture Overview

```
┌─────────────────┐     HTTP/REST      ┌─────────────────┐
│   SuperSplat    │ ◄──────────────────► │   SAM2 Server   │
│   (Browser)     │    JSON + Binary    │   (Python)      │
└─────────────────┘                     └─────────────────┘
                                               │
                                               ▼
                                        ┌─────────────────┐
                                        │  SAM2 Model     │
                                        │  (GPU/CPU)      │
                                        └─────────────────┘
```

---

## 2. API Specification

### 2.1 Health Check Endpoint

**Purpose:** Verify server is running and model is loaded.

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model": "sam2_hiera_large",
  "device": "cuda",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK` - Server healthy
- `503 Service Unavailable` - Model not loaded

---

### 2.2 Segmentation Endpoint (Primary)

**Purpose:** Generate segmentation mask from image + point prompts.

```
POST /segment
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "<base64-encoded PNG or JPEG>",
  "width": 1920,
  "height": 1080,
  "points": [
    { "x": 450, "y": 300, "type": "fg" },
    { "x": 800, "y": 600, "type": "fg" },
    { "x": 100, "y": 100, "type": "bg" }
  ],
  "options": {
    "threshold": 0.5,
    "return_logits": false,
    "multimask_output": false
  }
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Base64-encoded image (PNG/JPEG). RGB or RGBA. |
| `width` | number | Yes | Image width in pixels |
| `height` | number | Yes | Image height in pixels |
| `points` | array | Yes | Array of point prompts (at least 1 required) |
| `points[].x` | number | Yes | X coordinate (0 = left edge) |
| `points[].y` | number | Yes | Y coordinate (0 = top edge) |
| `points[].type` | string | Yes | `"fg"` (foreground/include) or `"bg"` (background/exclude) |
| `options.threshold` | number | No | Mask threshold 0-1. Default: 0.5 |
| `options.return_logits` | boolean | No | Return soft logits. Default: false |
| `options.multimask_output` | boolean | No | Return multiple masks. Default: false |

**Response (Success):**
```json
{
  "width": 1920,
  "height": 1080,
  "mask": "<base64-encoded binary mask>",
  "logits": "<base64-encoded float32 array>",
  "confidence": 0.92
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `width` | number | Mask width in pixels |
| `height` | number | Mask height in pixels |
| `mask` | string | Base64-encoded binary mask (H×W bytes, 0=bg, 255=fg) |
| `logits` | string | Optional: Base64-encoded Float32Array (H×W×4 bytes) |
| `confidence` | number | Optional: Model confidence score 0-1 |

**Mask Format Details:**
- Binary mask: 1 byte per pixel, values 0 (background) or 255 (foreground)
- Total bytes = width × height
- Row-major order (first row, then second row, etc.)

**Error Response:**
```json
{
  "error": "INVALID_REQUEST",
  "message": "At least one point is required",
  "details": { "received_points": 0 }
}
```

**Error Codes:**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request or missing fields |
| `INVALID_IMAGE` | 400 | Cannot decode image |
| `INVALID_POINTS` | 400 | Points outside image bounds |
| `MODEL_ERROR` | 500 | SAM2 inference failed |
| `TIMEOUT` | 504 | Request took too long |

---

### 2.3 Alternative: Binary Endpoint (Optional, Higher Performance)

For lower latency, support raw binary image upload:

```
POST /segment/binary
Content-Type: application/octet-stream
X-Image-Width: 1920
X-Image-Height: 1080
X-Points: [{"x":450,"y":300,"type":"fg"}]
```

**Body:** Raw RGBA bytes (width × height × 4)

**Response:** Same as `/segment`

---

## 3. Technical Requirements

### 3.1 Model Requirements

| Requirement | Specification |
|-------------|---------------|
| Model | SAM2 (sam2_hiera_large recommended, sam2_hiera_small for lower resources) |
| Framework | PyTorch ≥ 2.0 |
| CUDA | 11.8+ (for GPU acceleration) |
| Python | 3.10+ |

**Available Model Variants:**

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| sam2_hiera_tiny | ~40MB | Fastest | Good |
| sam2_hiera_small | ~80MB | Fast | Better |
| sam2_hiera_base_plus | ~160MB | Medium | Great |
| sam2_hiera_large | ~320MB | Slower | Best |

**Model Loading Example:**
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
```

### 3.2 Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| Cold start | < 30s | Model loading time |
| Inference latency | < 2s | 1080p image, 5 points |
| Throughput | 1 req/s | Single GPU |
| Max image size | 4096×4096 | Larger images should be downscaled |
| Max points | 20 | More points ≠ better results |

### 3.3 Hardware Requirements

**Minimum (CPU-only):**
- CPU: 4+ cores
- RAM: 16GB
- Inference time: 5-15 seconds

**Recommended (GPU):**
- GPU: NVIDIA RTX 3060 or better (8GB+ VRAM)
- CPU: 4+ cores
- RAM: 16GB
- Inference time: 0.5-2 seconds

**Production:**
- GPU: NVIDIA A10/A100 or RTX 4090
- CPU: 8+ cores
- RAM: 32GB
- Inference time: < 0.5 seconds

### 3.4 Infrastructure Options

**Option A: Local Server (Development)**
```bash
# Start server on localhost:8000
python sam2_server.py --port 8000 --model sam2_hiera_large
```

**Option B: Docker Container**
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-runtime

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn pillow numpy
RUN pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Copy model weights and server code
COPY checkpoints/ /app/checkpoints/
COPY sam2_server.py /app/

EXPOSE 8000

CMD ["uvicorn", "sam2_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Option C: Cloud Deployment**
- AWS: EC2 with GPU (g4dn.xlarge ~$0.50/hr) + API Gateway
- GCP: Cloud Run with GPU or Compute Engine
- Azure: Container Instances with GPU

### 3.5 CORS Configuration (Critical)

The server **must** support CORS for browser requests:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific origins like ["https://supersplat.io"]
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-Image-Width", "X-Image-Height", "X-Points"],
)
```

---

## 4. Implementation Guide

### 4.1 Complete Python Server (FastAPI)

```python
"""
SAM2 Segmentation Server for SuperSplat
"""
import base64
import io
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configuration
MODEL_CONFIG = "sam2_hiera_l.yaml"
MODEL_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="SAM2 Segmentation Server", version="1.0.0")

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (loaded at startup)
predictor: SAM2ImagePredictor | None = None


class Point(BaseModel):
    x: int = Field(..., ge=0, description="X coordinate in pixels")
    y: int = Field(..., ge=0, description="Y coordinate in pixels")
    type: Literal["fg", "bg"] = Field(..., description="Point type: fg=foreground, bg=background")


class SegmentOptions(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Mask threshold")
    return_logits: bool = Field(False, description="Return soft logits")
    multimask_output: bool = Field(False, description="Return multiple mask candidates")


class SegmentRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG or JPEG image")
    width: int = Field(..., gt=0, le=4096, description="Image width")
    height: int = Field(..., gt=0, le=4096, description="Image height")
    points: list[Point] = Field(..., min_length=1, max_length=20, description="Point prompts")
    options: SegmentOptions = Field(default_factory=SegmentOptions)


class SegmentResponse(BaseModel):
    width: int
    height: int
    mask: str  # base64-encoded binary mask
    logits: str | None = None  # base64-encoded float32 array
    confidence: float | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    version: str


@app.on_event("startup")
async def load_model():
    """Load SAM2 model at server startup."""
    global predictor
    print(f"Loading SAM2 model on {DEVICE}...")
    model = build_sam2(MODEL_CONFIG, MODEL_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(model)
    print("Model loaded successfully!")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server health and model status."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model=MODEL_CONFIG.replace(".yaml", ""),
        device=DEVICE,
        version="1.0.0"
    )


@app.post("/segment", response_model=SegmentResponse)
async def segment(request: SegmentRequest):
    """Generate segmentation mask from image and point prompts."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)

        # Validate dimensions
        if image_array.shape[1] != request.width or image_array.shape[0] != request.height:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions mismatch: got {image_array.shape[1]}x{image_array.shape[0]}, expected {request.width}x{request.height}"
            )

        # Validate points are within bounds
        for point in request.points:
            if point.x >= request.width or point.y >= request.height:
                raise HTTPException(
                    status_code=400,
                    detail=f"Point ({point.x}, {point.y}) is outside image bounds"
                )

        # Convert points to SAM2 format
        point_coords = np.array([[p.x, p.y] for p in request.points])
        point_labels = np.array([1 if p.type == "fg" else 0 for p in request.points])

        # Run inference
        predictor.set_image(image_array)
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=request.options.multimask_output
        )

        # Select best mask (highest confidence)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        confidence = float(scores[best_idx])

        # Apply threshold and convert to binary (0 or 255)
        binary_mask = ((mask > request.options.threshold) * 255).astype(np.uint8)
        mask_b64 = base64.b64encode(binary_mask.tobytes()).decode("ascii")

        # Optionally return logits
        logits_b64 = None
        if request.options.return_logits:
            logits_b64 = base64.b64encode(logits[best_idx].astype(np.float32).tobytes()).decode("ascii")

        return SegmentResponse(
            width=request.width,
            height=request.height,
            mask=mask_b64,
            logits=logits_b64,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4.2 Running the Server

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install fastapi uvicorn pillow numpy torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# 3. Download model weights
mkdir -p checkpoints
wget -O checkpoints/sam2_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# 4. Run server
python sam2_server.py
# or: uvicorn sam2_server:app --host 0.0.0.0 --port 8000 --reload
```

### 4.3 Dependencies (requirements.txt)

```
fastapi>=0.100.0
uvicorn>=0.23.0
pillow>=10.0.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
pydantic>=2.0.0
```

---

## 5. Frontend Integration Notes

### 5.1 What Frontend Sends

```typescript
// Capture current viewport render as PNG
const canvas = document.createElement('canvas');
// ... render scene to canvas ...
const dataUrl = canvas.toDataURL('image/png');
const base64Image = dataUrl.split(',')[1];

// Collect user clicks (from SAM tool)
const points = [
  { x: 450, y: 300, type: 'fg' },  // user left-clicked here
  { x: 100, y: 100, type: 'bg' }   // user right-clicked here
];

// Send request
const response = await fetch('http://localhost:8000/segment', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: base64Image,
    width: canvas.width,
    height: canvas.height,
    points: points
  })
});

if (!response.ok) {
  const error = await response.json();
  throw new Error(error.message);
}

const data = await response.json();
```

### 5.2 What Frontend Receives

```typescript
const data = await response.json();

// Decode mask from base64
const maskBytes = Uint8Array.from(atob(data.mask), c => c.charCodeAt(0));

// maskBytes is width * height bytes
// maskBytes[y * width + x] = 0 (background) or 255 (foreground)

// Example: Check if pixel (100, 200) is selected
const x = 100, y = 200;
const isSelected = maskBytes[y * data.width + x] === 255;
```

### 5.3 Frontend Types (Already Defined)

The frontend types in `src/segmentation/types.ts` align with this API:

```typescript
interface SegmentationRequest {
  image: Uint8Array;  // Will be converted to base64
  width: number;
  height: number;
  points: SegmentationPoint[];
  options?: SegmentationOptions;
}

interface SegmentationResponse {
  width: number;
  height: number;
  mask: Uint8Array;  // Decoded from base64
  logits?: Float32Array;
}
```

---

## 6. Testing

### 6.1 Health Check Test
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","model":"sam2_hiera_l","device":"cuda","version":"1.0.0"}
```

### 6.2 Segmentation Test

Create a test script:
```python
import base64
import requests
from PIL import Image
import io

# Create a simple test image
image = Image.new('RGB', (100, 100), color='white')
buffer = io.BytesIO()
image.save(buffer, format='PNG')
image_b64 = base64.b64encode(buffer.getvalue()).decode()

# Send request
response = requests.post('http://localhost:8000/segment', json={
    'image': image_b64,
    'width': 100,
    'height': 100,
    'points': [{'x': 50, 'y': 50, 'type': 'fg'}]
})

print(response.status_code)
print(response.json())
```

### 6.3 Load Testing

```bash
# Install hey (HTTP load generator)
brew install hey  # macOS

# Test with 10 concurrent requests
hey -n 10 -c 2 -m POST \
  -H "Content-Type: application/json" \
  -D test_request.json \
  http://localhost:8000/segment
```

---

## 7. Future Extensions (POC 2)

### 7.1 Multi-View Segmentation

For better 3D selection accuracy, accept multiple views:

```
POST /segment/multiview
```

```json
{
  "views": [
    {
      "image": "<base64>",
      "width": 1920,
      "height": 1080,
      "points": [{"x": 450, "y": 300, "type": "fg"}],
      "camera": {
        "intrinsics": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
        "extrinsics": [/* 4x4 matrix */],
        "width": 1920,
        "height": 1080
      }
    },
    // ... more views
  ]
}
```

### 7.2 Video Segmentation

SAM2 supports video - track objects across frames:

```
POST /segment/video
```

This could be used for animated Gaussian splat sequences.

### 7.3 Box Prompts

Add support for bounding box prompts:

```json
{
  "image": "<base64>",
  "boxes": [
    { "x1": 100, "y1": 100, "x2": 500, "y2": 400 }
  ]
}
```

---

## 8. Summary

The SAM2 backend is a REST API server that:

1. **Accepts:** Base64-encoded image + point coordinates (foreground/background)
2. **Processes:** Runs SAM2 model inference
3. **Returns:** Binary segmentation mask (0=background, 255=foreground)

**Key Requirements:**
- CORS support for browser access
- < 2 second inference latency (with GPU)
- Binary mask format (H×W bytes)
- Proper error handling with clear error codes

**Minimal Viable Implementation:**
- FastAPI server
- Single `/segment` endpoint
- SAM2 hiera_large model
- GPU with 8GB+ VRAM

The frontend already has the types and provider interface defined - the backend just needs to implement this specification.
