# SuperSplat Gaussian Splat Editing via SAM2 / SAGS  
## Implementation Plan

_Last updated: 2025-12-02_

---

## 1. Goals & Scope

### 1.1 Primary goals

1. **POC 1 – Single-view smart selection (SAM2)**
   - Allow users to click in the SuperSplat viewport, place keypoints, and get a 2D segmentation mask from SAM2.
   - Convert the 2D mask into a **3D Gaussian selection** and plug it into existing selection / delete / hide flows.
   - Keep all heavy ML (SAM2) in a **backend service**, with SuperSplat as a pure frontend.

2. **POC 2 – Multi-view / SAGS-style selection**
   - Use multiple camera views and SAGS-like fusion to produce **stable, multi-view consistent selections**.
   - Optionally output **per-Gaussian weights** (0–1) instead of just hard selection.

### 1.2 Non-goals (for now)

- Running SAM2 fully in the browser (frontend-only).
- Full reproduction of SAGS training / optimization pipeline.
- Editing the renderer’s core Gaussian rasterization shaders.

---

## 2. High-Level Architecture

```text
+-------------------------------+       +-----------------------------+
| Azure Static Web App          |       | Azure Container Apps        |
| (SuperSplat Frontend)         |       | (Segmentation Service)      |
+-------------------------------+       +-----------------------------+
| - React/TS app                |  HTTPS| - Python + FastAPI          |
| - SuperSplat viewer/editor    +------>+ - PyTorch (SAM2, SAGS)      |
| - Selection UI & tools        |       | - /segment-single-view      |
| - 2D->3D selection logic      |       | - /segment-multiview        |
+-------------------------------+       +-----------------------------+
            |
            | engine API (JS/TS)
            v
+-------------------------------+
| GS Engine (PlayCanvas-based)  |
+-------------------------------+
| - Render-to-texture (RGB/ID)  |
| - Camera transforms           |
| - Gaussian projection helpers |
+-------------------------------+
```

- **SuperSplat** remains a static SPA hosted on Azure Static Web Apps.
- **Segmentation Service** runs as a separate containerized backend (Azure Container Apps / App Service).
- **Engine** provides small utility hooks for camera capture and projection; no SAM2/SAGS inside it.

---

## 3. SuperSplat (Frontend) Changes

### 3.1 Segmentation provider abstraction

Define a generic interface so that SuperSplat doesn’t care whether segmentation is done remotely or locally:

```ts
export interface SegmentationRequest {
  image: ImageData | ArrayBuffer; // rendered RGB
  points: { x: number; y: number; type: "fg" | "bg" }[];
  camera?: {
    intrinsics: number[];
    extrinsics: number[];
    width: number;
    height: number;
  };
  options?: {
    threshold?: number;
    multiViewSessionId?: string;
  };
}

export interface SegmentationResponse {
  width: number;
  height: number;
  mask: Uint8Array;        // H * W binary mask
  logits?: Float32Array;   // optional soft mask for client-side thresholding
}

export interface SegmentationProvider {
  segmentSingleView(req: SegmentationRequest): Promise<SegmentationResponse>;
  segmentMultiView?(reqs: SegmentationRequest[]): Promise<SegmentationResponse>;
}
```

Implementations:

- `RemoteSAM2Provider` – calls the backend API.
- (Future) `WebGPUSmallModelProvider` – optional local model, same interface.

### 3.2 New UI / UX

Add a **“Smart Selection (SAM2)”** mode:

- Tool toggle / icon in SuperSplat toolbar.
- While active:
  - Click in the viewport → add keypoint at cursor position.
  - Keypoints can be:
    - **Foreground** (default).
    - **Background** (modifier key / alternate tool).
  - Visualize keypoints as small circles with color coding.

Controls:

- **“Run segmentation”** button or auto-run after N keypoints.
- **Mask threshold slider**:
  - 0–1 (or 0–100%) to control mask hardness.
  - Tied to either:
    - Client-side re-thresholding of logits, or
    - Backend re-thresholding via options.

Optional:

- **“Expand / shrink”** mask radius slider (simple erosion/dilation in 2D).
- “Clear keypoints” and “Clear mask” buttons.

### 3.3 View capture & request building

On “Run segmentation”:

1. Determine active camera in SuperSplat.
2. Ask engine to render current view to an offscreen buffer:
   - At a reasonable resolution (e.g. 1024px max dimension).
   - Return RGB data (and optionally depth/ID).
3. Convert user click positions into **image pixel coordinates**.
4. Build a `SegmentationRequest`:
   - `image`: RGB buffer (or encoded PNG/JPEG).
   - `points`: {x, y, type}.
   - `camera`: intrinsics, extrinsics, width/height.
   - `options.threshold`: current mask threshold (if backend controls it).
5. Call `SegmentationProvider.segmentSingleView`.

### 3.4 Mask → 3D Gaussian selection

When `SegmentationResponse` arrives:

1. For each Gaussian (or for a filtered subset of “in-view” splats):
   - Use engine API to **project Gaussian center to screen space** for the current camera.
   - Obtain `(u, v)` pixel coordinates in the mask.
2. Sample `mask[v * width + u]` (and possibly check depth/ID for correctness):
   - If above threshold → mark Gaussian as “selected”.
3. Build a `SelectionSet<GaussianID>` and pass it into existing selection flows:
   - Use same mechanisms that box/lasso selection already use.
   - Existing delete/hide/invert operations should just work.

Optional optimization:

- If engine can render an **ID buffer** (per-Gaussian ID), we can invert control:
  - For each **mask pixel** that is active → look up the Gaussian ID from ID buffer.
  - Add that ID to the selection set.
  - This avoids per-Gaussian projection loops.

---

## 4. Engine Changes (PlayCanvas GS Layer)

Goal: keep changes minimal and utility-focused.

### 4.1 Render-to-texture APIs

Expose a function from the engine bridge:

```ts
interface RenderOutputs {
  color: ImageData | ArrayBuffer;      // required
  depth?: Float32Array | Uint16Array;  // optional
  idBuffer?: Uint32Array;              // optional per-Gaussian IDs
  width: number;
  height: number;
}

function renderCameraToBuffers(cameraId: string, options?: {
  width?: number;
  height?: number;
  includeDepth?: boolean;
  includeIdBuffer?: boolean;
}): Promise<RenderOutputs>;
```

- Internally, this sets up appropriate render targets in PlayCanvas and reads back their contents.

### 4.2 Projection helpers

Provide helpers (via JS/TS) for mapping Gaussians into screen-space:

```ts
function projectGaussianToScreen(
  gaussianId: number,
  cameraId: string
): { x: number; y: number; depth: number } | null;
```

Or a batch version:

```ts
function projectGaussiansToScreen(
  gaussianIds: number[],
  cameraId: string
): { id: number; x: number; y: number; depth: number }[];
```

These wrap world → clip → NDC → pixel coordinate transforms using the engine’s camera matrices.

### 4.3 Optional ID-buffer mode

To make 2D→3D mapping more robust and cheaper:

- Add an **“ID render pass”**:
  - A material/shader variant that writes a unique Gaussian ID per fragment.
  - Render to a `R32UI` (or equivalent) texture.
- `renderCameraToBuffers` can then return `idBuffer` alongside color.

This enables:

- For each active mask pixel, sample `idBuffer` to find the owning Gaussian.
- Avoid dealing with ambiguous projections and occlusion issues.

### 4.4 Data structures

Ensure we have:

- Stable `GaussianID` indices for the entire life of a loaded splat.
- A mapping from `GaussianID` → `position`, `radius`, etc., for projection math.

---

## 5. Segmentation Backend (SAM2 / SAGS Service)

### 5.1 Tech stack

- **Runtime**: Python 3.x
- **Web framework**: FastAPI (or Flask)
- **ML stack**:
  - PyTorch
  - SAM2 implementation
  - SAGS implementation (or custom multi-view fusion logic)

### 5.2 API endpoints

#### `POST /segment-single-view`

- Input (JSON or multipart):
  - `image`: RGB image (base64 or binary).
  - `points`: list of `{x, y, type}`.
  - `camera`: optional camera intrinsics/extrinsics.
  - `options`: `{threshold, modelVariant, ...}`.

- Output:
  - `width`, `height`
  - `mask`: binary mask (e.g. base64-encoded PNG or raw bytes)
  - Optional `logits` for client-side thresholding.

#### `POST /segment-multiview` (POC 2)

- Input:
  - List of `SegmentationRequest` objects or a structured payload:
    - Each with `image`, `camera`, `points` (or reused points in world space).
- Backend steps:
  1. Run SAM2 per view → per-view 2D masks.
  2. Backproject masks into 3D using camera & depth (if provided) or geometry.
  3. Fuse into per-Gaussian scores (0–1) using SAGS-like consistency.
- Output:
  - Either a fused 2D mask (main view) **or**
  - Direct `[GaussianID: weight]` array.

### 5.3 Threshold strategy

Two main modes:

1. **Backend controls threshold**:
   - Client sends `options.threshold`.
   - Backend applies threshold to logits and returns a binary mask.
   - Changing threshold requires calling backend again (unless it caches logits).

2. **Client controls threshold (recommended for POC 1)**:
   - Backend returns logits or soft mask.
   - SuperSplat:
     - Applies threshold locally.
     - Updates selection instantly when user moves the slider.

Backend should support both for flexibility.

### 5.4 Authentication & CORS

- Use an API key or JWT-based auth:
  - `Authorization: Bearer <token>` or `x-api-key: ...`.
- Configure CORS:
  - Allow origin: SuperSplat Static Web App domain.
  - Allow methods: `POST`, `OPTIONS`.
  - Allow headers: `Content-Type`, `Authorization` / `x-api-key`.

---

## 6. Azure Deployment & Wiring

### 6.1 SuperSplat (Azure Static Web Apps)

- Existing deployment pipeline remains:
  - Build React/TS app.
  - Deploy to Azure Static Web Apps.
- Add environment variables for:
  - `SEGMENTATION_API_URL`
  - `SEGMENTATION_API_KEY` (if using API key, or use client-side token acquisition logic for AAD).

### 6.2 Segmentation Service (Azure Container Apps or App Service)

- Build Docker image containing:
  - Python + FastAPI app.
  - SAM2/SAGS models & weights.
- Deploy image to:
  - **Azure Container Apps** (preferred for autoscaling), or
  - Azure App Service for Containers.
- Set env vars:
  - `MODEL_PATH`, `GPU_ENABLED`, etc.
- Configure scaling rules based on CPU/GPU/requests.

### 6.3 DNS & routing

- Optional custom domains:
  - `https://supersplat.yourdomain.com` → Static Web Apps.
  - `https://segmentation-api.yourdomain.com` → Container Apps / App Service.
- Update CORS on the backend to allow the frontend origin.

### 6.4 Environments

- **Dev**:
  - Dev SWA, dev segmentation API, permissive CORS.
- **Staging**:
  - Staging SWA + API, limited access.
- **Prod**:
  - Locked-down API (rate limiting, auth) + prod SWA.

---

## 7. Incremental Rollout Plan

### Phase 0 – Design & Engine hooks

- [ ] Define `SegmentationProvider` interface.
- [ ] Implement engine APIs:
  - `renderCameraToBuffers`.
  - `projectGaussianToScreen` (and/or ID buffer mode).
- [ ] Add simple debug UI to visualize RGB/ID render outputs.

### Phase 1 – Backend POC (SAM2 single-view)

- [ ] Implement `POST /segment-single-view` with a minimal SAM2 model.
- [ ] Accept one image + few keypoints, return a binary mask.
- [ ] Test with a standalone script / Jupyter notebook.

### Phase 2 – Frontend POC integration

- [ ] Implement `RemoteSAM2Provider`.
- [ ] Add “Smart Selection” tool + keypoint UI.
- [ ] Wire view capture → segmentation call → mask→3D selection.
- [ ] Verify selection correctness on simple scenes.

### Phase 3 – Threshold & UX refinement

- [ ] Add threshold slider and local re-thresholding.
- [ ] Add mask visualization overlay in 2D and 3D.
- [ ] Performance tuning:
  - Downsample capture resolution.
  - Optimize selection mapping.

### Phase 4 – Multi-view / SAGS-style refinement (POC 2)

- [ ] Extend backend with `POST /segment-multiview`.
- [ ] Decide on data contract:
  - Return per-Gaussian weights vs combined mask.
- [ ] Add UI flow for multi-view refinement:
  - “Add additional views”.
  - “Refine selection” button.
- [ ] Evaluate quality vs runtime on target hardware.

### Phase 5 – Hardening & polishing

- [ ] Proper auth & rate limiting on segmentation service.
- [ ] Logging and error reporting from frontend (failed calls, timeouts).
- [ ] Documentation & examples for internal users.

---

## 8. Open Questions / Risks

1. **Model size & latency**
   - How large a SAM2 variant can we run with acceptable latency?
   - Do we need GPU-backed instances in Azure?

2. **Per-Gaussian mapping accuracy**
   - Is projection-only mapping enough, or do we need ID buffer rendering for robust mapping in dense / overlapping regions?

3. **SAGS fidelity vs complexity**
   - How close do we need to be to full SAGS?
   - Can we approximate with fewer views and simpler fusion?

4. **Multi-user / multi-tenant concerns**
   - If Supersplat is used by many users, do we need isolation and quotas per user/tenant?

5. **Future local mode**
   - Do we want a future option to run a **tiny** segmentation model locally (e.g., WebGPU) for offline demos?
   - If yes, ensure the `SegmentationProvider` abstraction is clean enough to plug that in later.

---

This document should be enough to:

- Align on architecture (frontend vs backend vs engine).
- Split work between:
  - Engine / SuperSplat devs.
  - Backend / ML devs.
- Track progress through POC1 and POC2.
