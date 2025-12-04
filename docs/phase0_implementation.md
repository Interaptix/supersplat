# Phase 0: Segmentation Infrastructure Implementation Plan

_Last updated: 2025-12-02_

---

## 1. Overview

### 1.1 Purpose

Phase 0 establishes the foundational infrastructure required before backend SAM2 work can begin. This phase creates the abstraction layer and integration hooks that will allow the frontend to communicate with any segmentation backend.

### 1.2 Scope

**In scope:**
- `SegmentationProvider` interface definition
- Mask-to-selection bridge function
- Type definitions for requests/responses

**Out of scope:**
- Backend implementation (Phase 1)
- UI/UX for keypoint placement (Phase 2)
- Camera intrinsics/extrinsics extraction (deferred to POC 2)
- ID buffer rendering optimization (future enhancement)

### 1.3 Estimated Effort

~2 hours total

---

## 2. Prerequisites Analysis

### 2.1 What Already Exists

| Capability | Location | Description |
|------------|----------|-------------|
| RGB capture | `src/render.ts` | `render.offscreen` event returns `Uint8Array` RGBA at specified resolution |
| Mask selection | `src/editor.ts` | `select.byMask` event applies canvas mask to gaussian selection |
| Projection (centers) | `src/data-processor.ts` | `DataProcessor.intersect()` projects gaussians to screen space |
| Projection (rings) | `src/camera.ts` | `pickPrep()` + `pickRect()` for ID-based selection |

### 2.2 Existing Code References

#### RGB Capture (`src/render.ts`)
```typescript
events.function('render.offscreen', async (width: number, height: number): Promise<Uint8Array> => {
    scene.camera.startOffscreenMode(width, height);
    // ... renders frame, reads pixels ...
    return data; // Uint8Array RGBA
});
```

#### Mask Selection (`src/editor.ts`)
```typescript
events.on('select.byMask', (op: 'add'|'remove'|'set', canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) => {
    const mode = events.invoke('camera.mode');
    if (mode === 'centers') {
        // Uses DataProcessor.intersect() with mask texture
        intersectCenters(splat, op, { mask: maskTexture });
    } else if (mode === 'rings') {
        // Uses picker to get gaussian IDs from mask pixels
        // ...builds selection from picked IDs
    }
});
```

### 2.3 What Needs to Be Built

1. **Type definitions** - Request/response interfaces for segmentation
2. **SegmentationProvider interface** - Abstraction for any segmentation backend
3. **Mask-to-selection bridge** - Convert backend mask response to `select.byMask` call

---

## 3. Implementation Tasks

### 3.1 Task 1: Create Type Definitions

**File:** `src/segmentation/types.ts`

```typescript
/**
 * A point indicating foreground or background for segmentation.
 */
export interface SegmentationPoint {
    /** X coordinate in image pixels (0 = left edge) */
    x: number;
    /** Y coordinate in image pixels (0 = top edge) */
    y: number;
    /** Point type: foreground (include) or background (exclude) */
    type: 'fg' | 'bg';
}

/**
 * Camera parameters for multi-view segmentation (POC 2).
 * Optional for single-view SAM2.
 */
export interface CameraParams {
    /** Intrinsic matrix as flat array [fx, 0, cx, 0, fy, cy, 0, 0, 1] */
    intrinsics: number[];
    /** Extrinsic matrix (world-to-camera) as flat 4x4 array */
    extrinsics: number[];
    /** Image width in pixels */
    width: number;
    /** Image height in pixels */
    height: number;
}

/**
 * Options for segmentation request.
 */
export interface SegmentationOptions {
    /** Threshold for mask binarization (0-1). Default: 0.5 */
    threshold?: number;
    /** Session ID for multi-view refinement (POC 2) */
    multiViewSessionId?: string;
    /** Model variant to use (if backend supports multiple) */
    modelVariant?: string;
}

/**
 * Request payload for segmentation.
 */
export interface SegmentationRequest {
    /** RGBA image data from render.offscreen */
    image: Uint8Array;
    /** Image width in pixels */
    width: number;
    /** Image height in pixels */
    height: number;
    /** Keypoints indicating foreground/background regions */
    points: SegmentationPoint[];
    /** Optional camera parameters (for POC 2 multi-view) */
    camera?: CameraParams;
    /** Optional segmentation options */
    options?: SegmentationOptions;
}

/**
 * Response from segmentation service.
 */
export interface SegmentationResponse {
    /** Mask width in pixels */
    width: number;
    /** Mask height in pixels */
    height: number;
    /** Binary mask (H*W bytes, 0=background, 255=foreground) */
    mask: Uint8Array;
    /** Optional soft mask/logits for client-side thresholding */
    logits?: Float32Array;
}

/**
 * Selection operation type.
 */
export type SelectionOp = 'add' | 'remove' | 'set';
```

**Rationale:**
- Matches the interface defined in the architecture plan
- `camera` is optional since POC 1 doesn't need it
- `logits` enables client-side threshold adjustment without re-calling backend

---

### 3.2 Task 2: Create SegmentationProvider Interface

**File:** `src/segmentation/provider.ts`

```typescript
import { SegmentationRequest, SegmentationResponse } from './types';

/**
 * Abstract interface for segmentation providers.
 * Implementations can be remote (API) or local (WebGPU model).
 */
export interface SegmentationProvider {
    /**
     * Unique identifier for this provider.
     */
    readonly id: string;

    /**
     * Human-readable name for UI display.
     */
    readonly name: string;

    /**
     * Whether this provider is currently available.
     * May depend on network connectivity, API key, etc.
     */
    isAvailable(): Promise<boolean>;

    /**
     * Perform single-view segmentation.
     * 
     * @param request - Segmentation request with image and keypoints
     * @returns Promise resolving to segmentation response with mask
     * @throws Error if segmentation fails
     */
    segmentSingleView(request: SegmentationRequest): Promise<SegmentationResponse>;

    /**
     * Perform multi-view segmentation (POC 2).
     * Optional - not all providers support this.
     * 
     * @param requests - Array of segmentation requests from different views
     * @returns Promise resolving to fused segmentation response
     */
    segmentMultiView?(requests: SegmentationRequest[]): Promise<SegmentationResponse>;

    /**
     * Abort any in-progress segmentation request.
     */
    abort?(): void;
}

/**
 * Error thrown when segmentation fails.
 */
export class SegmentationError extends Error {
    constructor(
        message: string,
        public readonly code: 'NETWORK_ERROR' | 'TIMEOUT' | 'INVALID_RESPONSE' | 'SERVER_ERROR' | 'ABORTED',
        public readonly details?: unknown
    ) {
        super(message);
        this.name = 'SegmentationError';
    }
}
```

**Rationale:**
- Clean abstraction allows swapping backends (remote SAM2, local WebGPU, mock)
- `isAvailable()` enables graceful degradation if backend is down
- `abort()` supports cancellation for UX responsiveness
- Typed errors help with UI error handling

---

### 3.3 Task 3: Create Mask-to-Selection Bridge

**File:** `src/segmentation/mask-selection.ts`

```typescript
import { Events } from '../events';
import { SegmentationResponse, SelectionOp } from './types';

/**
 * Options for mask-to-selection conversion.
 */
export interface MaskSelectionOptions {
    /** Selection operation: add, remove, or set */
    op: SelectionOp;
    /** Threshold for binarizing logits (0-1). Only used if response has logits. */
    threshold?: number;
    /** Target canvas size. If different from mask size, will scale. */
    targetWidth?: number;
    targetHeight?: number;
}

/**
 * Convert a segmentation response mask into a selection by firing the
 * 'select.byMask' event with an appropriately constructed canvas.
 * 
 * @param response - Segmentation response containing the mask
 * @param options - Selection options
 * @param events - Events instance to fire selection event
 */
export function applyMaskToSelection(
    response: SegmentationResponse,
    options: MaskSelectionOptions,
    events: Events
): void {
    const { width, height, mask, logits } = response;
    const { op, threshold = 0.5, targetWidth, targetHeight } = options;

    // Create canvas at mask dimensions
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth ?? width;
    canvas.height = targetHeight ?? height;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
        throw new Error('Failed to create canvas 2D context');
    }

    // Determine which mask data to use
    let maskData: Uint8Array;
    if (logits && threshold !== undefined) {
        // Apply threshold to logits (convert from float [-inf, inf] to binary)
        maskData = new Uint8Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            // Logits are typically in log-odds space, apply sigmoid
            const prob = 1 / (1 + Math.exp(-logits[i]));
            maskData[i] = prob >= threshold ? 255 : 0;
        }
    } else {
        maskData = mask;
    }

    // Create ImageData from mask
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < maskData.length; i++) {
        const v = maskData[i];
        imageData.data[i * 4] = v;       // R
        imageData.data[i * 4 + 1] = v;   // G
        imageData.data[i * 4 + 2] = v;   // B
        imageData.data[i * 4 + 3] = v;   // A (255 = selected, 0 = not)
    }

    // If target size differs, we need to scale
    if (targetWidth && targetHeight && (targetWidth !== width || targetHeight !== height)) {
        // Create temporary canvas at original size
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.putImageData(imageData, 0, 0);

        // Scale to target size
        ctx.imageSmoothingEnabled = false; // Keep hard edges
        ctx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
    } else {
        ctx.putImageData(imageData, 0, 0);
    }

    // Fire the selection event
    events.fire('select.byMask', op, canvas, ctx);
}

/**
 * Create a preview canvas from a segmentation response without applying selection.
 * Useful for visualizing the mask before committing.
 * 
 * @param response - Segmentation response containing the mask
 * @param threshold - Threshold for binarizing logits (0-1)
 * @returns Canvas element with mask visualization
 */
export function createMaskPreviewCanvas(
    response: SegmentationResponse,
    threshold: number = 0.5
): HTMLCanvasElement {
    const { width, height, mask, logits } = response;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;

    // Determine mask values
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < mask.length; i++) {
        let v: number;
        if (logits) {
            const prob = 1 / (1 + Math.exp(-logits[i]));
            v = prob >= threshold ? 255 : 0;
        } else {
            v = mask[i];
        }
        // Semi-transparent red overlay for preview
        imageData.data[i * 4] = v > 0 ? 255 : 0;      // R
        imageData.data[i * 4 + 1] = 0;                 // G
        imageData.data[i * 4 + 2] = 0;                 // B
        imageData.data[i * 4 + 3] = v > 0 ? 128 : 0;  // A (semi-transparent)
    }
    ctx.putImageData(imageData, 0, 0);

    return canvas;
}
```

**Rationale:**
- Bridges the gap between backend response format and existing `select.byMask` event
- Supports client-side thresholding via logits (slider UX)
- `createMaskPreviewCanvas()` enables mask visualization before applying
- Handles resolution mismatch if backend returns different size than viewport

---

### 3.4 Task 4: Create Index File

**File:** `src/segmentation/index.ts`

```typescript
// Types
export type {
    SegmentationPoint,
    CameraParams,
    SegmentationOptions,
    SegmentationRequest,
    SegmentationResponse,
    SelectionOp
} from './types';

// Provider interface
export type { SegmentationProvider } from './provider';
export { SegmentationError } from './provider';

// Mask selection utilities
export {
    applyMaskToSelection,
    createMaskPreviewCanvas
} from './mask-selection';
export type { MaskSelectionOptions } from './mask-selection';
```

---

## 4. File Structure

```
src/
├── segmentation/           # NEW - Phase 0
│   ├── index.ts           # Public exports
│   ├── types.ts           # Type definitions
│   ├── provider.ts        # SegmentationProvider interface
│   └── mask-selection.ts  # Mask-to-selection bridge
├── editor.ts              # (existing - select.byMask handler)
├── render.ts              # (existing - render.offscreen)
├── data-processor.ts      # (existing - intersect with mask)
└── ...
```

---

## 5. Testing Plan

### 5.1 Unit Tests (if test framework exists)

```typescript
// Test mask-to-selection bridge
describe('applyMaskToSelection', () => {
    it('should create canvas from binary mask', () => {
        const response: SegmentationResponse = {
            width: 2,
            height: 2,
            mask: new Uint8Array([255, 0, 0, 255])
        };
        // Mock events, verify select.byMask called with correct canvas
    });

    it('should apply threshold to logits', () => {
        const response: SegmentationResponse = {
            width: 2,
            height: 2,
            mask: new Uint8Array([0, 0, 0, 0]),
            logits: new Float32Array([2.0, -2.0, -2.0, 2.0]) // sigmoid: ~0.88, ~0.12, ~0.12, ~0.88
        };
        // With threshold 0.5, should select indices 0 and 3
    });
});
```

### 5.2 Manual Integration Test

1. Load a splat scene in SuperSplat
2. Open browser console
3. Run test script:

```typescript
// Manual test in browser console
(async () => {
    const events = window.__supersplat_events; // Assuming exposed for debug
    
    // 1. Capture current view
    const rgba = await events.invoke('render.offscreen', 512, 512);
    console.log('Captured RGBA:', rgba.length, 'bytes');
    
    // 2. Create mock segmentation response (center square selected)
    const mockResponse = {
        width: 512,
        height: 512,
        mask: new Uint8Array(512 * 512)
    };
    // Fill center 256x256 region
    for (let y = 128; y < 384; y++) {
        for (let x = 128; x < 384; x++) {
            mockResponse.mask[y * 512 + x] = 255;
        }
    }
    
    // 3. Apply to selection
    const { applyMaskToSelection } = await import('./src/segmentation/index.js');
    applyMaskToSelection(mockResponse, { op: 'set' }, events);
    
    console.log('Selection applied - check viewport');
})();
```

---

## 6. Definition of Done

### 6.1 Checklist

- [ ] `src/segmentation/types.ts` created with all type definitions
- [ ] `src/segmentation/provider.ts` created with interface and error class
- [ ] `src/segmentation/mask-selection.ts` created with bridge functions
- [ ] `src/segmentation/index.ts` created with exports
- [ ] Code compiles without TypeScript errors
- [ ] Manual integration test passes (mock mask → selection works)
- [ ] Code follows project conventions (formatting, naming)

### 6.2 Acceptance Criteria

1. **Interface Stability**: The `SegmentationProvider` interface can be implemented by backend developers without frontend changes
2. **Integration Works**: A mock `SegmentationResponse` can be converted to gaussian selection using existing `select.byMask` infrastructure
3. **Threshold Support**: Client-side threshold adjustment works with logits response

---

## 7. Follow-on Work (Phase 1 Dependencies)

Phase 1 (Backend POC) will need:

| What | From Phase 0 | Notes |
|------|--------------|-------|
| Request format | `SegmentationRequest` | Backend implements endpoint matching this |
| Response format | `SegmentationResponse` | Backend returns data in this shape |
| Image format | `Uint8Array` RGBA | May need to convert to PNG/JPEG for transmission |

Phase 2 (Frontend POC) will need:

| What | From Phase 0 | Notes |
|------|--------------|-------|
| `SegmentationProvider` | Interface | Implement `RemoteSAM2Provider` |
| `applyMaskToSelection` | Bridge function | Wire to UI "Apply" button |
| `createMaskPreviewCanvas` | Preview utility | Overlay on viewport |

---

## 8. Open Questions

1. **Image encoding**: Should we send raw RGBA or encode to PNG/JPEG? PNG is lossless but larger; JPEG is smaller but lossy. Recommendation: Start with raw, optimize later.

2. **Resolution**: What resolution should we capture? 512x512? 1024x1024? Larger = better accuracy but slower. Recommendation: 1024 max dimension, maintain aspect ratio.

3. **Event exposure**: For debugging, should we expose `events` globally? Current code doesn't seem to. May need to add debug hook.

---

_End of Phase 0 Implementation Plan_
