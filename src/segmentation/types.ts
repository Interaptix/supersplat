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
 * Individual mask candidate from SAM2 decoder.
 * SAM2 outputs 3 masks with different granularity levels.
 */
export interface MaskCandidate {
    /** Mask index (0=tight, 1=medium, 2=broad) */
    index: number;
    /** IoU confidence score */
    iouScore: number;
    /** Binary mask (H*W bytes, 0=background, 255=foreground) */
    mask: Uint8Array;
    /** Mask width in pixels */
    width: number;
    /** Mask height in pixels */
    height: number;
    /** Raw logits for this mask (256x256) */
    logits: Float32Array;
}

/**
 * Response from segmentation service.
 */
export interface SegmentationResponse {
    /** Mask width in pixels */
    width: number;
    /** Mask height in pixels */
    height: number;
    /** Binary mask (H*W bytes, 0=background, 255=foreground) - the selected mask */
    mask: Uint8Array;
    /** Optional soft mask/logits for client-side thresholding */
    logits?: Float32Array;
    /** All mask candidates with IoU scores (for multi-mask selection UI) */
    allMasks?: MaskCandidate[];
    /** Index of the currently selected mask */
    selectedMaskIndex?: number;
}

/**
 * Selection operation type.
 */
export type SelectionOp = 'add' | 'remove' | 'set';
