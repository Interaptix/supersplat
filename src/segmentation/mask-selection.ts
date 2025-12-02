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
