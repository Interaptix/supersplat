/**
 * Image conversion utilities for SAM2 inference.
 * Handles conversion between Canvas/ImageData and ONNX tensors.
 */

import type * as ort from 'onnxruntime-web';
import type { DebugImageData, DebugDataType } from './sam2-worker';

/** SAM2 model input image size */
export const SAM2_INPUT_SIZE = 1024;

/** Debug mode flag - set to true to enable debug downloads */
export let DEBUG_SAM2 = false;

/** Callback type for sending debug data from worker to main thread */
export type DebugDataCallback = (data: DebugImageData) => void;

/** Global callback for debug data (set by sam2-core when running in worker) */
let debugDataCallback: DebugDataCallback | null = null;

/**
 * Set the debug data callback (called from worker context).
 */
export function setDebugDataCallback(callback: DebugDataCallback | null): void {
    debugDataCallback = callback;
}

/**
 * Check if we're running in a worker context (no DOM).
 */
function isWorkerContext(): boolean {
    return typeof document === 'undefined';
}

/**
 * Download a blob as a file (main thread only).
 */
function downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Download canvas as PNG image (main thread only).
 */
export async function downloadCanvasAsPNG(canvas: OffscreenCanvas, filename: string): Promise<void> {
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    downloadBlob(blob, filename);
}

/**
 * Convert OffscreenCanvas to RGBA Uint8Array.
 */
function canvasToRGBA(canvas: OffscreenCanvas): Uint8Array {
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return new Uint8Array(imageData.data.buffer);
}

/**
 * Send debug data to main thread (worker) or download directly (main thread).
 */
async function sendDebugData(
    canvas: OffscreenCanvas,
    debugType: DebugDataType,
    filename: string,
    extra?: Partial<DebugImageData>
): Promise<void> {
    if (isWorkerContext()) {
        // In worker: send data via callback
        if (debugDataCallback) {
            const imageData = canvasToRGBA(canvas);
            debugDataCallback({
                debugType,
                imageData,
                width: canvas.width,
                height: canvas.height,
                filename,
                ...extra
            });
        }
    } else {
        // In main thread: download directly
        await downloadCanvasAsPNG(canvas, filename);
    }
}

/**
 * Debug: Save the preprocessed 1024x1024 RGB image going to encoder.
 */
export async function debugSaveEncoderInput(
    imageData: Uint8Array,
    width: number,
    height: number
): Promise<void> {
    if (!DEBUG_SAM2) return;

    // Recreate the 1024x1024 preprocessed image
    const canvas = new OffscreenCanvas(SAM2_INPUT_SIZE, SAM2_INPUT_SIZE);
    const ctx = canvas.getContext('2d')!;

    // Create ImageData from input
    const inputCanvas = new OffscreenCanvas(width, height);
    const inputCtx = inputCanvas.getContext('2d')!;
    const clampedArray = new Uint8ClampedArray(imageData.length);
    clampedArray.set(imageData);
    const inputImageData = new ImageData(clampedArray, width, height);
    inputCtx.putImageData(inputImageData, 0, 0);

    // Resize to SAM2 input size (1024x1024)
    ctx.drawImage(inputCanvas, 0, 0, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE);

    console.log(`[SAM2 DEBUG] Saving encoder input: ${SAM2_INPUT_SIZE}x${SAM2_INPUT_SIZE}`);
    const filename = `sam2_encoder_input_${SAM2_INPUT_SIZE}x${SAM2_INPUT_SIZE}.png`;
    await sendDebugData(canvas, 'encoder_input', filename);
}

/**
 * Debug: Save the raw decoder output mask at native 256x256 resolution.
 */
export async function debugSaveDecoderOutput(
    mask: Uint8Array,
    width: number,
    height: number,
    logits: Float32Array
): Promise<void> {
    if (!DEBUG_SAM2) return;

    // Create canvas with mask
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < mask.length; i++) {
        const idx = i * 4;
        imageData.data[idx] = mask[i];     // R
        imageData.data[idx + 1] = mask[i]; // G
        imageData.data[idx + 2] = mask[i]; // B
        imageData.data[idx + 3] = 255;     // A
    }
    ctx.putImageData(imageData, 0, 0);

    // Calculate logits statistics
    let min = Infinity, max = -Infinity, sum = 0;
    let positiveCount = 0;
    for (let i = 0; i < logits.length; i++) {
        const v = logits[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v > 0) positiveCount++;
    }
    const mean = sum / logits.length;

    console.log(`[SAM2 DEBUG] Decoder output logits stats:`);
    console.log(`  - Shape: 1x1x${height}x${width}`);
    console.log(`  - Min: ${min.toFixed(4)}, Max: ${max.toFixed(4)}, Mean: ${mean.toFixed(4)}`);
    console.log(`  - Positive (foreground) pixels: ${positiveCount} / ${logits.length} (${(100 * positiveCount / logits.length).toFixed(1)}%)`);

    console.log(`[SAM2 DEBUG] Saving decoder output: ${width}x${height}`);
    const filename = `sam2_decoder_output_${width}x${height}.png`;
    await sendDebugData(canvas, 'decoder_output', filename, {
        logitsStats: {
            min,
            max,
            mean,
            positiveCount,
            totalCount: logits.length
        }
    });
}

/**
 * Debug: Save ALL decoder output masks (SAM2 outputs multiple mask candidates).
 * Each mask is saved with its IoU score in the filename.
 */
export async function debugSaveAllMasks(
    maskTensor: { data: Float32Array; dims: readonly number[] },
    iouScores: Float32Array | null
): Promise<void> {
    if (!DEBUG_SAM2) return;

    const dims = maskTensor.dims;
    const numMasks = dims[1] as number;
    const height = dims[2] as number;
    const width = dims[3] as number;
    const maskSize = height * width;

    console.log(`[SAM2 DEBUG] Saving all ${numMasks} mask candidates...`);

    for (let maskIndex = 0; maskIndex < numMasks; maskIndex++) {
        // Extract this mask's data
        const mask = new Uint8Array(maskSize);
        const maskOffset = maskIndex * maskSize;
        const data = maskTensor.data;
        
        let positiveCount = 0;
        for (let i = 0; i < maskSize; i++) {
            const isForeground = data[maskOffset + i] > 0;
            mask[i] = isForeground ? 255 : 0;
            if (isForeground) positiveCount++;
        }

        // Create canvas with mask
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d')!;
        const imageData = ctx.createImageData(width, height);

        for (let i = 0; i < mask.length; i++) {
            const idx = i * 4;
            imageData.data[idx] = mask[i];     // R
            imageData.data[idx + 1] = mask[i]; // G
            imageData.data[idx + 2] = mask[i]; // B
            imageData.data[idx + 3] = 255;     // A
        }
        ctx.putImageData(imageData, 0, 0);

        // Build filename with IoU score
        const iouStr = iouScores ? `_iou${iouScores[maskIndex].toFixed(4)}` : '';
        const coverage = (100 * positiveCount / maskSize).toFixed(1);
        const filename = `sam2_mask_${maskIndex}${iouStr}_${coverage}pct.png`;
        
        console.log(`[SAM2 DEBUG] Mask ${maskIndex}: IoU=${iouScores ? iouScores[maskIndex].toFixed(4) : 'N/A'}, coverage=${coverage}%`);
        await sendDebugData(canvas, 'decoder_output', filename);
    }
}

/**
 * Debug: Save mask overlay on original image (like SAM2 demo visualization).
 * Creates a semi-transparent colored overlay on foreground regions.
 */
export async function debugSaveMaskOverlay(
    originalImage: Uint8Array,
    originalWidth: number,
    originalHeight: number,
    mask: Uint8Array,
    maskWidth: number,
    maskHeight: number,
    overlayColor: [number, number, number, number] = [30, 200, 30, 128]  // Green with 50% opacity
): Promise<void> {
    if (!DEBUG_SAM2) return;

    // Create canvas at original image size
    const canvas = new OffscreenCanvas(originalWidth, originalHeight);
    const ctx = canvas.getContext('2d')!;

    // Draw original image first
    const inputCanvas = new OffscreenCanvas(originalWidth, originalHeight);
    const inputCtx = inputCanvas.getContext('2d')!;
    const clampedArray = new Uint8ClampedArray(originalImage.length);
    clampedArray.set(originalImage);
    const inputImageData = new ImageData(clampedArray, originalWidth, originalHeight);
    inputCtx.putImageData(inputImageData, 0, 0);
    ctx.drawImage(inputCanvas, 0, 0);

    // Resize mask to match original image size
    const resizedMask = resizeMask(mask, maskWidth, maskHeight, originalWidth, originalHeight);

    // Get the image data to apply overlay
    const outputImageData = ctx.getImageData(0, 0, originalWidth, originalHeight);
    const pixels = outputImageData.data;

    // Alpha blend overlay color where mask is foreground
    const alpha = overlayColor[3] / 255;
    const invAlpha = 1 - alpha;

    for (let i = 0; i < resizedMask.length; i++) {
        if (resizedMask[i] > 127) {  // Foreground pixel
            const idx = i * 4;
            // Blend: result = original * (1 - alpha) + overlay * alpha
            pixels[idx] = Math.round(pixels[idx] * invAlpha + overlayColor[0] * alpha);         // R
            pixels[idx + 1] = Math.round(pixels[idx + 1] * invAlpha + overlayColor[1] * alpha); // G
            pixels[idx + 2] = Math.round(pixels[idx + 2] * invAlpha + overlayColor[2] * alpha); // B
            // Alpha channel stays at 255
        }
    }

    ctx.putImageData(outputImageData, 0, 0);

    console.log(`[SAM2 DEBUG] Saving mask overlay: ${originalWidth}x${originalHeight}`);
    const filename = `sam2_overlay_${originalWidth}x${originalHeight}.png`;
    await sendDebugData(canvas, 'mask_overlay', filename);
}

/**
 * Debug: Save the final resized mask.
 */
export async function debugSaveFinalMask(
    mask: Uint8Array,
    width: number,
    height: number
): Promise<void> {
    if (!DEBUG_SAM2) return;

    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < mask.length; i++) {
        const idx = i * 4;
        imageData.data[idx] = mask[i];     // R
        imageData.data[idx + 1] = mask[i]; // G
        imageData.data[idx + 2] = mask[i]; // B
        imageData.data[idx + 3] = 255;     // A
    }
    ctx.putImageData(imageData, 0, 0);

    console.log(`[SAM2 DEBUG] Saving final mask: ${width}x${height}`);
    const filename = `sam2_final_mask_${width}x${height}.png`;
    await sendDebugData(canvas, 'final_mask', filename);
}

/**
 * Debug: Save image with points overlay.
 */
export async function debugSavePointsOverlay(
    imageData: Uint8Array,
    originalWidth: number,
    originalHeight: number,
    points: Array<{ x: number; y: number; type: 'fg' | 'bg' }>
): Promise<void> {
    if (!DEBUG_SAM2) return;

    // Create canvas at 1024x1024 (same as encoder input)
    const canvas = new OffscreenCanvas(SAM2_INPUT_SIZE, SAM2_INPUT_SIZE);
    const ctx = canvas.getContext('2d')!;

    // Draw the original image resized to 1024x1024
    const inputCanvas = new OffscreenCanvas(originalWidth, originalHeight);
    const inputCtx = inputCanvas.getContext('2d')!;
    const clampedArray = new Uint8ClampedArray(imageData.length);
    clampedArray.set(imageData);
    const inputImageData = new ImageData(clampedArray, originalWidth, originalHeight);
    inputCtx.putImageData(inputImageData, 0, 0);
    ctx.drawImage(inputCanvas, 0, 0, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE);

    // Draw points (they are already in 1024x1024 coordinate space)
    const pointRadius = 10;
    for (const point of points) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, pointRadius, 0, 2 * Math.PI);
        
        if (point.type === 'fg') {
            ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';  // Green for foreground
            ctx.strokeStyle = 'white';
        } else {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';  // Red for background
            ctx.strokeStyle = 'white';
        }
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw crosshair
        ctx.beginPath();
        ctx.moveTo(point.x - pointRadius - 5, point.y);
        ctx.lineTo(point.x + pointRadius + 5, point.y);
        ctx.moveTo(point.x, point.y - pointRadius - 5);
        ctx.lineTo(point.x, point.y + pointRadius + 5);
        ctx.strokeStyle = point.type === 'fg' ? 'lime' : 'red';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Add legend
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, 10, 180, 60);
    ctx.font = '14px Arial';
    ctx.fillStyle = 'lime';
    ctx.fillText('● Foreground (fg)', 20, 35);
    ctx.fillStyle = 'red';
    ctx.fillText('● Background (bg)', 20, 55);

    console.log(`[SAM2 DEBUG] Saving points overlay with ${points.length} points`);
    for (const p of points) {
        console.log(`  - Point (${p.x.toFixed(1)}, ${p.y.toFixed(1)}) type=${p.type}`);
    }
    const filename = `sam2_points_overlay_${SAM2_INPUT_SIZE}x${SAM2_INPUT_SIZE}.png`;
    await sendDebugData(canvas, 'points_overlay', filename, { points });
}

/**
 * Download debug image data from worker on main thread.
 * This is called by webgpu-provider when receiving debug messages.
 */
export async function downloadDebugImage(data: DebugImageData): Promise<void> {
    // Create canvas from RGBA data
    const canvas = new OffscreenCanvas(data.width, data.height);
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(data.width, data.height);
    
    // Copy RGBA data
    imageData.data.set(new Uint8ClampedArray(data.imageData.buffer));
    ctx.putImageData(imageData, 0, 0);

    // Log additional info
    if (data.logitsStats) {
        console.log(`[SAM2 DEBUG] Received ${data.debugType} from worker:`);
        console.log(`  - Shape: ${data.height}x${data.width}`);
        console.log(`  - Logits: Min=${data.logitsStats.min.toFixed(4)}, Max=${data.logitsStats.max.toFixed(4)}, Mean=${data.logitsStats.mean.toFixed(4)}`);
        console.log(`  - Positive: ${data.logitsStats.positiveCount}/${data.logitsStats.totalCount} (${(100 * data.logitsStats.positiveCount / data.logitsStats.totalCount).toFixed(1)}%)`);
    }

    if (data.points) {
        console.log(`[SAM2 DEBUG] Points overlay with ${data.points.length} points`);
    }

    // Download
    await downloadCanvasAsPNG(canvas, data.filename);
}

/** ORT Module type */
export type OrtModule = typeof ort;

/**
 * Preprocess RGBA image data for SAM2 encoder.
 * Converts RGBA Uint8Array to normalized RGB Float32 tensor.
 * 
 * NOTE: Uses simple [0-1] normalization to match the reference SAM2 WebGPU implementation
 * (https://github.com/geronimi73/next-sam). The HuggingFace SAM2 ONNX export handles
 * normalization internally, so we should NOT apply ImageNet normalization here.
 * 
 * @param ortModule - ONNX Runtime module
 * @param imageData - RGBA pixel data
 * @param width - Image width
 * @param height - Image height
 * @returns ONNX tensor ready for encoder input [1, 3, 1024, 1024]
 */
export function preprocessImageWithOrt(
    ortModule: OrtModule,
    imageData: Uint8Array,
    width: number,
    height: number
): ort.Tensor {
    // Create canvas for resizing
    const canvas = new OffscreenCanvas(SAM2_INPUT_SIZE, SAM2_INPUT_SIZE);
    const ctx = canvas.getContext('2d')!;

    // Create ImageData from input
    const inputCanvas = new OffscreenCanvas(width, height);
    const inputCtx = inputCanvas.getContext('2d')!;
    const clampedArray = new Uint8ClampedArray(imageData.length);
    clampedArray.set(imageData);
    const inputImageData = new ImageData(clampedArray, width, height);
    inputCtx.putImageData(inputImageData, 0, 0);

    // Resize to SAM2 input size (1024x1024)
    ctx.drawImage(inputCanvas, 0, 0, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE);
    const resizedData = ctx.getImageData(0, 0, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE);

    // Convert to normalized RGB float tensor with simple [0-1] normalization
    // This matches the reference implementation (geronimi73/next-sam imageutils.js)
    // The HuggingFace SAM2 ONNX model expects [0-1] normalized inputs
    const numPixels = SAM2_INPUT_SIZE * SAM2_INPUT_SIZE;
    const float32Data = new Float32Array(3 * numPixels);

    for (let i = 0; i < numPixels; i++) {
        const rgbaIndex = i * 4;
        // Simple [0-255] to [0-1] normalization (NO ImageNet mean/std)
        float32Data[i] = resizedData.data[rgbaIndex] / 255;                    // R
        float32Data[numPixels + i] = resizedData.data[rgbaIndex + 1] / 255;    // G
        float32Data[2 * numPixels + i] = resizedData.data[rgbaIndex + 2] / 255; // B
    }

    return new ortModule.Tensor('float32', float32Data, [1, 3, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE]);
}

/**
 * Scale point coordinates from original image to SAM2 input size.
 * 
 * @param x - X coordinate in original image
 * @param y - Y coordinate in original image
 * @param originalWidth - Original image width
 * @param originalHeight - Original image height
 * @returns Scaled coordinates for SAM2 model
 */
export function scalePoint(
    x: number,
    y: number,
    originalWidth: number,
    originalHeight: number
): { x: number; y: number } {
    return {
        x: (x / originalWidth) * SAM2_INPUT_SIZE,
        y: (y / originalHeight) * SAM2_INPUT_SIZE
    };
}

/**
 * Create point coordinates tensor for SAM2 decoder.
 * 
 * @param ort - ONNX Runtime module
 * @param points - Array of {x, y, type} points in SAM2 coordinate space
 * @returns ONNX tensor [1, N, 2] for point coordinates
 */
export function createPointCoordsTensorWithOrt(
    ortModule: OrtModule,
    points: Array<{ x: number; y: number; type: 'fg' | 'bg' }>
): ort.Tensor {
    const numPoints = points.length;
    const coords = new Float32Array(numPoints * 2);

    for (let i = 0; i < numPoints; i++) {
        coords[i * 2] = points[i].x;
        coords[i * 2 + 1] = points[i].y;
    }

    return new ortModule.Tensor('float32', coords, [1, numPoints, 2]);
}

/**
 * Create point labels tensor for SAM2 decoder.
 * 
 * @param ortModule - ONNX Runtime module
 * @param points - Array of points with type (fg=1, bg=0)
 * @returns ONNX tensor [1, N] for point labels
 */
export function createPointLabelsTensorWithOrt(
    ortModule: OrtModule,
    points: Array<{ x: number; y: number; type: 'fg' | 'bg' }>
): ort.Tensor {
    const numPoints = points.length;
    const labels = new Float32Array(numPoints);

    for (let i = 0; i < numPoints; i++) {
        // SAM2 labels: 1 = foreground, 0 = background
        labels[i] = points[i].type === 'fg' ? 1 : 0;
    }

    return new ortModule.Tensor('float32', labels, [1, numPoints]);
}

/**
 * Process SAM2 decoder output mask.
 * Converts logits to binary mask.
 * 
 * SAM2 decoder outputs multiple masks: [1, num_masks, H, W]
 * - num_masks is typically 3 or 4
 * - Each mask represents a different quality/confidence level
 * - Use maskIndex to select which mask to extract (default 0 = best mask based on IoU)
 * 
 * @param logits - Raw output from decoder [1, num_masks, 256, 256]
 * @param maskIndex - Which mask to extract (0 = first/best mask)
 * @param threshold - Threshold for binarization (default 0.0 for logits)
 * @returns Binary mask at model output resolution
 */
export function processMaskLogits(
    logits: ort.Tensor,
    maskIndex: number = 0,
    threshold: number = 0.0
): { mask: Uint8Array; width: number; height: number } {
    const data = logits.data as Float32Array;
    const dims = logits.dims;
    
    // Output format: [batch, num_masks, height, width]
    // Typically: [1, 4, 256, 256] or [1, 3, 256, 256]
    const numMasks = dims[1] as number;
    const height = dims[2] as number;
    const width = dims[3] as number;
    const maskSize = height * width;
    
    console.log(`[SAM2 DEBUG???] processMaskLogits: dims=[${dims.join(', ')}], numMasks=${numMasks}, selecting mask ${maskIndex}`);
    
    // Validate mask index
    if (maskIndex < 0 || maskIndex >= numMasks) {
        console.warn(`[SAM2] Invalid maskIndex ${maskIndex}, clamping to valid range [0, ${numMasks - 1}]`);
        maskIndex = Math.max(0, Math.min(maskIndex, numMasks - 1));
    }
    
    const mask = new Uint8Array(maskSize);
    
    // Extract only the selected mask's data
    // Data layout is [batch][mask][height][width], so mask N starts at offset N * maskSize
    const maskOffset = maskIndex * maskSize;
    
    for (let i = 0; i < maskSize; i++) {
        // Logits > threshold means foreground
        mask[i] = data[maskOffset + i] > threshold ? 255 : 0;
    }

    return { mask, width, height };
}

/**
 * Resize mask to target dimensions for BINARY selection.
 * Uses bilinear interpolation via canvas, then re-thresholds for hard edges.
 * 
 * NOTE: This function re-thresholds at 127 after bilinear interpolation,
 * which is appropriate for selection but causes holes in visualization.
 * For smooth visualization, use `resizeMaskAsCanvas()` instead.
 * 
 * @param mask - Binary mask data
 * @param maskWidth - Current mask width
 * @param maskHeight - Current mask height
 * @param targetWidth - Target width
 * @param targetHeight - Target height
 * @returns Resized binary mask (0 or 255 values only)
 */
export function resizeMask(
    mask: Uint8Array,
    maskWidth: number,
    maskHeight: number,
    targetWidth: number,
    targetHeight: number
): Uint8Array {
    // Create canvas with mask data
    const srcCanvas = new OffscreenCanvas(maskWidth, maskHeight);
    const srcCtx = srcCanvas.getContext('2d')!;
    
    // Convert mask to ImageData (grayscale to RGBA)
    const srcImageData = srcCtx.createImageData(maskWidth, maskHeight);
    for (let i = 0; i < mask.length; i++) {
        const idx = i * 4;
        srcImageData.data[idx] = mask[i];
        srcImageData.data[idx + 1] = mask[i];
        srcImageData.data[idx + 2] = mask[i];
        srcImageData.data[idx + 3] = 255;
    }
    srcCtx.putImageData(srcImageData, 0, 0);

    // Resize
    const dstCanvas = new OffscreenCanvas(targetWidth, targetHeight);
    const dstCtx = dstCanvas.getContext('2d')!;
    dstCtx.drawImage(srcCanvas, 0, 0, targetWidth, targetHeight);

    // Extract resized mask
    const dstImageData = dstCtx.getImageData(0, 0, targetWidth, targetHeight);
    const result = new Uint8Array(targetWidth * targetHeight);
    for (let i = 0; i < result.length; i++) {
        // Re-threshold to binary for selection (causes hard edges)
        result[i] = dstImageData.data[i * 4] > 127 ? 255 : 0;
    }

    return result;
}

/**
 * Resize mask to target dimensions as a canvas for VISUALIZATION.
 * Uses bilinear interpolation WITHOUT re-thresholding, preserving smooth edges.
 * 
 * This matches the reference SAM2 implementation (geronimi73/next-sam) which
 * renders masks directly to canvas with smooth alpha blending.
 * 
 * The output canvas has:
 * - R,G,B: Overlay color (default: green #22c55e)
 * - Alpha: Smooth mask values from bilinear interpolation
 * 
 * @param mask - Binary mask data (0 or 255)
 * @param maskWidth - Current mask width  
 * @param maskHeight - Current mask height
 * @param targetWidth - Target width
 * @param targetHeight - Target height
 * @param overlayColor - RGB color for the mask overlay [R, G, B] (default: green)
 * @returns OffscreenCanvas with colored mask overlay and smooth alpha
 */
export function resizeMaskAsCanvas(
    mask: Uint8Array,
    maskWidth: number,
    maskHeight: number,
    targetWidth: number,
    targetHeight: number,
    overlayColor: [number, number, number] = [34, 197, 94]  // #22c55e green
): OffscreenCanvas {
    // Create source canvas with mask in alpha channel
    // We put the mask value in the alpha channel so bilinear interpolation
    // produces smooth alpha edges when scaled
    const srcCanvas = new OffscreenCanvas(maskWidth, maskHeight);
    const srcCtx = srcCanvas.getContext('2d')!;
    
    const srcImageData = srcCtx.createImageData(maskWidth, maskHeight);
    for (let i = 0; i < mask.length; i++) {
        const idx = i * 4;
        // Use the overlay color for RGB
        srcImageData.data[idx] = overlayColor[0];     // R
        srcImageData.data[idx + 1] = overlayColor[1]; // G
        srcImageData.data[idx + 2] = overlayColor[2]; // B
        // Put mask value in alpha - this is what gets interpolated smoothly
        srcImageData.data[idx + 3] = mask[i];
    }
    srcCtx.putImageData(srcImageData, 0, 0);

    // Resize with bilinear interpolation - alpha values will be smoothly interpolated
    const dstCanvas = new OffscreenCanvas(targetWidth, targetHeight);
    const dstCtx = dstCanvas.getContext('2d')!;
    // imageSmoothingEnabled defaults to true, giving us bilinear interpolation
    dstCtx.drawImage(srcCanvas, 0, 0, targetWidth, targetHeight);

    return dstCanvas;
}

/**
 * Resize mask to target dimensions preserving smooth alpha values.
 * Returns a Float32Array with values in range [0, 1] for visualization.
 * 
 * Unlike `resizeMask()` which re-thresholds to binary, this preserves
 * the smooth interpolated values from bilinear scaling.
 * 
 * @param mask - Binary mask data (0 or 255)
 * @param maskWidth - Current mask width
 * @param maskHeight - Current mask height
 * @param targetWidth - Target width
 * @param targetHeight - Target height
 * @returns Float32Array with smooth alpha values [0, 1]
 */
export function resizeMaskSmooth(
    mask: Uint8Array,
    maskWidth: number,
    maskHeight: number,
    targetWidth: number,
    targetHeight: number
): Float32Array {
    // Create canvas with mask in grayscale
    const srcCanvas = new OffscreenCanvas(maskWidth, maskHeight);
    const srcCtx = srcCanvas.getContext('2d')!;
    
    const srcImageData = srcCtx.createImageData(maskWidth, maskHeight);
    for (let i = 0; i < mask.length; i++) {
        const idx = i * 4;
        srcImageData.data[idx] = mask[i];
        srcImageData.data[idx + 1] = mask[i];
        srcImageData.data[idx + 2] = mask[i];
        srcImageData.data[idx + 3] = 255;
    }
    srcCtx.putImageData(srcImageData, 0, 0);

    // Resize with bilinear interpolation
    const dstCanvas = new OffscreenCanvas(targetWidth, targetHeight);
    const dstCtx = dstCanvas.getContext('2d')!;
    dstCtx.drawImage(srcCanvas, 0, 0, targetWidth, targetHeight);

    // Extract smooth values as Float32Array normalized to [0, 1]
    const dstImageData = dstCtx.getImageData(0, 0, targetWidth, targetHeight);
    const result = new Float32Array(targetWidth * targetHeight);
    for (let i = 0; i < result.length; i++) {
        // Keep smooth interpolated values (no re-thresholding!)
        result[i] = dstImageData.data[i * 4] / 255;
    }

    return result;
}

/**
 * Create input tensor for mask (for iterative refinement).
 * When no previous mask exists, creates zero tensor.
 * 
 * @param ort - ONNX Runtime module
 * @param maskInput - Previous mask or null
 * @returns ONNX tensor [1, 1, 256, 256]
 */
export function createMaskInputTensorWithOrt(
    ortModule: OrtModule,
    maskInput: Float32Array | null
): ort.Tensor {
    const size = 256 * 256;
    const data = maskInput ?? new Float32Array(size);
    return new ortModule.Tensor('float32', data, [1, 1, 256, 256]);
}

/**
 * Create "has mask input" flag tensor.
 * 
 * @param ortModule - ONNX Runtime module
 * @param hasMask - Whether a previous mask is provided
 * @returns ONNX tensor [1]
 */
export function createHasMaskTensorWithOrt(
    ortModule: OrtModule,
    hasMask: boolean
): ort.Tensor {
    return new ortModule.Tensor('float32', new Float32Array([hasMask ? 1 : 0]), [1]);
}
