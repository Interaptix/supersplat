/**
 * Debug visualization utilities for SAM2 inference.
 * TypeScript port from next-sam/lib/debug-utils.js
 */

/** SAM2 model input image size */
export const SAM2_INPUT_SIZE = 1024;

/** Debug mode flag - set to true to enable console visualizations */
export let DEBUG_SAM2 = true;

/** Point interface for click coordinates */
export interface SAM2Point {
    x: number;
    y: number;
    label: number;
}

/** Channel statistics interface */
interface ChannelStats {
    min: number;
    max: number;
    mean: number;
    std: number;
    sum: number;
    l2Norm: number;
    nanCount: number;
    infCount: number;
}

/** Letterbox info interface */
export interface LetterboxInfo {
    padX: number;
    padY: number;
    squareSize: number;
    scale: number;
}

/** Decoder output interface */
export interface DecoderOutput {
    masks?: { dims: number[]; data: Float32Array };
    low_res_masks?: { dims: number[]; data: Float32Array };
    iou_predictions?: { data: Float32Array };
    scores?: { data: Float32Array };
}

/** Extracted mask result */
export interface ExtractedMask {
    mask: Uint8Array;
    logits: Float32Array;
    width: number;
    height: number;
    iouScore: number | null;
}

/**
 * Compute per-channel statistics for Float32Array tensor data.
 * @param {Float32Array} data - The tensor data array
 * @param {number} offset - The offset into the data array
 * @param {number} length - The number of elements to process
 * @returns {ChannelStats} Statistics for the channel
 */
function computeChannelStats(data: Float32Array, offset: number, length: number): ChannelStats {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let sumSq = 0;
    let nanCount = 0;
    let infCount = 0;

    for (let i = 0; i < length; i++) {
        const v = data[offset + i];
        if (isNaN(v)) {
            nanCount++;
            continue;
        }
        if (!isFinite(v)) {
            infCount++;
            continue;
        }
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sumSq += v * v;
    }

    const validCount = length - nanCount - infCount;
    const mean = validCount > 0 ? sum / validCount : 0;
    const variance = validCount > 0 ? (sumSq / validCount) - (mean * mean) : 0;
    const std = Math.sqrt(Math.max(0, variance));
    const l2Norm = Math.sqrt(sumSq);

    return { min, max, mean, std, sum, l2Norm, nanCount, infCount };
}

/**
 * Log comprehensive image characteristics for SAM2 model input.
 * @param {Float32Array} tensorData - The tensor data array
 * @param {number} originalWidth - Original image width
 * @param {number} originalHeight - Original image height
 * @param {LetterboxInfo} letterboxInfo - Letterbox padding information
 */
export function logImageCharacteristics(
    tensorData: Float32Array,
    originalWidth: number,
    originalHeight: number,
    letterboxInfo: LetterboxInfo
): void {
    if (!DEBUG_SAM2) return;

    const numPixels = SAM2_INPUT_SIZE * SAM2_INPUT_SIZE;

    console.group('📊 SAM2 Model Input Visualization');
    console.log(`Original image: ${originalWidth}×${originalHeight}`);
    console.log(`Letterbox padding: (${letterboxInfo.padX}, ${letterboxInfo.padY})`);
    console.log(`Square size: ${letterboxInfo.squareSize}`);
    console.log(`Scale factor: ${letterboxInfo.scale.toFixed(4)}`);
    console.log(`Final input size: ${SAM2_INPUT_SIZE}×${SAM2_INPUT_SIZE}`);
    console.log('');
    console.log('Per-channel statistics:');

    const channels = ['R', 'G', 'B'];
    for (let c = 0; c < 3; c++) {
        const stats = computeChannelStats(tensorData, c * numPixels, numPixels);
        console.log(
            `  Channel ${channels[c]}: ` +
            `min=${stats.min.toFixed(4)}, max=${stats.max.toFixed(4)}, ` +
            `mean=${stats.mean.toFixed(4)}, std=${stats.std.toFixed(4)}, ` +
            `sum=${stats.sum.toExponential(2)}, L2=${stats.l2Norm.toExponential(2)}, ` +
            `NaN/Inf=${stats.nanCount + stats.infCount}`
        );
    }

    const overallStats = computeChannelStats(tensorData, 0, tensorData.length);
    console.log(
        `  Overall: min=${overallStats.min.toFixed(4)}, max=${overallStats.max.toFixed(4)}, ` +
        `mean=${overallStats.mean.toFixed(4)}, NaN/Inf=${overallStats.nanCount + overallStats.infCount}`
    );
    console.groupEnd();
}

/** Check if running in a Web Worker context */
const isWorker = typeof document === 'undefined' && typeof self !== 'undefined';

/**
 * Enable or disable debug visualizations.
 * @param {boolean} enabled - Whether to enable debug mode
 */
export function setDebugMode(enabled: boolean): void {
    DEBUG_SAM2 = enabled;
}

/**
 * Log an image to the Chrome DevTools console.
 * @param {string} dataUrl - The data URL of the image
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @param {string} label - Label to display with the image
 */
function logImageToConsole(dataUrl: string, width: number, height: number, label: string): void {
    const maxSize = 256;
    const scale = Math.min(1, maxSize / Math.max(width, height));
    const displayWidth = Math.round(width * scale);
    const displayHeight = Math.round(height * scale);

    console.log(
        `%c${label}`,
        'font-weight: bold; font-size: 14px; color: #22c55e;'
    );
    console.log(
        '%c ',
        `background-image: url(${dataUrl});
         background-size: ${displayWidth}px ${displayHeight}px;
         background-repeat: no-repeat;
         padding: ${displayHeight / 2}px ${displayWidth / 2}px;
         border: 1px solid #444;`
    );
}

/**
 * Send debug image to main thread (when running in worker).
 * @param {string} label - Label for the debug image
 * @param {string} dataUrl - The data URL of the image
 * @param {number} width - Image width
 * @param {number} height - Image height
 */
function postDebugImage(label: string, dataUrl: string, width: number, height: number): void {
    if (isWorker) {
        (self as unknown as Worker).postMessage({
            type: 'debugImage',
            data: { label, dataUrl, width, height }
        });
    }
}

/**
 * Convert a canvas to a data URL.
 * @param {HTMLCanvasElement | OffscreenCanvas} canvas - The canvas to convert
 * @returns {Promise<string>} The data URL
 */
async function canvasToDataUrl(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<string> {
    if (typeof HTMLCanvasElement !== 'undefined' && canvas instanceof HTMLCanvasElement) {
        return canvas.toDataURL('image/png');
    }
    const blob = await (canvas as OffscreenCanvas).convertToBlob({ type: 'image/png' });
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

/**
 * Create a canvas that works in both main thread and worker contexts.
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
 * @returns {HTMLCanvasElement | OffscreenCanvas} The created canvas
 */
function createCanvas(width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
    if (isWorker) {
        return new OffscreenCanvas(width, height);
    }
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;

}

/**
 * Output a debug image - works in both main thread and worker contexts.
 * @param {string} label - Label for the debug image
 * @param {string} dataUrl - The data URL of the image
 * @param {number} width - Image width
 * @param {number} height - Image height
 */
function outputDebugImage(label: string, dataUrl: string, width: number, height: number): void {
    if (isWorker) {
        postDebugImage(label, dataUrl, width, height);
    } else {
        logImageToConsole(dataUrl, width, height, label);
    }
}

/**
 * VISUALIZATION: Display the preprocessed model input in Chrome DevTools.
 * @param {HTMLCanvasElement | OffscreenCanvas} canvas - The canvas to visualize
 * @param {SAM2Point[]} [points] - Optional array of points to overlay
 * @returns {Promise<void>} Resolves when visualization is complete
 */
export async function visualizeModelInput(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    points?: SAM2Point[]
): Promise<void> {
    if (!DEBUG_SAM2) return;

    console.log('[SAM2 DEBUG] Model Input Visualization');
    console.log('Input size:', `${canvas.width}×${canvas.height}`);

    const vizCanvas = createCanvas(canvas.width, canvas.height);
    const ctx = vizCanvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    ctx.drawImage(canvas, 0, 0);

    if (points && points.length > 0) {
        console.log(`Points (in ${SAM2_INPUT_SIZE}×${SAM2_INPUT_SIZE} model space):`);
        const pointRadius = 8;
        for (const point of points) {
            const isForeground = point.label === 1;
            console.log(`  ${isForeground ? '🟢' : '🔴'} (${point.x.toFixed(1)}, ${point.y.toFixed(1)}) ${isForeground ? 'fg' : 'bg'}`);

            ctx.beginPath();
            ctx.arc(point.x, point.y, pointRadius, 0, 2 * Math.PI);
            ctx.fillStyle = isForeground ? 'rgba(0, 255, 0, 0.9)' : 'rgba(255, 0, 0, 0.9)';
            ctx.fill();

            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(point.x - pointRadius - 4, point.y);
            ctx.lineTo(point.x + pointRadius + 4, point.y);
            ctx.moveTo(point.x, point.y - pointRadius - 4);
            ctx.lineTo(point.x, point.y + pointRadius + 4);
            ctx.strokeStyle = isForeground ? 'lime' : 'red';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    const dataUrl = await canvasToDataUrl(vizCanvas);
    const pointsLabel = points && points.length > 0 ? ` + ${points.length} points` : '';
    outputDebugImage(`Model Input (${vizCanvas.width}×${vizCanvas.height})${pointsLabel}`, dataUrl, vizCanvas.width, vizCanvas.height);
}

/**
 * VISUALIZATION: Display the encoder embedding in Chrome DevTools.
 * @param {Float32Array} embedding - The embedding tensor data
 * @param {number[]} dims - The tensor dimensions [batch, channels, height, width]
 * @returns {Promise<void>} Resolves when visualization is complete
 */
export async function visualizeEncoderEmbedding(embedding: Float32Array, dims: number[]): Promise<void> {
    if (!DEBUG_SAM2) return;

    console.log('[SAM2 DEBUG] Encoder Embedding Visualization');

    const [, channels, height, width] = dims;
    const spatialSize = height * width;

    console.log('Embedding shape:', `[${dims.join(', ')}]`);
    console.log('Spatial resolution:', `${width}×${height}`);
    console.log('Feature channels:', channels);

    let min = Infinity, max = -Infinity, sum = 0;
    for (let i = 0; i < embedding.length; i++) {
        const v = embedding[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
    }
    const mean = sum / embedding.length;

    console.log('Statistics:', { min: min.toFixed(4), max: max.toFixed(4), mean: mean.toFixed(4) });

    const meanActivation = new Float32Array(spatialSize);
    for (let c = 0; c < channels; c++) {
        const channelOffset = c * spatialSize;
        for (let i = 0; i < spatialSize; i++) {
            meanActivation[i] += embedding[channelOffset + i];
        }
    }

    let actMin = Infinity, actMax = -Infinity;
    for (let i = 0; i < spatialSize; i++) {
        meanActivation[i] /= channels;
        if (meanActivation[i] < actMin) actMin = meanActivation[i];
        if (meanActivation[i] > actMax) actMax = meanActivation[i];
    }

    const actRange = actMax - actMin || 1;
    console.log('Mean activation range:', { min: actMin.toFixed(4), max: actMax.toFixed(4) });

    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < spatialSize; i++) {
        const normalized = ((meanActivation[i] - actMin) / actRange) * 255;
        const idx = i * 4;
        const t = normalized / 255;
        imageData.data[idx] = Math.round(68 + 187 * t);
        imageData.data[idx + 1] = Math.round(1 + 188 * t);
        imageData.data[idx + 2] = Math.round(84 - 84 * t);
        imageData.data[idx + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    const dataUrl = await canvasToDataUrl(canvas);
    outputDebugImage(`Encoder Embedding Mean Activation (${width}×${height})`, dataUrl, width, height);
}

/**
 * VISUALIZATION: Display the decoder output mask in Chrome DevTools.
 * @param {Uint8Array | Float32Array} mask - The mask data
 * @param {number} maskWidth - Mask width in pixels
 * @param {number} maskHeight - Mask height in pixels
 * @param {Float32Array} [logits] - Optional raw logit values
 * @param {SAM2Point[]} [points] - Optional array of points to overlay
 * @returns {Promise<void>} Resolves when visualization is complete
 */
export async function visualizeModelOutput(
    mask: Uint8Array | Float32Array,
    maskWidth: number,
    maskHeight: number,
    logits?: Float32Array,
    points?: SAM2Point[]
): Promise<void> {
    if (!DEBUG_SAM2) return;

    console.log('[SAM2 DEBUG] Model Output Visualization');
    console.log('Mask shape:', `${maskWidth}×${maskHeight}`);

    let foregroundCount = 0;
    for (let i = 0; i < mask.length; i++) {
        const isForeground = mask instanceof Float32Array ? mask[i] > 0 : mask[i] > 127;
        if (isForeground) foregroundCount++;
    }
    const coverage = (100 * foregroundCount / mask.length).toFixed(1);
    console.log('Foreground coverage:', `${foregroundCount}/${mask.length} (${coverage}%)`);

    if (logits) {
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
        console.log('Logits stats:', {
            min: min.toFixed(4),
            max: max.toFixed(4),
            mean: mean.toFixed(4),
            positive: `${positiveCount}/${logits.length}`
        });
    }

    const canvas = createCanvas(maskWidth, maskHeight);
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    const imageData = ctx.createImageData(maskWidth, maskHeight);

    for (let i = 0; i < mask.length; i++) {
        const idx = i * 4;
        const isForeground = mask instanceof Float32Array ? mask[i] > 0 : mask[i] > 127;
        if (isForeground) {
            imageData.data[idx] = 34;
            imageData.data[idx + 1] = 197;
            imageData.data[idx + 2] = 94;
        } else {
            imageData.data[idx] = 30;
            imageData.data[idx + 1] = 30;
            imageData.data[idx + 2] = 30;
        }
        imageData.data[idx + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    if (points && points.length > 0) {
        const scaleToMask = maskWidth / SAM2_INPUT_SIZE;
        console.log(`Points (scaled from ${SAM2_INPUT_SIZE} to ${maskWidth} space, factor=${scaleToMask}):`);

        const pointRadius = 4;
        for (const point of points) {
            const scaledX = point.x * scaleToMask;
            const scaledY = point.y * scaleToMask;
            const isForeground = point.label === 1;

            console.log(`  ${isForeground ? '🟢' : '🔴'} (${point.x.toFixed(1)}, ${point.y.toFixed(1)}) → (${scaledX.toFixed(1)}, ${scaledY.toFixed(1)}) ${isForeground ? 'fg' : 'bg'}`);

            ctx.beginPath();
            ctx.arc(scaledX, scaledY, pointRadius, 0, 2 * Math.PI);
            ctx.fillStyle = isForeground ? 'rgba(0, 255, 0, 0.9)' : 'rgba(255, 0, 0, 0.9)';
            ctx.fill();

            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(scaledX - pointRadius - 2, scaledY);
            ctx.lineTo(scaledX + pointRadius + 2, scaledY);
            ctx.moveTo(scaledX, scaledY - pointRadius - 2);
            ctx.lineTo(scaledX, scaledY + pointRadius + 2);
            ctx.strokeStyle = isForeground ? 'lime' : 'red';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    const dataUrl = await canvasToDataUrl(canvas);
    const pointsLabel = points && points.length > 0 ? ` + ${points.length} points` : '';
    outputDebugImage(`Decoder Output Mask (${maskWidth}×${maskHeight})${pointsLabel}`, dataUrl, maskWidth, maskHeight);
}

/**
 * Extract best mask from multi-mask decoder output.
 * @param {DecoderOutput} decoderOutput - The decoder output containing masks and scores
 * @param {number} [maskIndex] - Index of the mask to extract (default: 0)
 * @returns {ExtractedMask} The extracted mask with logits and metadata
 */
export function extractBestMask(decoderOutput: DecoderOutput, maskIndex = 0): ExtractedMask {
    const masks = decoderOutput.masks || decoderOutput.low_res_masks;
    const scores = decoderOutput.iou_predictions || decoderOutput.scores;

    if (!masks) {
        throw new Error('No masks found in decoder output');
    }

    const dims = masks.dims;
    const numMasks = dims[1];
    const height = dims[2];
    const width = dims[3];
    const maskSize = height * width;

    console.log(`[SAM2 DEBUG] extractBestMask: dims=[${dims.join(', ')}], numMasks=${numMasks}, selecting mask ${maskIndex}`);

    if (maskIndex < 0 || maskIndex >= numMasks) {
        console.warn(`[SAM2] Invalid maskIndex ${maskIndex}, clamping to valid range [0, ${numMasks - 1}]`);
        maskIndex = Math.max(0, Math.min(maskIndex, numMasks - 1));
    }

    const data = masks.data;
    const mask = new Uint8Array(maskSize);
    const logits = new Float32Array(maskSize);

    const maskOffset = maskIndex * maskSize;

    for (let i = 0; i < maskSize; i++) {
        logits[i] = data[maskOffset + i];
        mask[i] = data[maskOffset + i] > 0 ? 255 : 0;
    }

    const iouScore = scores ? scores.data[maskIndex] : null;

    return { mask, logits, width, height, iouScore };
}

/**
 * Get the raw logits from a specific mask index without thresholding.
 * @param {DecoderOutput} decoderOutput - The decoder output containing masks
 * @param {number} [maskIndex] - Index of the mask to extract logits from (default: 0)
 * @returns {Float32Array} The raw logit values for the specified mask
 */
export function getMaskLogits(decoderOutput: DecoderOutput, maskIndex = 0): Float32Array {
    const masks = decoderOutput.masks || decoderOutput.low_res_masks;

    if (!masks) {
        throw new Error('No masks found in decoder output');
    }

    const dims = masks.dims;
    const height = dims[2];
    const width = dims[3];
    const maskSize = height * width;

    const data = masks.data;
    const logits = new Float32Array(maskSize);
    const maskOffset = maskIndex * maskSize;

    for (let i = 0; i < maskSize; i++) {
        logits[i] = data[maskOffset + i];
    }

    return logits;
}

/**
 * VISUALIZATION: Display the decoder INPUT (mask_input + points) in Chrome DevTools.
 * @param {{ data: Float32Array } | Float32Array | null} maskInputTensor - The mask input tensor or null
 * @param {boolean} hasMask - Whether a previous mask exists
 * @param {SAM2Point[]} [points] - Optional array of points to overlay
 * @returns {Promise<void>} Resolves when visualization is complete
 */
export async function visualizeDecoderInput(
    maskInputTensor: { data: Float32Array } | Float32Array | null,
    hasMask: boolean,
    points?: SAM2Point[]
): Promise<void> {
    if (!DEBUG_SAM2) return;

    console.log('[SAM2 DEBUG] ═══════════════════════════════════════════════════');
    console.log('[SAM2 DEBUG] DECODER INPUT Visualization');
    console.log('[SAM2 DEBUG] ═══════════════════════════════════════════════════');

    if (points && points.length > 0) {
        console.log(`[SAM2 DEBUG] Point prompts (${points.length}):`);
        for (const point of points) {
            const isForeground = point.label === 1;
            console.log(`  ${isForeground ? '🟢 FG' : '🔴 BG'} (${point.x.toFixed(1)}, ${point.y.toFixed(1)})`);
        }
    } else {
        console.log('[SAM2 DEBUG] No point prompts');
    }

    console.log(`[SAM2 DEBUG] has_mask_input: ${hasMask}`);

    if (!hasMask || !maskInputTensor) {
        console.log('[SAM2 DEBUG] No previous mask (first click or dummy mask)');
        return;
    }

    const maskData = (maskInputTensor as { data: Float32Array }).data || (maskInputTensor as Float32Array);
    const maskWidth = 256;
    const maskHeight = 256;
    const maskSize = maskWidth * maskHeight;

    console.log(`[SAM2 DEBUG] mask_input shape: [1, 1, ${maskWidth}, ${maskHeight}]`);

    let min = Infinity, max = -Infinity, sum = 0;
    let positiveCount = 0;
    for (let i = 0; i < maskSize; i++) {
        const v = maskData[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v > 0) positiveCount++;
    }
    const coverage = (100 * positiveCount / maskSize).toFixed(1);

    console.log('[SAM2 DEBUG] mask_input logits stats:', {
        min: min.toFixed(4),
        max: max.toFixed(4),
        mean: (sum / maskSize).toFixed(4),
        positive: `${positiveCount}/${maskSize} (${coverage}%)`
    });

    const canvas = createCanvas(maskWidth, maskHeight);
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    const imageData = ctx.createImageData(maskWidth, maskHeight);

    for (let i = 0; i < maskSize; i++) {
        const idx = i * 4;
        const isForeground = maskData[i] > 0;
        if (isForeground) {
            imageData.data[idx] = 59;
            imageData.data[idx + 1] = 130;
            imageData.data[idx + 2] = 246;
        } else {
            imageData.data[idx] = 30;
            imageData.data[idx + 1] = 30;
            imageData.data[idx + 2] = 30;
        }
        imageData.data[idx + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    if (points && points.length > 0) {
        const scaleToMask = maskWidth / SAM2_INPUT_SIZE;
        const pointRadius = 4;

        for (const point of points) {
            const scaledX = point.x * scaleToMask;
            const scaledY = point.y * scaleToMask;
            const isForeground = point.label === 1;

            ctx.beginPath();
            ctx.arc(scaledX, scaledY, pointRadius, 0, 2 * Math.PI);
            ctx.fillStyle = isForeground ? 'rgba(0, 255, 0, 0.9)' : 'rgba(255, 0, 0, 0.9)';
            ctx.fill();

            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(scaledX - pointRadius - 2, scaledY);
            ctx.lineTo(scaledX + pointRadius + 2, scaledY);
            ctx.moveTo(scaledX, scaledY - pointRadius - 2);
            ctx.lineTo(scaledX, scaledY + pointRadius + 2);
            ctx.strokeStyle = isForeground ? 'lime' : 'red';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    const dataUrl = await canvasToDataUrl(canvas);
    const pointsLabel = points && points.length > 0 ? ` + ${points.length} points` : '';
    outputDebugImage(`DECODER INPUT: Previous Mask (${maskWidth}×${maskHeight})${pointsLabel}`, dataUrl, maskWidth, maskHeight);
}

/**
 * Handle debug image message in main thread.
 * @param {object} data - Debug image data from worker
 * @param {string} data.label - Label for the debug image
 * @param {string} data.dataUrl - The data URL of the image
 * @param {number} data.width - Image width
 * @param {number} data.height - Image height
 */
export function handleDebugImageMessage(data: { label: string; dataUrl: string; width: number; height: number }): void {
    const { label, dataUrl, width, height } = data;
    logImageToConsole(dataUrl, width, height, label);
}
