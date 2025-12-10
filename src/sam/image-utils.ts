/**
 * Image utility functions for SAM2 inference.
 * TypeScript port from next-sam/lib/imageutils.js
 */

import { Tensor } from 'onnxruntime-web';

import { debugLog } from './debug-utils';

/**
 * Dilate a binary mask to expand its regions.
 * Uses a box kernel for efficient morphological dilation.
 * This expands masked regions by the kernel radius in all directions.
 *
 * @param {HTMLCanvasElement} canvas - The mask canvas to dilate (uses alpha channel)
 * @param {number} kernelSize - The dilation kernel size in pixels (will be made odd if even)
 * @returns {HTMLCanvasElement} A new canvas with the dilated mask
 */
export function dilateMask(canvas: HTMLCanvasElement, kernelSize: number): HTMLCanvasElement {
    if (kernelSize <= 0) {
        // No dilation needed, return a copy
        const result = document.createElement('canvas');
        result.width = canvas.width;
        result.height = canvas.height;
        const ctx = result.getContext('2d')!;
        ctx.drawImage(canvas, 0, 0);
        return result;
    }

    // Ensure kernel size is odd for symmetric dilation
    kernelSize = Math.max(1, Math.floor(kernelSize));
    if (kernelSize % 2 === 0) {
        kernelSize += 1;
    }
    const radius = Math.floor(kernelSize / 2);

    const width = canvas.width;
    const height = canvas.height;

    // Get source image data
    const ctx = canvas.getContext('2d')!;
    const srcData = ctx.getImageData(0, 0, width, height);
    const src = srcData.data;

    // Create output canvas
    const result = document.createElement('canvas');
    result.width = width;
    result.height = height;
    const resultCtx = result.getContext('2d')!;
    const dstData = resultCtx.createImageData(width, height);
    const dst = dstData.data;

    // Extract alpha channel into binary array (255 = masked, 0 = not masked)
    const srcAlpha = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
        srcAlpha[i] = src[i * 4 + 3] > 0 ? 255 : 0;
    }

    // Efficient separable dilation using two passes (horizontal + vertical)
    // This reduces complexity from O(n * k^2) to O(n * k)

    // Intermediate buffer for horizontal pass
    const temp = new Uint8Array(width * height);

    // Pass 1: Horizontal dilation
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxVal = 0;
            for (let kx = -radius; kx <= radius; kx++) {
                const nx = x + kx;
                if (nx >= 0 && nx < width) {
                    if (srcAlpha[y * width + nx] > maxVal) {
                        maxVal = srcAlpha[y * width + nx];
                    }
                }
            }
            temp[y * width + x] = maxVal;
        }
    }

    // Pass 2: Vertical dilation on the horizontally dilated result
    const dilatedAlpha = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxVal = 0;
            for (let ky = -radius; ky <= radius; ky++) {
                const ny = y + ky;
                if (ny >= 0 && ny < height) {
                    if (temp[ny * width + x] > maxVal) {
                        maxVal = temp[ny * width + x];
                    }
                }
            }
            dilatedAlpha[y * width + x] = maxVal;
        }
    }

    // Write output: copy RGB from source, use dilated alpha
    // For dilated pixels, use the mask color (green) consistent with float32ArrayToCanvas
    for (let i = 0; i < width * height; i++) {
        const srcIdx = i * 4;
        const alpha = dilatedAlpha[i];

        if (alpha > 0) {
            // Use the same green mask color as float32ArrayToCanvas
            dst[srcIdx] = 0x32;     // R
            dst[srcIdx + 1] = 0xcd; // G
            dst[srcIdx + 2] = 0x32; // B
            dst[srcIdx + 3] = 255;  // A
        } else {
            dst[srcIdx] = 0;
            dst[srcIdx + 1] = 0;
            dst[srcIdx + 2] = 0;
            dst[srcIdx + 3] = 0;
        }
    }

    resultCtx.putImageData(dstData, 0, 0);
    return result;
}

/**
 * Mask an image canvas with a mask canvas.
 * Applies the mask as an alpha channel to the image.
 * @param {HTMLCanvasElement} imageCanvas - The source image canvas
 * @param {HTMLCanvasElement} maskCanvas - The mask canvas to apply
 * @returns {HTMLCanvasElement} A new canvas with the mask applied as alpha
 */
export function maskImageCanvas(imageCanvas: HTMLCanvasElement, maskCanvas: HTMLCanvasElement): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.height = imageCanvas.height;
    canvas.width = imageCanvas.width;

    context.drawImage(
        maskCanvas,
        0,
        0,
        maskCanvas.width,
        maskCanvas.height,
        0,
        0,
        canvas.width,
        canvas.height
    );
    context.globalCompositeOperation = 'source-in';
    context.drawImage(
        imageCanvas,
        0,
        0,
        imageCanvas.width,
        imageCanvas.height,
        0,
        0,
        canvas.width,
        canvas.height
    );

    return canvas;
}

/** Size interface */
export interface Size {
    w: number;
    h: number;
}

/** Box interface */
export interface Box {
    x: number;
    y: number;
    w: number;
    h: number;
}

/**
 * Resize a canvas to a new size.
 * @param {HTMLCanvasElement} canvasOrig - The original canvas to resize
 * @param {Size} size - The target size with w and h properties
 * @returns {HTMLCanvasElement} A new canvas at the specified size
 */
export function resizeCanvas(canvasOrig: HTMLCanvasElement, size: Size): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    canvas.height = size.h;
    canvas.width = size.w;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'low';
    debugLog(`[SAM2] resizeCanvas smoothing: ${ctx.imageSmoothingEnabled}, quality: ${ctx.imageSmoothingQuality}`);

    ctx.drawImage(
        canvasOrig,
        0,
        0,
        canvasOrig.width,
        canvasOrig.height,
        0,
        0,
        canvas.width,
        canvas.height
    );

    return canvas;
}

/**
 * Merge two mask canvases together.
 * Draws source mask onto target mask.
 * @param {HTMLCanvasElement} sourceMask - The source mask canvas to merge from
 * @param {HTMLCanvasElement} targetMask - The target mask canvas to merge onto
 * @returns {HTMLCanvasElement} A new canvas with both masks combined
 */
export function mergeMasks(sourceMask: HTMLCanvasElement, targetMask: HTMLCanvasElement): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    canvas.height = targetMask.height;
    canvas.width = targetMask.width;

    ctx.drawImage(targetMask, 0, 0);
    ctx.drawImage(
        sourceMask,
        0,
        0,
        sourceMask.width,
        sourceMask.height,
        0,
        0,
        targetMask.width,
        targetMask.height
    );

    return canvas;
}

/**
 * Calculate box dimensions to fit source into target preserving aspect ratio.
 * Returns the position and size for letterbox/pillarbox padding.
 * @param {Size} sourceDim - The source dimensions with w and h properties
 * @param {Size} targetDim - The target dimensions with w and h properties
 * @returns {Box} The calculated box with position and size for padding
 */
export function resizeAndPadBox(sourceDim: Size, targetDim: Size): Box {
    if (sourceDim.h === sourceDim.w) {
        return { x: 0, y: 0, w: targetDim.w, h: targetDim.h };
    } else if (sourceDim.h > sourceDim.w) {
        // portrait => resize and pad left
        const newW = (sourceDim.w / sourceDim.h) * targetDim.w;
        const padLeft = Math.floor((targetDim.w - newW) / 2);
        return { x: padLeft, y: 0, w: newW, h: targetDim.h };
    }
    // landscape => resize and pad top
    const newH = (sourceDim.h / sourceDim.w) * targetDim.h;
    const padTop = Math.floor((targetDim.h - newH) / 2);
    return { x: 0, y: padTop, w: targetDim.w, h: newH };

}

/**
 * Slice a tensor to extract a specific mask.
 * Input: onnx Tensor [B, *, W, H] and index idx
 * Output: Float32Array for mask at index idx
 * @param {Tensor} tensor - The ONNX tensor to slice from
 * @param {number} idx - The index of the mask to extract
 * @returns {Float32Array} The extracted mask data
 */
export function sliceTensor(tensor: Tensor, idx: number): Float32Array {
    const [, , width, height] = tensor.dims as number[];
    const stride = width * height;
    const start = stride * idx;
    const end = start + stride;

    // Use tensor.data which is the typed array data
    const data = tensor.data as Float32Array;
    return data.slice(start, end);
}

/**
 * Convert Float32Array mask to HTMLCanvasElement.
 * Input: Float32Array representing ORT.Tensor of shape [1, 1, width, height]
 * Output: HTMLCanvasElement (4 channels, RGBA)
 * @param {Float32Array} array - The mask data as Float32Array
 * @param {number} width - The width of the output canvas
 * @param {number} height - The height of the output canvas
 * @returns {HTMLCanvasElement} A canvas with the mask rendered as RGBA
 */
export function float32ArrayToCanvas(array: Float32Array, width: number, height: number): HTMLCanvasElement {
    const C = 4; // 4 output channels, RGBA
    const imageData = new Uint8ClampedArray(array.length * C);

    for (let srcIdx = 0; srcIdx < array.length; srcIdx++) {
        const trgIdx = srcIdx * C;
        const maskedPx = array[srcIdx] > 0;
        imageData[trgIdx] = maskedPx ? 0x32 : 0;
        imageData[trgIdx + 1] = maskedPx ? 0xcd : 0;
        imageData[trgIdx + 2] = maskedPx ? 0x32 : 0;
        imageData[trgIdx + 3] = maskedPx ? 255 : 0; // alpha
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    canvas.height = height;
    canvas.width = width;
    ctx.putImageData(new ImageData(imageData, width, height), 0, 0);

    return canvas;
}

/** Canvas tensor data interface */
export interface CanvasTensorData {
    float32Array: Float32Array;
    shape: [number, number, number, number];
}

/**
 * Convert HTMLCanvasElement to Float32Array for ONNX tensor.
 * Input: HTMLCanvasElement (RGB)
 * Output: Float32Array for ORT.Tensor of shape [1, 3, canvas.width, canvas.height]
 * @param {HTMLCanvasElement} canvas - The canvas to convert
 * @returns {CanvasTensorData} Object containing float32Array and shape
 */
export function canvasToFloat32Array(canvas: HTMLCanvasElement): CanvasTensorData {
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const shape: [number, number, number, number] = [1, 3, canvas.width, canvas.height];

    const redArray: number[] = [];
    const greenArray: number[] = [];
    const blueArray: number[] = [];

    for (let i = 0; i < imageData.length; i += 4) {
        redArray.push(imageData[i]);
        greenArray.push(imageData[i + 1]);
        blueArray.push(imageData[i + 2]);
        // skip data[i + 3] to filter out the alpha channel
    }

    const transposedData = redArray.concat(greenArray).concat(blueArray);

    const float32Array = new Float32Array(shape[1] * shape[2] * shape[3]);
    for (let i = 0; i < transposedData.length; i++) {
        float32Array[i] = transposedData[i] / 255.0; // convert to float
    }

    return { float32Array, shape };
}

/**
 * Convert mask canvas to Float32Array.
 * Input: HTMLCanvasElement (RGB mask)
 * Output: Float32Array for ORT.Tensor of shape [1, 1, canvas.width, canvas.height]
 * @param {HTMLCanvasElement} canvas - The mask canvas to convert
 * @returns {Float32Array} The mask data as Float32Array
 */
export function maskCanvasToFloat32Array(canvas: HTMLCanvasElement): Float32Array {
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    const shape = [1, 1, canvas.width, canvas.height];
    const float32Array = new Float32Array(shape[1] * shape[2] * shape[3]);

    for (let i = 0; i < float32Array.length; i++) {
        const pixelIdx = i * 4;
        float32Array[i] = (imageData[pixelIdx] + imageData[pixelIdx + 1] + imageData[pixelIdx + 2]) / (3 * 255.0);
    }

    return float32Array;
}

/**
 * Create a canvas from an image URL.
 * Returns a promise that resolves with the canvas.
 * @param {string} url - The URL of the image to load
 * @returns {Promise<HTMLCanvasElement>} A promise that resolves with the canvas
 */
export function imageUrlToCanvas(url: string): Promise<HTMLCanvasElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d')!;
            ctx.drawImage(img, 0, 0);
            resolve(canvas);
        };
        img.onerror = reject;
        img.src = url;
    });
}

/**
 * Create a canvas from an HTMLImageElement.
 * @param {HTMLImageElement} img - The image element to convert
 * @returns {HTMLCanvasElement} A canvas containing the image
 */
export function imageToCanvas(img: HTMLImageElement): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    return canvas;
}

/**
 * Crop a canvas to a bounding box with the mask applied.
 * @param {HTMLCanvasElement} imageCanvas - The source image canvas
 * @param {HTMLCanvasElement} maskCanvas - The mask canvas defining the crop area
 * @returns {{ canvas: HTMLCanvasElement; bounds: Box } | null} The cropped canvas and bounds, or null if no mask
 */
export function cropCanvasWithMask(
    imageCanvas: HTMLCanvasElement,
    maskCanvas: HTMLCanvasElement
): { canvas: HTMLCanvasElement; bounds: Box } | null {
    // First, find the bounding box of the mask
    const maskCtx = maskCanvas.getContext('2d')!;
    const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;

    let minX = maskCanvas.width;
    let minY = maskCanvas.height;
    let maxX = 0;
    let maxY = 0;
    let hasMask = false;

    for (let y = 0; y < maskCanvas.height; y++) {
        for (let x = 0; x < maskCanvas.width; x++) {
            const idx = (y * maskCanvas.width + x) * 4;
            // Check alpha channel for mask presence
            if (maskData[idx + 3] > 0) {
                hasMask = true;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (!hasMask) {
        return null;
    }

    // Add some padding
    const padding = 2;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(maskCanvas.width - 1, maxX + padding);
    maxY = Math.min(maskCanvas.height - 1, maxY + padding);

    const cropWidth = maxX - minX + 1;
    const cropHeight = maxY - minY + 1;

    // Scale crop coordinates to image canvas
    const scaleX = imageCanvas.width / maskCanvas.width;
    const scaleY = imageCanvas.height / maskCanvas.height;

    const imgMinX = Math.floor(minX * scaleX);
    const imgMinY = Math.floor(minY * scaleY);
    const imgCropWidth = Math.ceil(cropWidth * scaleX);
    const imgCropHeight = Math.ceil(cropHeight * scaleY);

    // Create masked image first
    const maskedCanvas = maskImageCanvas(imageCanvas, maskCanvas);

    // Create cropped canvas
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = imgCropWidth;
    croppedCanvas.height = imgCropHeight;
    const croppedCtx = croppedCanvas.getContext('2d')!;

    croppedCtx.drawImage(
        maskedCanvas,
        imgMinX,
        imgMinY,
        imgCropWidth,
        imgCropHeight,
        0,
        0,
        imgCropWidth,
        imgCropHeight
    );

    return {
        canvas: croppedCanvas,
        bounds: {
            x: imgMinX,
            y: imgMinY,
            w: imgCropWidth,
            h: imgCropHeight
        }
    };
}
