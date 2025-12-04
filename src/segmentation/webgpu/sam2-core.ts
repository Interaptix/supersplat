/**
 * SAM2 core inference engine using ONNX Runtime Web with WebGPU.
 */

// Direct import from npm package - Rollup will bundle this
import * as ort from 'onnxruntime-web';
import {
    preprocessImageWithOrt,
    scalePoint,
    createPointCoordsTensorWithOrt,
    createPointLabelsTensorWithOrt,
    createMaskInputTensorWithOrt,
    createHasMaskTensorWithOrt,
    processMaskLogits,
    resizeMask,
    SAM2_INPUT_SIZE,
    debugSaveEncoderInput,
    debugSaveDecoderOutput,
    debugSaveFinalMask,
    debugSavePointsOverlay,
    debugSaveAllMasks,
    debugSaveMaskOverlay
} from './image-utils';

// ONNX Runtime Web module type
type OrtModule = typeof ort;

/**
 * Get ONNX Runtime Web module.
 * Since we now use direct import, this simply returns the imported module.
 */
export async function loadOnnxRuntime(): Promise<OrtModule> {
    console.log('[SAM2] Using bundled ONNX Runtime Web');
    
    // Configure ONNX Runtime to use WebGPU WASM files from CDN
    // This is needed because the WASM files are large and not bundled
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';
    
    return ort;
}

/** Execution provider type */
export type ExecutionProvider = 'webgpu' | 'wasm';

/** SAM2 initialization options */
export interface SAM2Options {
    /** Preferred execution provider */
    preferredProvider?: ExecutionProvider;
    /** Enable verbose logging */
    verbose?: boolean;
}

/** Individual mask candidate from SAM2 decoder */
export interface SAM2MaskCandidate {
    /** Mask index (0=tight, 1=medium, 2=broad) */
    index: number;
    /** IoU confidence score */
    iouScore: number;
    /** Binary mask (0 or 255) at original resolution */
    mask: Uint8Array;
    /** Mask width in pixels */
    width: number;
    /** Mask height in pixels */
    height: number;
    /** Raw logits (256x256) for this mask */
    logits: Float32Array;
}

/** SAM2 inference result */
export interface SAM2Result {
    /** Binary mask (0 or 255) - the selected mask */
    mask: Uint8Array;
    /** Mask width */
    width: number;
    /** Mask height */
    height: number;
    /** Raw logits for client-side thresholding (selected mask) */
    logits?: Float32Array;
    /** All mask candidates with IoU scores */
    allMasks?: SAM2MaskCandidate[];
    /** Index of the selected mask */
    selectedMaskIndex?: number;
    /** Inference timing in milliseconds */
    timing: {
        encode?: number;
        decode: number;
        total: number;
    };
}

/** Point input for segmentation */
export interface SAM2Point {
    x: number;
    y: number;
    type: 'fg' | 'bg';
}

/**
 * SAM2 inference engine.
 * Manages encoder/decoder sessions and cached embeddings.
 */
export class SAM2Core {
    private ort: OrtModule | null = null;
    private encoderSession: ort.InferenceSession | null = null;
    private decoderSession: ort.InferenceSession | null = null;
    private imageEmbeddings: Map<string, ort.Tensor> = new Map();
    private highResFeatures: Map<string, ort.Tensor[]> = new Map();
    private cachedImageData: Map<string, { data: Uint8Array; width: number; height: number }> = new Map();
    private currentProvider: ExecutionProvider = 'wasm';
    private verbose: boolean = false;

    /**
     * Initialize SAM2 with model data.
     * 
     * @param encoderData - Encoder model ArrayBuffer
     * @param decoderData - Decoder model ArrayBuffer
     * @param options - Initialization options
     */
    async initialize(
        encoderData: ArrayBuffer,
        decoderData: ArrayBuffer,
        options: SAM2Options = {}
    ): Promise<ExecutionProvider> {
        this.verbose = options.verbose ?? false;
        this.verbose = true;

        // Load ONNX Runtime first
        this.ort = await loadOnnxRuntime();

        // Try to create sessions with preferred provider
        const provider = await this.createSessions(
            encoderData,
            decoderData,
            options.preferredProvider ?? 'webgpu'
        );

        this.currentProvider = provider;
        return provider;
    }

    /**
     * Create ONNX sessions with fallback.
     * Due to ORT bug, we must try providers individually rather than as an array.
     */
    private async createSessions(
        encoderData: ArrayBuffer,
        decoderData: ArrayBuffer,
        preferredProvider: ExecutionProvider
    ): Promise<ExecutionProvider> {
        if (!this.ort) {
            throw new Error('ONNX Runtime not loaded');
        }

        const providers: ExecutionProvider[] = preferredProvider === 'webgpu'
            ? ['webgpu', 'wasm']
            : ['wasm'];

        for (const provider of providers) {
            try {
                this.log(`Attempting to create sessions with ${provider} provider...`);

                const sessionOptions: ort.InferenceSession.SessionOptions = {
                    executionProviders: [provider]
                };

                // Create encoder session
                this.encoderSession = await this.ort.InferenceSession.create(
                    encoderData,
                    sessionOptions
                );

                // Create decoder session
                this.decoderSession = await this.ort.InferenceSession.create(
                    decoderData,
                    sessionOptions
                );

                this.log(`Successfully created sessions with ${provider}`);
                return provider;
            } catch (error) {
                this.log(`Failed with ${provider}: ${error instanceof Error ? error.message : String(error)}`);
                // Clean up partial initialization
                await this.disposeSessions();
            }
        }

        throw new Error('Failed to create ONNX sessions with any available provider');
    }

    /**
     * Encode an image and cache the embeddings.
     * 
     * @param imageId - Unique ID for caching
     * @param imageData - RGBA pixel data
     * @param width - Image width
     * @param height - Image height
     * @returns Encoding time in milliseconds
     */
    async encodeImage(
        imageId: string,
        imageData: Uint8Array,
        width: number,
        height: number
    ): Promise<number> {
        if (!this.encoderSession || !this.ort) {
            throw new Error('SAM2 not initialized');
        }

        // Check cache
        if (this.imageEmbeddings.has(imageId)) {
            this.log(`Using cached embeddings for ${imageId}`);
            return 0;
        }

        const startTime = performance.now();

        // DEBUG: Save encoder input image
        await debugSaveEncoderInput(imageData, width, height);

        // Preprocess image
        const inputTensor = preprocessImageWithOrt(this.ort, imageData, width, height);

        // Run encoder
        const feeds: Record<string, ort.Tensor> = {
            image: inputTensor
        };

        const results = await this.encoderSession.run(feeds);

        // Cache image data for debug overlay (needed in decode for points visualization)
        this.cachedImageData.set(imageId, { data: imageData, width, height });

        // Cache embeddings
        // The encoder outputs image_embed and high_res_feats_0, high_res_feats_1
        if (results.image_embed) {
            this.imageEmbeddings.set(imageId, results.image_embed);
        }

        // Cache high resolution features if present
        const highResFeats: ort.Tensor[] = [];
        if (results.high_res_feats_0) {
            highResFeats.push(results.high_res_feats_0);
        }
        if (results.high_res_feats_1) {
            highResFeats.push(results.high_res_feats_1);
        }
        if (highResFeats.length > 0) {
            this.highResFeatures.set(imageId, highResFeats);
        }

        const encodeTime = performance.now() - startTime;
        this.log(`Encoded image ${imageId} in ${encodeTime.toFixed(0)}ms`);

        return encodeTime;
    }

    /**
     * Decode/segment with point prompts.
     * 
     * @param imageId - ID of previously encoded image
     * @param points - Point prompts
     * @param originalWidth - Original image width (for output scaling)
     * @param originalHeight - Original image height (for output scaling)
     * @param previousMask - Optional previous mask for refinement
     * @returns Segmentation result
     */
    async decode(
        imageId: string,
        points: SAM2Point[],
        originalWidth: number,
        originalHeight: number,
        previousMask?: Float32Array
    ): Promise<SAM2Result> {
        if (!this.decoderSession || !this.ort) {
            throw new Error('SAM2 not initialized');
        }

        const embeddings = this.imageEmbeddings.get(imageId);
        if (!embeddings) {
            throw new Error(`No embeddings found for image ${imageId}. Call encodeImage first.`);
        }

        const startTime = performance.now();

        // Scale points to SAM2 coordinate space
        const scaledPoints = points.map(p => ({
            ...scalePoint(p.x, p.y, originalWidth, originalHeight),
            type: p.type
        }));

        // Create input tensors
        const pointCoords = createPointCoordsTensorWithOrt(this.ort, scaledPoints);
        const pointLabels = createPointLabelsTensorWithOrt(this.ort, scaledPoints);
        const maskInput = createMaskInputTensorWithOrt(this.ort, previousMask ?? null);
        const hasMaskInput = createHasMaskTensorWithOrt(this.ort, previousMask !== undefined);

        // NOTE: multimask_output is baked into the ONNX model at export time, NOT a runtime input.
        // The HuggingFace sam2-tiny decoder uses multimask_output=True (always returns 3 masks).
        // We select the best mask using IoU scores.
        const hasPreviousMask = previousMask !== undefined && previousMask !== null;
        this.log(`Points: ${points.length}, has_mask_input: ${hasPreviousMask}`);

        // Build feeds - DO NOT include multimask_output (causes "invalid input" error)
        const feeds: Record<string, ort.Tensor> = {
            image_embed: embeddings,
            point_coords: pointCoords,
            point_labels: pointLabels,
            mask_input: maskInput,
            has_mask_input: hasMaskInput
        };

        // Add high-res features if available
        const highResFeats = this.highResFeatures.get(imageId);
        if (highResFeats) {
            if (highResFeats[0]) feeds.high_res_feats_0 = highResFeats[0];
            if (highResFeats[1]) feeds.high_res_feats_1 = highResFeats[1];
        }

        console.log('[SAM2] Decoder feeds:', Object.keys(feeds));
        console.log('[SAM2] Embeddings dims:', embeddings.dims);
        console.log('[SAM2] Point coords dims:', pointCoords.dims);
        console.log('[SAM2] Point labels dims:', pointLabels.dims);
        console.log('[SAM2] Mask input dims:', maskInput.dims);
        console.log('[SAM2] Has mask input dims:', hasMaskInput.dims);
        console.log('[SAM2] High-res feats:', highResFeats ? highResFeats.map(t => t.dims) : 'none');
        console.log('[SAM2] Running decoder...');

        console.log('[SAM2]:' , feeds)
        
        // Run decoder
        const results = await this.decoderSession.run(feeds);

        const decodeTime = performance.now() - startTime;

        // Process output mask
        // Output is typically 'masks' or 'low_res_masks'
        const maskTensor = results.masks ?? results.low_res_masks;
        if (!maskTensor) {
            throw new Error('No mask output from decoder');
        }

        // Log available outputs for debugging
        this.log(`Decoder outputs: ${Object.keys(results).join(', ')}`);
        this.log(`Mask tensor dims: [${maskTensor.dims.join(', ')}]`);

        // Select the best mask using IoU predictions
        // The ONNX model has multimask_output=True baked in, so it always returns 3 masks
        // with different granularity levels (0=tight, 1=medium, 2=broad).
        // We select the mask with the highest IoU (confidence) score.
        let bestMaskIndex = 0;
        const iouTensor = results.iou_predictions ?? results.iou_pred;
        if (iouTensor) {
            const iouData = iouTensor.data as Float32Array;
            this.log(`IoU predictions: [${Array.from(iouData).map(v => v.toFixed(4)).join(', ')}]`);
            
            // Always select the mask with highest IoU score
            if (iouData.length > 1) {
                let maxIoU = -Infinity;
                for (let i = 0; i < iouData.length; i++) {
                    if (iouData[i] > maxIoU) {
                        maxIoU = iouData[i];
                        bestMaskIndex = i;
                    }
                }
                this.log(`Selected mask ${bestMaskIndex} with IoU ${maxIoU.toFixed(4)} (0=tight, 1=medium, 2=broad)`);
            } else {
                this.log(`Single mask output with IoU ${iouData[0]?.toFixed(4) ?? 'N/A'}`);
            }
        } else {
            this.log('No IoU predictions found, using first mask (index 0)');
        }

        const { mask: rawMask, width: maskWidth, height: maskHeight } = processMaskLogits(maskTensor, bestMaskIndex);

        // DEBUG: Save points overlay on the image
        const cachedImage = this.cachedImageData.get(imageId);
        if (cachedImage) {
            await debugSavePointsOverlay(cachedImage.data, cachedImage.width, cachedImage.height, scaledPoints);
        }

        // DEBUG: Save raw decoder output mask (256x256)
        // Extract only the selected mask's logits for accurate statistics
        const maskSize = maskWidth * maskHeight;
        const selectedMaskLogits = new Float32Array(maskSize);
        const fullLogits = maskTensor.data as Float32Array;
        const logitsOffset = bestMaskIndex * maskSize;
        for (let i = 0; i < maskSize; i++) {
            selectedMaskLogits[i] = fullLogits[logitsOffset + i];
        }
        await debugSaveDecoderOutput(rawMask, maskWidth, maskHeight, selectedMaskLogits);

        // DEBUG: Save all mask candidates (SAM2 outputs 3-4 masks with different IoU scores)
        const iouData = iouTensor ? iouTensor.data as Float32Array : null;
        await debugSaveAllMasks({ data: fullLogits, dims: maskTensor.dims }, iouData);

        // DEBUG: Save mask overlay on original image (like SAM2 demo visualization)
        if (cachedImage) {
            await debugSaveMaskOverlay(
                cachedImage.data,
                cachedImage.width,
                cachedImage.height,
                rawMask,
                maskWidth,
                maskHeight
            );
        }

        // Resize mask to original dimensions
        const finalMask = resizeMask(rawMask, maskWidth, maskHeight, originalWidth, originalHeight);

        // DEBUG: Save final resized mask
        await debugSaveFinalMask(finalMask, originalWidth, originalHeight);

        // Build all mask candidates for multi-mask selection UI
        const numMasks = iouData ? iouData.length : 1;
        const allMasks: SAM2MaskCandidate[] = [];
        
        for (let i = 0; i < numMasks; i++) {
            // Extract logits for this mask
            const maskLogits = new Float32Array(maskSize);
            const offset = i * maskSize;
            for (let j = 0; j < maskSize; j++) {
                maskLogits[j] = fullLogits[offset + j];
            }
            
            // Process this mask's logits to get binary mask at 256x256
            const { mask: rawMaskForCandidate } = processMaskLogits(maskTensor, i);
            
            // Resize to original dimensions
            const resizedMask = resizeMask(rawMaskForCandidate, maskWidth, maskHeight, originalWidth, originalHeight);
            
            allMasks.push({
                index: i,
                iouScore: iouData ? iouData[i] : 1.0,
                mask: resizedMask,
                width: originalWidth,
                height: originalHeight,
                logits: maskLogits
            });
        }
        
        this.log(`Decoded in ${decodeTime.toFixed(0)}ms, returning ${allMasks.length} mask candidates`);

        // Return all masks along with the selected one
        return {
            mask: finalMask,
            width: originalWidth,
            height: originalHeight,
            logits: selectedMaskLogits,
            allMasks,
            selectedMaskIndex: bestMaskIndex,
            timing: {
                decode: decodeTime,
                total: decodeTime
            }
        };
    }

    /**
     * Combined encode + decode for convenience.
     * 
     * @param imageId - Unique ID for caching
     * @param imageData - RGBA pixel data
     * @param width - Image width
     * @param height - Image height
     * @param points - Point prompts
     * @param previousMask - Optional previous mask logits for iterative refinement
     */
    async segment(
        imageId: string,
        imageData: Uint8Array,
        width: number,
        height: number,
        points: SAM2Point[],
        previousMask?: Float32Array
    ): Promise<SAM2Result> {
        const encodeTime = await this.encodeImage(imageId, imageData, width, height);
        const result = await this.decode(imageId, points, width, height, previousMask);
        
        result.timing.encode = encodeTime;
        result.timing.total = encodeTime + result.timing.decode;

        return result;
    }

    /**
     * Clear cached embeddings for an image.
     */
    clearImageCache(imageId: string): void {
        this.imageEmbeddings.delete(imageId);
        this.highResFeatures.delete(imageId);
    }

    /**
     * Clear all cached embeddings.
     */
    clearAllCaches(): void {
        this.imageEmbeddings.clear();
        this.highResFeatures.clear();
    }

    /**
     * Get current execution provider.
     */
    getProvider(): ExecutionProvider {
        return this.currentProvider;
    }

    /**
     * Check if initialized.
     */
    isInitialized(): boolean {
        return this.encoderSession !== null && this.decoderSession !== null;
    }

    /**
     * Dispose sessions and clean up.
     */
    private async disposeSessions(): Promise<void> {
        if (this.encoderSession) {
            try {
                await this.encoderSession.release();
            } catch { /* ignore */ }
            this.encoderSession = null;
        }
        if (this.decoderSession) {
            try {
                await this.decoderSession.release();
            } catch { /* ignore */ }
            this.decoderSession = null;
        }
    }

    /**
     * Full cleanup.
     */
    async dispose(): Promise<void> {
        this.clearAllCaches();
        await this.disposeSessions();
    }

    private log(message: string): void {
        console.log(`[SAM2] ${message}`);
        if (this.verbose) {
            console.log(`[SAM2] ${message}`);
        }
    }
}

/**
 * Create a new SAM2 instance.
 */
export function createSAM2(): SAM2Core {
    return new SAM2Core();
}
