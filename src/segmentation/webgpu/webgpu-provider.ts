/**
 * WebGPU-based SAM2 segmentation provider.
 * 
 * Implements the SegmentationProvider interface using in-browser
 * SAM2 inference via ONNX Runtime Web with WebGPU acceleration.
 */

import { SegmentationProvider, SegmentationError } from '../provider';
import { SegmentationRequest, SegmentationResponse } from '../types';
import { checkWebGPUCapabilities, WebGPUCapabilities } from './capability-check';
import { loadAllModels, areModelsCached, clearModelCache, formatBytes, getTotalModelSize } from './model-loader';
import { downloadDebugImage } from './image-utils';
import type { WorkerRequestMessage, WorkerResponseMessage, ExecutionProvider, SAM2Options, WorkerMaskCandidate } from './sam2-worker';
import type { MaskCandidate } from '../types';

/** Provider state */
export type ProviderState = 'idle' | 'loading-models' | 'initializing' | 'ready' | 'processing' | 'error';

/** Progress callback for model loading */
export type ModelLoadProgressCallback = (loaded: number, total: number, stage: string) => void;

/** Status change callback */
export type StatusChangeCallback = (state: ProviderState, details?: string) => void;

/** WebGPU provider options */
export interface WebGPUProviderOptions {
    /** Callback for model download progress */
    onProgress?: ModelLoadProgressCallback;
    /** Callback for status changes */
    onStatusChange?: StatusChangeCallback;
    /** Preferred execution provider (webgpu or wasm) */
    preferredProvider?: ExecutionProvider;
    /** Enable verbose logging */
    verbose?: boolean;
}

/**
 * WebGPU SAM2 Segmentation Provider.
 * 
 * Uses a Web Worker for inference to keep the main thread responsive.
 */
export class WebGPUProvider implements SegmentationProvider {
    readonly id = 'webgpu-sam2';
    readonly name = 'In-Browser SAM2 (WebGPU)';

    private worker: Worker | null = null;
    private state: ProviderState = 'idle';
    private capabilities: WebGPUCapabilities | null = null;
    private currentProvider: ExecutionProvider | null = null;
    private pendingRequests: Map<string, {
        resolve: (value: unknown) => void;
        reject: (reason: Error) => void;
    }> = new Map();
    private requestCounter = 0;
    private abortController: AbortController | null = null;
    private options: WebGPUProviderOptions;
    private imageIdCounter = 0;
    
    /** 
     * Previous mask logits for iterative refinement.
     * Key: imageId, Value: raw float32 logits from best mask (256x256)
     */
    private previousMaskLogits: Map<string, Float32Array> = new Map();
    
    /** Current active image ID for iterative refinement tracking */
    private currentImageId: string | null = null;

    constructor(options: WebGPUProviderOptions = {}) {
        this.options = options;
    }

    /**
     * Check if WebGPU SAM2 is available.
     */
    async isAvailable(): Promise<boolean> {
        if (!this.capabilities) {
            this.capabilities = await checkWebGPUCapabilities();
        }
        // Available if WebGPU is supported OR if WASM fallback is acceptable
        // We always have WASM fallback, so return true
        return true;
    }

    /**
     * Get WebGPU capabilities.
     */
    async getCapabilities(): Promise<WebGPUCapabilities> {
        if (!this.capabilities) {
            this.capabilities = await checkWebGPUCapabilities();
        }
        return this.capabilities;
    }

    /**
     * Check if models are already cached.
     */
    async areModelsCached(): Promise<boolean> {
        return areModelsCached();
    }

    /**
     * Clear cached models.
     */
    async clearCache(): Promise<void> {
        await clearModelCache();
    }

    /**
     * Get total model size for download progress display.
     */
    getTotalModelSize(): number {
        return getTotalModelSize();
    }

    /**
     * Format bytes for display.
     */
    formatBytes(bytes: number): string {
        return formatBytes(bytes);
    }

    /**
     * Get current state.
     */
    getState(): ProviderState {
        return this.state;
    }

    /**
     * Get current execution provider.
     */
    getExecutionProvider(): ExecutionProvider | null {
        return this.currentProvider;
    }

    /**
     * Initialize the provider (load models, create worker).
     * Call this before using segmentSingleView.
     */
    async initialize(): Promise<ExecutionProvider> {
        if (this.state === 'ready') {
            return this.currentProvider!;
        }

        if (this.state === 'loading-models' || this.state === 'initializing') {
            throw new Error('Initialization already in progress');
        }

        try {
            // Load models (from cache or download)
            this.setState('loading-models');
            this.abortController = new AbortController();

            const { encoder, decoder } = await loadAllModels(
                this.options.onProgress,
                this.abortController.signal
            );

            // Create worker
            this.setState('initializing');
            await this.createWorker();

            // Initialize SAM2 in worker
            const provider = await this.initializeWorker(encoder, decoder);
            this.currentProvider = provider;

            this.setState('ready');
            return provider;
        } catch (error) {
            this.setState('error');
            if (error instanceof Error && error.name === 'AbortError') {
                throw new SegmentationError('Initialization aborted', 'ABORTED');
            }
            throw error;
        } finally {
            this.abortController = null;
        }
    }

    /**
     * Start a new segmentation session (new image).
     * Call this when the user starts segmenting a new image/view.
     * This clears previous mask logits so the decoder starts fresh.
     */
    startNewSession(): void {
        this.log('Starting new segmentation session');
        // Clear previous mask logits - next decode will start fresh
        this.previousMaskLogits.clear();
        // Generate new imageId for this session
        this.currentImageId = `img_${++this.imageIdCounter}`;
    }

    /**
     * Clear iterative refinement state (previous mask).
     * Call this to reset without starting a completely new session.
     */
    clearPreviousMask(): void {
        if (this.currentImageId) {
            this.previousMaskLogits.delete(this.currentImageId);
            this.log('Cleared previous mask for iterative refinement');
        }
    }

    /**
     * Pre-encode an image without running the decoder.
     * This allows encoding to happen in the background while the user positions their click.
     * Call this immediately when capturing a new image to pre-warm the embeddings.
     * 
     * @returns Promise that resolves with encode timing when complete
     */
    async preEncodeImage(imageData: Uint8Array, width: number, height: number): Promise<{ encodeTime: number }> {
        if (this.state !== 'ready') {
            if (this.state === 'idle') {
                await this.initialize();
            } else {
                throw new Error(`Provider not ready for encoding (state: ${this.state})`);
            }
        }

        // Use current session imageId, or create one if not started
        if (!this.currentImageId) {
            this.currentImageId = `img_${++this.imageIdCounter}`;
            this.log(`Auto-created session for pre-encoding: ${this.currentImageId}`);
        }
        const imageId = this.currentImageId;

        this.log(`Pre-encoding image ${imageId} (${width}x${height})`);

        // Send encode-only request to worker
        const result = await this.sendWorkerMessage({
            type: 'encode',
            imageId,
            imageData,
            width,
            height
        }) as Extract<WorkerResponseMessage, { type: 'encoded' }>;

        this.log(`Pre-encoding complete in ${result.encodeTime}ms`);

        return {
            encodeTime: result.encodeTime
        };
    }

    /**
     * Perform single-view segmentation with iterative refinement support.
     * 
     * IMPORTANT: For iterative refinement (clicking multiple points on same image):
     * - Call startNewSession() when the image changes
     * - Each subsequent call with more points will use the previous mask for refinement
     * - The decoder uses previous mask logits to maintain context across clicks
     */
    async segmentSingleView(request: SegmentationRequest): Promise<SegmentationResponse> {
        if (this.state !== 'ready') {
            if (this.state === 'idle') {
                // Auto-initialize
                await this.initialize();
            } else {
                throw new Error(`Provider not ready (state: ${this.state})`);
            }
        }

        this.setState('processing');

        try {
            // Use current session imageId, or create one if not started
            if (!this.currentImageId) {
                this.currentImageId = `img_${++this.imageIdCounter}`;
                this.log(`Auto-created session: ${this.currentImageId}`);
            }
            const imageId = this.currentImageId;

            // Convert points to SAM2 format
            const sam2Points = request.points.map(p => ({
                x: p.x,
                y: p.y,
                type: p.type
            }));

            // Get previous mask logits for iterative refinement
            const previousMask = this.previousMaskLogits.get(imageId);
            console.log(`all previousMasks keys:`, Array.from(this.previousMaskLogits.keys()));
            console.log(`current imageId:`, imageId);
            if (previousMask) {
                this.log(`Using previous mask for iterative refinement (${previousMask.length} floats)`);
            }

            // Send segment request to worker with previous mask for refinement
            const result = await this.sendWorkerMessage({
                type: 'segment',
                imageId,
                imageData: request.image,
                width: request.width,
                height: request.height,
                points: sam2Points,
                previousMask: previousMask ? new Float32Array(previousMask) : undefined
            }) as Extract<WorkerResponseMessage, { type: 'segmented' }>;

            // Store the best mask's logits for next iterative refinement
            // The logits returned are the raw float32 values from the decoder
            if (result.logits) {
                // Extract only the best mask logits (256x256 = 65536 values)
                // The full logits contain all mask candidates, we need just the selected one
                const maskSize = 256 * 256;
                const bestMaskLogits = new Float32Array(maskSize);
                
                // Find best mask index from IoU (same logic as in sam2-core)
                // For now, assume the logits returned are already the best mask's logits
                // If result.logits contains all masks, we'd need the IoU scores to select
                if (result.logits.length === maskSize) {
                    // Already extracted best mask
                    bestMaskLogits.set(result.logits);
                    console.warn('[WebGPUProvider] Received single mask logits for iterative refinement');
                } else if (result.logits.length >= maskSize) {
                    // Full output - take first mask (index 0) as default
                    // TODO: Use IoU scores to select best mask index
                    bestMaskLogits.set(result.logits.subarray(0, maskSize));
                    console.warn('[WebGPUProvider] Received full logits, using first mask for iterative refinement');
                }
                
                this.previousMaskLogits.set(imageId, bestMaskLogits);
                this.log(`Stored previous mask logits for iterative refinement`);
            }

            // DON'T clear cache - we want to reuse embeddings for iterative refinement
            // The cache will be cleared when startNewSession() is called

            this.setState('ready');

            // Convert worker mask candidates to MaskCandidate interface
            let allMasks: MaskCandidate[] | undefined;
            if (result.allMasks && result.allMasks.length > 0) {
                allMasks = result.allMasks.map((wm: WorkerMaskCandidate) => ({
                    index: wm.index,
                    iouScore: wm.iouScore,
                    mask: wm.mask,
                    width: wm.width,
                    height: wm.height,
                    logits: wm.logits
                }));
                this.log(`Received ${allMasks.length} mask candidates`);
            }

            return {
                width: result.width,
                height: result.height,
                mask: result.mask,
                logits: result.logits,
                allMasks,
                selectedMaskIndex: result.selectedMaskIndex
            };
        } catch (error) {
            this.setState('ready');
            throw error;
        }
    }

    /**
     * Abort any in-progress operation.
     */
    abort(): void {
        // Abort model loading if in progress
        if (this.abortController) {
            this.abortController.abort();
        }

        // Reject all pending worker requests
        for (const [, { reject }] of this.pendingRequests) {
            reject(new SegmentationError('Operation aborted', 'ABORTED'));
        }
        this.pendingRequests.clear();
    }

    /**
     * Dispose of resources.
     */
    async dispose(): Promise<void> {
        this.abort();

        if (this.worker) {
            try {
                await this.sendWorkerMessage({ type: 'dispose' });
            } catch {
                // Ignore errors during dispose
            }
            this.worker.terminate();
            this.worker = null;
        }

        this.setState('idle');
        this.currentProvider = null;
    }

    /**
     * Create the Web Worker.
     */
    private async createWorker(): Promise<void> {
        return new Promise((resolve, reject) => {
            try {
                // Create worker from bundled module (sam2-worker.js is built by Rollup)
                this.worker = new Worker('sam2-worker.js', { type: 'module' });

                this.worker.onmessage = (event: MessageEvent<WorkerResponseMessage>) => {
                    this.handleWorkerMessage(event.data);
                };

                this.worker.onerror = (error) => {
                    this.log(`Worker error: ${error.message}`);
                    this.setState('error');
                    reject(new Error(`Worker error: ${error.message}`));
                };

                resolve();
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Initialize SAM2 in the worker.
     */
    private async initializeWorker(
        encoderData: ArrayBuffer,
        decoderData: ArrayBuffer
    ): Promise<ExecutionProvider> {
        const options: SAM2Options = {
            preferredProvider: this.options.preferredProvider ?? 'webgpu',
            verbose: this.options.verbose
        };

        const result = await this.sendWorkerMessage({
            type: 'initialize',
            encoderData,
            decoderData,
            options
        }) as Extract<WorkerResponseMessage, { type: 'initialized' }>;

        return result.provider;
    }

    /**
     * Send a message to the worker and wait for response.
     */
    private sendWorkerMessage(message: WorkerRequestMessage): Promise<WorkerResponseMessage> {
        return new Promise((resolve, reject) => {
            if (!this.worker) {
                reject(new Error('Worker not created'));
                return;
            }

            const requestId = `req_${++this.requestCounter}`;
            this.pendingRequests.set(requestId, {
                resolve: resolve as (value: unknown) => void,
                reject
            });

            // Attach request ID for tracking (not used by worker, just for our mapping)
            const messageWithId = { ...message, _requestId: requestId };

            // Determine transferable objects
            const transferList: Transferable[] = [];
            if ('encoderData' in message && message.encoderData) {
                transferList.push(message.encoderData);
            }
            if ('decoderData' in message && message.decoderData) {
                transferList.push(message.decoderData);
            }
            if ('imageData' in message && message.imageData) {
                transferList.push(message.imageData.buffer);
            }
            if ('previousMask' in message && message.previousMask) {
                transferList.push(message.previousMask.buffer);
            }

            if (transferList.length > 0) {
                this.worker.postMessage(messageWithId, transferList);
            } else {
                this.worker.postMessage(messageWithId);
            }
        });
    }

    /**
     * Handle messages from the worker.
     */
    private handleWorkerMessage(message: WorkerResponseMessage): void {
        // Handle debug messages separately (don't resolve pending requests)
        if (message.type === 'debug') {
            this.log(`Received debug data: ${message.data.debugType} (${message.data.width}x${message.data.height})`);
            downloadDebugImage(message.data).catch(err => {
                console.error('[WebGPUProvider] Failed to download debug image:', err);
            });
            return;
        }

        // Find and resolve the pending request
        // Since we process messages in order, resolve the first pending request
        const [requestId, handler] = this.pendingRequests.entries().next().value || [];

        if (message.type === 'error') {
            if (handler) {
                this.pendingRequests.delete(requestId);
                handler.reject(new SegmentationError(
                    message.message,
                    'SERVER_ERROR',
                    { requestType: message.requestType }
                ));
            }
            return;
        }

        if (handler) {
            this.pendingRequests.delete(requestId);
            handler.resolve(message);
        }
    }

    /**
     * Update state and notify listener.
     */
    private setState(state: ProviderState, details?: string): void {
        this.state = state;
        this.options.onStatusChange?.(state, details);
        this.log(`State: ${state}${details ? ` (${details})` : ''}`);
    }

    /**
     * Log message if verbose.
     */
    private log(message: string): void {
        if (this.options.verbose) {
            console.log(`[WebGPUProvider] ${message}`);
        }
    }
}

/**
 * Create a new WebGPU provider instance.
 */
export function createWebGPUProvider(options?: WebGPUProviderOptions): WebGPUProvider {
    return new WebGPUProvider(options);
}
