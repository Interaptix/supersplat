import { Events } from '../events';
import {
    SegmentationPoint,
    SegmentationRequest,
    SegmentationResponse,
    MaskCandidate,
    applyMaskToSelection,
    WebGPUProvider,
    createWebGPUProvider,
    checkWebGPUCapabilities,
    areModelsCached,
    getTotalModelSize,
    formatBytes,
    LOW_VRAM_THRESHOLD,
    ProviderState
} from './index';

/** Pending mask data waiting for user confirmation */
interface PendingMask {
    response: SegmentationResponse;
    canvasWidth: number;
    canvasHeight: number;
}

/**
 * Segmentation Service
 * 
 * Manages the WebGPU segmentation provider and handles SAM tool events.
 * Connects the UI (sam-panel, sam-selection tool) with the WebGPU backend.
 */
export class SegmentationService {
    private provider: WebGPUProvider | null = null;
    private events: Events;
    private canvas: HTMLCanvasElement;
    private isInitializing = false;
    private initPromise: Promise<void> | null = null;
    private pendingMask: PendingMask | null = null;

    constructor(events: Events, canvas: HTMLCanvasElement) {
        this.events = events;
        this.canvas = canvas;
        this.registerEventHandlers();
    }

    /**
     * Register event handlers for SAM tool integration
     */
    private registerEventHandlers(): void {
        // Handle segmentation request from SAM tool
        this.events.on('sam.segment', async (points: SegmentationPoint[]) => {
            await this.handleSegmentRequest(points);
        });

        // Allow querying provider status
        this.events.function('sam.getProviderStatus', () => {
            return this.getStatus();
        });

        // Allow checking if models are cached
        this.events.function('sam.areModelsCached', async () => {
            return areModelsCached();
        });

        // Allow getting model download info
        this.events.function('sam.getModelDownloadInfo', () => {
            const totalSize = getTotalModelSize();
            return {
                totalSize,
                formattedSize: formatBytes(totalSize)
            };
        });

        // Handle provider initialization request
        this.events.on('sam.initializeProvider', async () => {
            await this.initializeProvider();
        });

        // Handle disposal
        this.events.on('sam.disposeProvider', () => {
            this.dispose();
        });

        // Handle apply mask (user confirms the mask preview)
        this.events.on('sam.applyMask', () => {
            this.applyPendingMask();
        });

        // Handle cancel mask (user rejects the mask preview)
        this.events.on('sam.cancelMask', () => {
            this.cancelPendingMask();
        });

        // Handle viewport preview capture (when SAM tool activates)
        this.events.on('sam.capturePreview', async () => {
            await this.captureViewportPreview();
        });
    }

    /**
     * Get current provider status
     */
    getStatus(): {
        state: ProviderState | 'not-initialized';
        modelsCached: boolean;
        error?: string;
    } {
        if (!this.provider) {
            return {
                state: 'not-initialized',
                modelsCached: false
            };
        }

        return {
            state: this.provider.getState(),
            modelsCached: false // Will be updated when provider has this method
        };
    }

    /**
     * Initialize the WebGPU provider
     */
    async initializeProvider(): Promise<void> {
        // Return existing promise if already initializing
        if (this.initPromise) {
            return this.initPromise;
        }

        // Already initialized
        if (this.provider && this.provider.getState() === 'ready') {
            return;
        }

        this.isInitializing = true;
        this.initPromise = this._doInitialize();

        try {
            await this.initPromise;
        } finally {
            this.isInitializing = false;
            this.initPromise = null;
        }
    }

    /**
     * Internal initialization logic
     */
    private async _doInitialize(): Promise<void> {
        // Check WebGPU capabilities first
        const capabilities = await checkWebGPUCapabilities();

        // Fire capability event for UI to handle warnings
        this.events.fire('sam.capabilities', capabilities);

        if (!capabilities.available) {
            const error = capabilities.unavailableReason || 'WebGPU not available';
            this.events.fire('sam.initError', error);
            throw new Error(error);
        }

        // Check for low VRAM
        if (capabilities.isLowVRAM) {
            this.events.fire('sam.lowVramWarning', {
                estimated: capabilities.estimatedVRAM,
                threshold: LOW_VRAM_THRESHOLD
            });
        }

        // Create provider with progress callbacks
        this.provider = createWebGPUProvider({
            verbose: true,
            onProgress: (loaded, total, stage) => {
                this.events.fire('sam.modelLoadProgress', { loaded, total, stage });
            },
            onStatusChange: (state, details) => {
                this.events.fire('sam.providerStatusChanged', { state, details });
            }
        });

        // Initialize the provider
        try {
            await this.provider.initialize();
            this.events.fire('sam.providerReady');
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.events.fire('sam.initError', errorMessage);
            throw error;
        }
    }

    /**
     * Handle a segmentation request from the SAM tool
     */
    private async handleSegmentRequest(points: SegmentationPoint[]): Promise<void> {
        if (points.length === 0) {
            return;
        }

        const startTime = performance.now();
        let encodeTime = 0;
        let decodeTime = 0;

        // Fire segment start event
        this.events.fire('sam.segmentStart');

        try {
            // Ensure provider is initialized
            if (!this.provider || this.provider.getState() !== 'ready') {
                await this.initializeProvider();
            }

            if (!this.provider) {
                throw new Error('Failed to initialize segmentation provider');
            }

            // Get canvas dimensions
            const width = this.canvas.width;
            const height = this.canvas.height;

            // Capture current canvas using PlayCanvas's proper render target reading
            // This properly reads from the WebGL render target instead of the canvas
            const captureStartTime = performance.now();
            const image = await this.events.invoke('render.offscreen', width, height) as Uint8Array;
            
            // Fire image captured event with the image data for panel preview
            this.events.fire('sam.imageCaptured', {
                image,
                width,
                height
            });

            // Create segmentation request
            const request: SegmentationRequest = {
                image,
                width,
                height,
                points: points.map(p => ({
                    x: p.x,
                    y: p.y,
                    type: p.type
                })),
                options: {
                    threshold: 0.5
                }
            };

            // Run segmentation and track timing
            const encodeStartTime = performance.now();
            const response = await this.provider.segmentSingleView(request);
            const segmentEndTime = performance.now();
            
            // Calculate timing (approximate - encode/decode split not available from provider yet)
            const totalSegmentTime = segmentEndTime - encodeStartTime;
            encodeTime = totalSegmentTime * 0.7; // Approximate: encoding takes ~70% of time
            decodeTime = totalSegmentTime * 0.3; // Approximate: decoding takes ~30% of time

            // Store pending mask for preview instead of applying immediately
            if (response.mask) {
                // Store the pending mask data for later application
                this.pendingMask = {
                    response,
                    canvasWidth: width,
                    canvasHeight: height
                };
                
                // Fire mask ready event with mask data for overlay preview
                // Include all mask candidates for multi-mask selection UI
                this.events.fire('sam.maskReady', {
                    mask: response.mask,
                    width: response.width,
                    height: response.height,
                    allMasks: response.allMasks,
                    selectedMaskIndex: response.selectedMaskIndex
                });

                // Fire segment complete with timing stats
                const totalTime = performance.now() - startTime;
                this.events.fire('sam.segmentComplete', {
                    width: response.width,
                    height: response.height,
                    hasPendingMask: true,
                    stats: {
                        totalTime: Math.round(totalTime),
                        encodeTime: Math.round(encodeTime),
                        decodeTime: Math.round(decodeTime)
                    }
                });
            } else {
                throw new Error('No mask returned from segmentation');
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.events.fire('sam.segmentError', errorMessage);
            console.error('Segmentation failed:', error);
        }
    }

    /**
     * Determine the selection operation based on current modifier keys
     */
    private determineSelectionOp(): 'add' | 'remove' | 'set' {
        // Check for modifier keys (this could be enhanced with actual key state)
        // For now, default to 'add' which is the most common use case
        return 'add';
    }

    /**
     * Apply the pending mask to selection (user confirmed the preview)
     */
    private applyPendingMask(): void {
        if (!this.pendingMask) {
            console.warn('No pending mask to apply');
            return;
        }

        const { response, canvasWidth, canvasHeight } = this.pendingMask;
        const selectionOp = this.determineSelectionOp();

        // Apply mask to splat selection
        applyMaskToSelection(
            response,
            {
                op: selectionOp,
                threshold: 0.5,
                targetWidth: canvasWidth,
                targetHeight: canvasHeight
            },
            this.events
        );

        // Clear pending mask
        this.pendingMask = null;

        // Fire mask applied event
        this.events.fire('sam.maskApplied', {
            width: response.width,
            height: response.height
        });
    }

    /**
     * Cancel the pending mask (user rejected the preview)
     */
    private cancelPendingMask(): void {
        if (!this.pendingMask) {
            return;
        }

        // Clear pending mask
        this.pendingMask = null;

        // Fire mask cancelled event
        this.events.fire('sam.maskCancelled');
    }

    /**
     * Capture viewport preview and immediately start pre-encoding.
     * Called when SAM tool activates to show current view in preview panel
     * and pre-warm the SAM2 encoder embeddings while user decides where to click.
     */
    private async captureViewportPreview(): Promise<void> {
        try {
            // Get canvas dimensions
            const width = this.canvas.width;
            const height = this.canvas.height;

            // Capture current canvas using PlayCanvas's proper render target reading
            const image = await this.events.invoke('render.offscreen', width, height) as Uint8Array;
            
            // Start a new segmentation session when capturing a new image
            // This clears any previous mask logits so the decoder starts fresh
            if (this.provider) {
                this.provider.startNewSession();
            }
            
            // Fire image captured event with the image data for panel preview
            this.events.fire('sam.imageCaptured', {
                image,
                width,
                height
            });

            // Immediately start pre-encoding the image in background
            // This happens while the user is looking at the preview and deciding where to click
            this.preEncodeImage(image, width, height);
        } catch (error) {
            console.warn('Failed to capture viewport preview:', error);
        }
    }

    /**
     * Pre-encode the captured image in background.
     * This pre-warms the SAM2 encoder so decoding is instant when user clicks.
     */
    private async preEncodeImage(image: Uint8Array, width: number, height: number): Promise<void> {
        // Fire encoding start event for UI feedback
        this.events.fire('sam.encodingStart');

        try {
            // Ensure provider is initialized
            if (!this.provider || this.provider.getState() !== 'ready') {
                await this.initializeProvider();
            }

            if (!this.provider) {
                throw new Error('Failed to initialize segmentation provider');
            }

            // Pre-encode the image (creates embeddings without running decoder)
            const result = await this.provider.preEncodeImage(image, width, height);

            // Fire encoding complete event with timing
            this.events.fire('sam.encodingComplete', {
                encodeTime: result.encodeTime
            });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.events.fire('sam.encodingError', errorMessage);
            console.warn('Failed to pre-encode image:', error);
        }
    }

    /**
     * Abort any in-progress segmentation
     */
    abort(): void {
        if (this.provider) {
            this.provider.abort();
        }
    }

    /**
     * Dispose of the provider and clean up
     */
    dispose(): void {
        if (this.provider) {
            this.provider.dispose();
            this.provider = null;
        }
    }
}

/**
 * Register segmentation events and create the service
 */
export function registerSegmentationEvents(events: Events, canvas: HTMLCanvasElement): SegmentationService {
    return new SegmentationService(events, canvas);
}
