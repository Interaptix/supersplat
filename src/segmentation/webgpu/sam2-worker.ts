/**
 * SAM2 Web Worker for off-main-thread inference.
 * 
 * This worker handles model loading, encoding, and decoding
 * to keep the main thread responsive during inference.
 * 
 * Note: This is a classic worker (not module) to allow importScripts()
 * for loading ONNX Runtime from CDN.
 */

import { SAM2Core, SAM2Point, SAM2Options, ExecutionProvider, SAM2MaskCandidate } from './sam2-core';
import { setDebugDataCallback } from './image-utils';

// Message types from main thread to worker
export type WorkerRequestMessage =
    | { type: 'initialize'; encoderData: ArrayBuffer; decoderData: ArrayBuffer; options?: SAM2Options }
    | { type: 'encode'; imageId: string; imageData: Uint8Array; width: number; height: number }
    | { type: 'decode'; imageId: string; points: SAM2Point[]; width: number; height: number; previousMask?: Float32Array }
    | { type: 'segment'; imageId: string; imageData: Uint8Array; width: number; height: number; points: SAM2Point[]; previousMask?: Float32Array }
    | { type: 'clearCache'; imageId?: string }
    | { type: 'dispose' }
    | { type: 'getStatus' };

// Debug data types
export type DebugDataType = 'encoder_input' | 'decoder_output' | 'final_mask' | 'points_overlay' | 'mask_overlay';

export interface DebugImageData {
    debugType: DebugDataType;
    imageData: Uint8Array;
    width: number;
    height: number;
    filename: string;
    /** Additional info for decoder output */
    logitsStats?: {
        min: number;
        max: number;
        mean: number;
        positiveCount: number;
        totalCount: number;
    };
    /** Points info for overlay */
    points?: Array<{ x: number; y: number; type: 'fg' | 'bg' }>;
}

/** Mask candidate for multi-mask selection UI (mirrors SAM2MaskCandidate) */
export interface WorkerMaskCandidate {
    index: number;
    iouScore: number;
    mask: Uint8Array;
    width: number;
    height: number;
    logits: Float32Array;
}

// Message types from worker to main thread
export type WorkerResponseMessage =
    | { type: 'initialized'; provider: ExecutionProvider }
    | { type: 'encoded'; imageId: string; encodeTime: number }
    | { type: 'decoded'; imageId: string; mask: Uint8Array; width: number; height: number; logits?: Float32Array; allMasks?: WorkerMaskCandidate[]; selectedMaskIndex?: number; timing: { encode?: number; decode: number; total: number } }
    | { type: 'segmented'; imageId: string; mask: Uint8Array; width: number; height: number; logits?: Float32Array; allMasks?: WorkerMaskCandidate[]; selectedMaskIndex?: number; timing: { encode?: number; decode: number; total: number } }
    | { type: 'cacheCleared'; imageId?: string }
    | { type: 'disposed' }
    | { type: 'status'; initialized: boolean; provider: ExecutionProvider | null }
    | { type: 'error'; message: string; requestType: string }
    | { type: 'debug'; data: DebugImageData };

// Worker-side implementation
let sam2: SAM2Core | null = null;

/**
 * Handle incoming messages from main thread.
 */
self.onmessage = async (event: MessageEvent<WorkerRequestMessage>) => {
    const message = event.data;

    try {
        switch (message.type) {
            case 'initialize':
                await handleInitialize(message);
                break;
            case 'encode':
                await handleEncode(message);
                break;
            case 'decode':
                await handleDecode(message);
                break;
            case 'segment':
                await handleSegment(message);
                break;
            case 'clearCache':
                handleClearCache(message);
                break;
            case 'dispose':
                await handleDispose();
                break;
            case 'getStatus':
                handleGetStatus();
                break;
            default:
                postError('Unknown message type', 'unknown');
        }
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        postError(errorMessage, message.type);
    }
};

/**
 * Initialize SAM2 with model data.
 */
async function handleInitialize(
    message: Extract<WorkerRequestMessage, { type: 'initialize' }>
): Promise<void> {
    // Dispose existing instance if any
    if (sam2) {
        await sam2.dispose();
    }

    // Set up debug callback to send debug data to main thread
    setDebugDataCallback((data: DebugImageData) => {
        postMessage({ type: 'debug', data } as WorkerResponseMessage);
    });

    sam2 = new SAM2Core();
    const provider = await sam2.initialize(
        message.encoderData,
        message.decoderData,
        message.options
    );

    postMessage({
        type: 'initialized',
        provider
    } as WorkerResponseMessage);
}

/**
 * Encode an image (compute embeddings).
 */
async function handleEncode(
    message: Extract<WorkerRequestMessage, { type: 'encode' }>
): Promise<void> {
    if (!sam2) {
        throw new Error('SAM2 not initialized');
    }

    const encodeTime = await sam2.encodeImage(
        message.imageId,
        message.imageData,
        message.width,
        message.height
    );

    postMessage({
        type: 'encoded',
        imageId: message.imageId,
        encodeTime
    } as WorkerResponseMessage);
}

/**
 * Decode with point prompts (generate mask).
 */
async function handleDecode(
    message: Extract<WorkerRequestMessage, { type: 'decode' }>
): Promise<void> {
    if (!sam2) {
        throw new Error('SAM2 not initialized');
    }

    const result = await sam2.decode(
        message.imageId,
        message.points,
        message.width,
        message.height,
        message.previousMask
    );

    // Convert SAM2MaskCandidate[] to WorkerMaskCandidate[] for transfer
    const allMasks: WorkerMaskCandidate[] | undefined = result.allMasks?.map(m => ({
        index: m.index,
        iouScore: m.iouScore,
        mask: m.mask,
        width: m.width,
        height: m.height,
        logits: m.logits
    }));

    // Transfer the mask buffer for efficiency
    const response: WorkerResponseMessage = {
        type: 'decoded',
        imageId: message.imageId,
        mask: result.mask,
        width: result.width,
        height: result.height,
        logits: result.logits,
        allMasks,
        selectedMaskIndex: result.selectedMaskIndex,
        timing: result.timing
    };

    // Use transferable objects for large arrays
    const transferList: Transferable[] = [result.mask.buffer];
    if (result.logits) {
        transferList.push(result.logits.buffer);
    }
    // Transfer all mask candidate buffers
    if (allMasks) {
        for (const m of allMasks) {
            transferList.push(m.mask.buffer);
            transferList.push(m.logits.buffer);
        }
    }

    (self as unknown as Worker).postMessage(response, transferList);
}

/**
 * Combined encode + decode operation.
 */
async function handleSegment(
    message: Extract<WorkerRequestMessage, { type: 'segment' }>
): Promise<void> {
    if (!sam2) {
        throw new Error('SAM2 not initialized');
    }

    const result = await sam2.segment(
        message.imageId,
        message.imageData,
        message.width,
        message.height,
        message.points,
        message.previousMask  // Pass previous mask for iterative refinement
    );

    // Convert SAM2MaskCandidate[] to WorkerMaskCandidate[] for transfer
    const allMasks: WorkerMaskCandidate[] | undefined = result.allMasks?.map(m => ({
        index: m.index,
        iouScore: m.iouScore,
        mask: m.mask,
        width: m.width,
        height: m.height,
        logits: m.logits
    }));

    // Transfer the mask buffer for efficiency
    const response: WorkerResponseMessage = {
        type: 'segmented',
        imageId: message.imageId,
        mask: result.mask,
        width: result.width,
        height: result.height,
        logits: result.logits,
        allMasks,
        selectedMaskIndex: result.selectedMaskIndex,
        timing: result.timing
    };

    // Use transferable objects for large arrays
    const transferList: Transferable[] = [result.mask.buffer];
    if (result.logits) {
        transferList.push(result.logits.buffer);
    }
    // Transfer all mask candidate buffers
    if (allMasks) {
        for (const m of allMasks) {
            transferList.push(m.mask.buffer);
            transferList.push(m.logits.buffer);
        }
    }

    (self as unknown as Worker).postMessage(response, transferList);
}

/**
 * Clear cached embeddings.
 */
function handleClearCache(
    message: Extract<WorkerRequestMessage, { type: 'clearCache' }>
): void {
    if (!sam2) {
        throw new Error('SAM2 not initialized');
    }

    if (message.imageId) {
        sam2.clearImageCache(message.imageId);
    } else {
        sam2.clearAllCaches();
    }

    postMessage({
        type: 'cacheCleared',
        imageId: message.imageId
    } as WorkerResponseMessage);
}

/**
 * Dispose SAM2 and clean up resources.
 */
async function handleDispose(): Promise<void> {
    if (sam2) {
        await sam2.dispose();
        sam2 = null;
    }

    postMessage({
        type: 'disposed'
    } as WorkerResponseMessage);
}

/**
 * Get current status.
 */
function handleGetStatus(): void {
    postMessage({
        type: 'status',
        initialized: sam2?.isInitialized() ?? false,
        provider: sam2?.getProvider() ?? null
    } as WorkerResponseMessage);
}

/**
 * Post error message to main thread.
 */
function postError(message: string, requestType: string): void {
    postMessage({
        type: 'error',
        message,
        requestType
    } as WorkerResponseMessage);
}

// Export types for main thread usage
export type { SAM2Point, SAM2Options, ExecutionProvider };
