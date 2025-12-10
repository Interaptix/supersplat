/**
 * SAM2 Web Worker
 * Handles SAM2 model loading and inference in a background thread.
 */

// Local debug flag for worker context (can't import from debug-utils.ts in worker)
const DEBUG_SAM2_WORKER = false;
const debugLog: (...args: unknown[]) => void = DEBUG_SAM2_WORKER ?
    (...args: unknown[]) => console.log(...args) :
    () => {};

debugLog('[SAM2 Worker] Worker script loaded');

import * as ort from 'onnxruntime-web';

import { SAM2 } from './SAM2';

// Worker context - use intersection type for worker global scope
declare const self: typeof globalThis & {
    postMessage: (message: unknown) => void;
    onmessage: ((e: MessageEvent) => void) | null;
};

debugLog('[SAM2 Worker] Imports complete, setting up message handler');

// SAM2 instance
const sam = new SAM2();

// Stats tracking
interface Stats {
    device: string;
    downloadModelsTime: number[];
    encodeImageTimes: number[];
    decodeTimes: number[];
}

const stats: Stats = {
    device: 'unknown',
    downloadModelsTime: [],
    encodeImageTimes: [],
    decodeTimes: []
};

// Message types
interface InitModelMessage {
    type: 'initModel';
}

interface EncodeImageMessage {
    type: 'encodeImage';
    data: {
        float32Array: Float32Array;
        shape: number[];
    };
}

interface DecodeMaskMessage {
    type: 'decodeMask';
    data: {
        points: Array<{ x: number; y: number; label: number }>;
        maskArray?: Float32Array;
        maskShape?: number[];
    };
}

interface StatsMessage {
    type: 'stats';
}

type WorkerMessage = InitModelMessage | EncodeImageMessage | DecodeMaskMessage | StatsMessage;

// Handle incoming messages
self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
    const { type } = e.data;

    if (type === 'initModel') {
        // Download models
        self.postMessage({ type: 'downloadInProgress' });
        const startTime = performance.now();
        await sam.downloadModels();
        const durationMs = performance.now() - startTime;
        stats.downloadModelsTime.push(durationMs);

        // Create sessions
        self.postMessage({ type: 'loadingInProgress' });
        const report = await sam.createSessions();

        stats.device = report.device || 'unknown';

        self.postMessage({ type: 'modelReady', data: report });
        self.postMessage({ type: 'stats', data: stats });
    } else if (type === 'encodeImage') {
        const message = e.data as EncodeImageMessage;
        const { float32Array, shape } = message.data;
        const imgTensor = new ort.Tensor('float32', float32Array, shape);

        const startTime = performance.now();
        await sam.encodeImage(imgTensor);
        const durationMs = performance.now() - startTime;
        stats.encodeImageTimes.push(durationMs);

        self.postMessage({
            type: 'encodeImageDone',
            data: { durationMs }
        });
        self.postMessage({ type: 'stats', data: stats });
    } else if (type === 'decodeMask') {
        const message = e.data as DecodeMaskMessage;
        const { points, maskArray, maskShape } = message.data;

        const startTime = performance.now();

        let decodingResults: ort.InferenceSession.OnnxValueMapType;
        if (maskArray && maskShape) {
            const maskTensor = new ort.Tensor('float32', maskArray, maskShape);
            decodingResults = await sam.decode(points, maskTensor);
        } else {
            decodingResults = await sam.decode(points);
        }

        const durationMs = performance.now() - startTime;
        stats.decodeTimes.push(durationMs);

        debugLog('Decoding results:', decodingResults);

        // Convert tensors to transferable format
        const serializedResults: Record<string, { data: Float32Array; dims: readonly number[] }> = {};
        for (const [key, tensor] of Object.entries(decodingResults)) {
            serializedResults[key] = {
                data: (tensor as ort.Tensor).data as Float32Array,
                dims: (tensor as ort.Tensor).dims
            };
        }

        self.postMessage({ type: 'decodeMaskResult', data: serializedResults });
        self.postMessage({ type: 'stats', data: stats });
    } else if (type === 'stats') {
        self.postMessage({ type: 'stats', data: stats });
    } else {
        throw new Error(`Unknown message type: ${type}`);
    }
};

// Export empty object for TypeScript module
export {};
