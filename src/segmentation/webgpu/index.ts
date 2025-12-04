/**
 * WebGPU SAM2 Segmentation Module
 * 
 * Provides in-browser SAM2 segmentation using ONNX Runtime Web
 * with WebGPU acceleration and WASM fallback.
 */

// Main provider
export {
    WebGPUProvider,
    createWebGPUProvider,
    type ProviderState,
    type ModelLoadProgressCallback,
    type StatusChangeCallback,
    type WebGPUProviderOptions
} from './webgpu-provider';

// Capability detection
export {
    checkWebGPUCapabilities,
    type WebGPUCapabilities,
    LOW_VRAM_THRESHOLD
} from './capability-check';

// Model loading utilities
export {
    loadAllModels,
    loadModel,
    areModelsCached,
    getCachedModelInfo,
    clearModelCache,
    formatBytes,
    getTotalModelSize,
    SAM2_MODELS,
    type ModelConfig,
    type ProgressCallback
} from './model-loader';

// Types re-exported from worker
export type { ExecutionProvider, SAM2Options, SAM2Point } from './sam2-worker';

// Image utilities for mask processing
export { resizeMaskSmooth, resizeMaskAsCanvas } from './image-utils';
