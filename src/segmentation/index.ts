// Types
export type {
    SegmentationPoint,
    CameraParams,
    SegmentationOptions,
    SegmentationRequest,
    SegmentationResponse,
    MaskCandidate,
    SelectionOp
} from './types';

// Provider interface
export type { SegmentationProvider } from './provider';
export { SegmentationError } from './provider';

// Mask selection utilities
export {
    applyMaskToSelection,
    createMaskPreviewCanvas
} from './mask-selection';
export type { MaskSelectionOptions } from './mask-selection';

// WebGPU provider
export {
    WebGPUProvider,
    createWebGPUProvider,
    checkWebGPUCapabilities,
    areModelsCached,
    clearModelCache,
    LOW_VRAM_THRESHOLD,
    formatBytes,
    getTotalModelSize,
    SAM2_MODELS
} from './webgpu';
export type {
    WebGPUCapabilities,
    ProviderState,
    ModelLoadProgressCallback,
    ProgressCallback,
    WebGPUProviderOptions
} from './webgpu';
