import { SegmentationRequest, SegmentationResponse } from './types';

/**
 * Abstract interface for segmentation providers.
 * Implementations can be remote (API) or local (WebGPU model).
 */
export interface SegmentationProvider {
    /**
     * Unique identifier for this provider.
     */
    readonly id: string;

    /**
     * Human-readable name for UI display.
     */
    readonly name: string;

    /**
     * Whether this provider is currently available.
     * May depend on network connectivity, API key, etc.
     */
    isAvailable(): Promise<boolean>;

    /**
     * Perform single-view segmentation.
     *
     * @param request - Segmentation request with image and keypoints
     * @returns Promise resolving to segmentation response with mask
     * @throws Error if segmentation fails
     */
    segmentSingleView(request: SegmentationRequest): Promise<SegmentationResponse>;

    /**
     * Perform multi-view segmentation (POC 2).
     * Optional - not all providers support this.
     *
     * @param requests - Array of segmentation requests from different views
     * @returns Promise resolving to fused segmentation response
     */
    segmentMultiView?(requests: SegmentationRequest[]): Promise<SegmentationResponse>;

    /**
     * Abort any in-progress segmentation request.
     */
    abort?(): void;
}

/**
 * Error thrown when segmentation fails.
 */
export class SegmentationError extends Error {
    constructor(
        message: string,
        public readonly code: 'NETWORK_ERROR' | 'TIMEOUT' | 'INVALID_RESPONSE' | 'SERVER_ERROR' | 'ABORTED',
        public readonly details?: unknown
    ) {
        super(message);
        this.name = 'SegmentationError';
    }
}
