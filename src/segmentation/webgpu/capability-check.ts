/**
 * WebGPU capability detection and VRAM estimation.
 */

export interface WebGPUCapabilities {
    /** Whether WebGPU is available in this browser */
    available: boolean;
    /** Reason if not available */
    unavailableReason?: string;
    /** GPU adapter info if available */
    adapterInfo?: GPUAdapterInfo;
    /** Estimated VRAM in bytes (may be 0 if not detectable) */
    estimatedVRAM: number;
    /** Whether this is likely a discrete GPU */
    isDiscreteGPU: boolean;
    /** Whether VRAM is considered low (<4GB) */
    isLowVRAM: boolean;
}

/**
 * Minimum VRAM threshold for warning (4GB in bytes)
 */
export const LOW_VRAM_THRESHOLD = 4 * 1024 * 1024 * 1024;

/**
 * Check WebGPU capabilities and estimate VRAM.
 * @returns Promise resolving to capability information
 */
export async function checkWebGPUCapabilities(): Promise<WebGPUCapabilities> {
    // Check if WebGPU API exists
    if (typeof navigator === 'undefined' || !navigator.gpu) {
        return {
            available: false,
            unavailableReason: 'WebGPU API not available in this browser',
            estimatedVRAM: 0,
            isDiscreteGPU: false,
            isLowVRAM: true
        };
    }

    try {
        // Request adapter
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });

        if (!adapter) {
            return {
                available: false,
                unavailableReason: 'No WebGPU adapter available',
                estimatedVRAM: 0,
                isDiscreteGPU: false,
                isLowVRAM: true
            };
        }

        // Get adapter info (use type assertion as this is a newer API)
        let adapterInfo: GPUAdapterInfo | undefined;
        if ('requestAdapterInfo' in adapter) {
            adapterInfo = await (adapter as unknown as { requestAdapterInfo: () => Promise<GPUAdapterInfo> }).requestAdapterInfo();
        }

        // Estimate VRAM from adapter limits
        // maxBufferSize gives us a hint about available memory
        const maxBufferSize = adapter.limits.maxBufferSize;
        
        // Heuristic: maxBufferSize is typically ~25% of VRAM
        // This is a rough estimate and varies by driver
        const estimatedVRAM = maxBufferSize * 4;

        // Detect discrete GPU based on adapter info
        const isDiscreteGPU = adapterInfo ? detectDiscreteGPU(adapterInfo) : false;

        // Check if VRAM is low
        const isLowVRAM = estimatedVRAM < LOW_VRAM_THRESHOLD && estimatedVRAM > 0;

        return {
            available: true,
            adapterInfo,
            estimatedVRAM,
            isDiscreteGPU,
            isLowVRAM
        };
    } catch (error) {
        return {
            available: false,
            unavailableReason: `WebGPU initialization failed: ${error instanceof Error ? error.message : String(error)}`,
            estimatedVRAM: 0,
            isDiscreteGPU: false,
            isLowVRAM: true
        };
    }
}

/**
 * Detect if the adapter is likely a discrete GPU based on vendor/architecture info.
 */
function detectDiscreteGPU(info: GPUAdapterInfo): boolean {
    const vendor = (info.vendor || '').toLowerCase();
    const architecture = (info.architecture || '').toLowerCase();
    const description = (info.description || '').toLowerCase();
    
    // Known discrete GPU vendors/keywords
    const discreteKeywords = [
        'nvidia',
        'geforce',
        'rtx',
        'gtx',
        'quadro',
        'amd',
        'radeon',
        'rx ',
        'vega'
    ];

    // Check description and vendor for discrete GPU indicators
    const combined = `${vendor} ${architecture} ${description}`;
    
    for (const keyword of discreteKeywords) {
        if (combined.includes(keyword)) {
            return true;
        }
    }

    // Intel Arc is discrete
    if (vendor.includes('intel') && (architecture.includes('arc') || description.includes('arc'))) {
        return true;
    }

    return false;
}

/**
 * Format VRAM size for display.
 */
export function formatVRAM(bytes: number): string {
    if (bytes === 0) return 'Unknown';
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) {
        return `${gb.toFixed(1)} GB`;
    }
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(0)} MB`;
}

/**
 * Get a user-friendly description of GPU capabilities.
 */
export function getCapabilityDescription(caps: WebGPUCapabilities): string {
    if (!caps.available) {
        return caps.unavailableReason || 'WebGPU not available';
    }

    const parts: string[] = [];
    
    if (caps.adapterInfo) {
        const vendor = caps.adapterInfo.vendor || 'Unknown';
        const arch = caps.adapterInfo.architecture || '';
        parts.push(`${vendor}${arch ? ` (${arch})` : ''}`);
    }

    parts.push(formatVRAM(caps.estimatedVRAM));
    
    if (caps.isDiscreteGPU) {
        parts.push('Discrete GPU');
    } else {
        parts.push('Integrated GPU');
    }

    return parts.join(' • ');
}
