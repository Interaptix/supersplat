import { Events } from '../events';

/**
 * SAM Mask Overlay - Renders SAM segmentation mask preview on the viewport.
 * 
 * Shows the mask as a semi-transparent colored overlay before applying
 * to Gaussian splat selection. Users can confirm or cancel the selection.
 */
class SamMaskOverlay {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private events: Events;
    private parent: HTMLElement;
    private currentMask: { mask: Uint8Array; width: number; height: number } | null = null;
    private visible = false;

    // Overlay color (semi-transparent blue)
    private maskColor = { r: 59, g: 130, b: 246, a: 0.5 }; // Tailwind blue-500

    constructor(events: Events, parent: HTMLElement) {
        this.events = events;
        this.parent = parent;

        // Create canvas overlay
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'sam-mask-overlay';
        this.canvas.classList.add('sam-mask-overlay', 'hidden');
        this.ctx = this.canvas.getContext('2d')!;
        parent.appendChild(this.canvas);

        // Register event handlers
        this.registerEventHandlers();
    }

    private registerEventHandlers(): void {
        // Listen for mask ready event
        this.events.on('sam.maskReady', (data: { mask: Uint8Array; width: number; height: number }) => {
            this.showMask(data);
        });

        // Listen for apply mask event
        this.events.on('sam.applyMask', () => {
            if (this.currentMask) {
                this.hide();
            }
        });

        // Listen for cancel mask event
        this.events.on('sam.cancelMask', () => {
            this.hide();
        });

        // Hide when SAM tool is deactivated
        this.events.on('sam.deactivated', () => {
            this.hide();
        });

        // Hide when SAM is cancelled
        this.events.on('sam.cancelled', () => {
            this.hide();
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            if (this.visible && this.currentMask) {
                this.renderMask();
            }
        });
    }

    /**
     * Show the mask overlay
     */
    private showMask(data: { mask: Uint8Array; width: number; height: number }): void {
        this.currentMask = data;
        this.visible = true;
        this.canvas.classList.remove('hidden');
        this.renderMask();

        // Fire event to notify UI that mask preview is ready
        this.events.fire('sam.maskPreviewReady', {
            width: data.width,
            height: data.height
        });
    }

    /**
     * Render the mask to the canvas
     */
    private renderMask(): void {
        if (!this.currentMask) return;

        const { mask, width: maskWidth, height: maskHeight } = this.currentMask;

        // Get parent dimensions for canvas sizing
        const parentRect = this.parent.getBoundingClientRect();
        const canvasWidth = parentRect.width;
        const canvasHeight = parentRect.height;

        // Resize canvas to match parent
        if (this.canvas.width !== canvasWidth || this.canvas.height !== canvasHeight) {
            this.canvas.width = canvasWidth;
            this.canvas.height = canvasHeight;
        }

        // Clear canvas
        this.ctx.clearRect(0, 0, canvasWidth, canvasHeight);

        // Create ImageData for the mask
        const imageData = this.ctx.createImageData(canvasWidth, canvasHeight);
        const pixels = imageData.data;

        // Calculate scale factors
        const scaleX = maskWidth / canvasWidth;
        const scaleY = maskHeight / canvasHeight;

        const { r, g, b, a } = this.maskColor;
        const alpha = Math.round(a * 255);

        // Render mask pixels with scaling
        for (let y = 0; y < canvasHeight; y++) {
            for (let x = 0; x < canvasWidth; x++) {
                // Map canvas coordinates to mask coordinates
                const maskX = Math.floor(x * scaleX);
                const maskY = Math.floor(y * scaleY);
                const maskIndex = maskY * maskWidth + maskX;

                // Check if this pixel is part of the mask
                if (mask[maskIndex] > 0) {
                    const pixelIndex = (y * canvasWidth + x) * 4;
                    pixels[pixelIndex] = r;     // R
                    pixels[pixelIndex + 1] = g; // G
                    pixels[pixelIndex + 2] = b; // B
                    pixels[pixelIndex + 3] = alpha; // A
                }
            }
        }

        // Draw the mask
        this.ctx.putImageData(imageData, 0, 0);

        // Add a subtle outline effect for better visibility
        this.renderMaskOutline(mask, maskWidth, maskHeight, canvasWidth, canvasHeight);
    }

    /**
     * Render a subtle outline around the mask edges
     */
    private renderMaskOutline(
        mask: Uint8Array,
        maskWidth: number,
        maskHeight: number,
        canvasWidth: number,
        canvasHeight: number
    ): void {
        const scaleX = maskWidth / canvasWidth;
        const scaleY = maskHeight / canvasHeight;

        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.lineWidth = 2;

        // Simple edge detection - find pixels where mask changes
        this.ctx.beginPath();

        for (let y = 1; y < canvasHeight - 1; y++) {
            for (let x = 1; x < canvasWidth - 1; x++) {
                const maskX = Math.floor(x * scaleX);
                const maskY = Math.floor(y * scaleY);
                const maskIndex = maskY * maskWidth + maskX;

                if (mask[maskIndex] > 0) {
                    // Check if this is an edge pixel (any neighbor is 0)
                    const neighbors = [
                        Math.floor((y - 1) * scaleY) * maskWidth + Math.floor(x * scaleX),
                        Math.floor((y + 1) * scaleY) * maskWidth + Math.floor(x * scaleX),
                        Math.floor(y * scaleY) * maskWidth + Math.floor((x - 1) * scaleX),
                        Math.floor(y * scaleY) * maskWidth + Math.floor((x + 1) * scaleX)
                    ];

                    const isEdge = neighbors.some(ni => mask[ni] === 0);
                    if (isEdge) {
                        this.ctx.rect(x, y, 1, 1);
                    }
                }
            }
        }

        this.ctx.stroke();
    }

    /**
     * Hide the overlay
     */
    hide(): void {
        this.visible = false;
        this.canvas.classList.add('hidden');
        this.currentMask = null;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Check if overlay is currently visible
     */
    isVisible(): boolean {
        return this.visible;
    }

    /**
     * Get the current mask data
     */
    getCurrentMask(): { mask: Uint8Array; width: number; height: number } | null {
        return this.currentMask;
    }

    /**
     * Set the mask overlay color
     */
    setColor(r: number, g: number, b: number, a: number): void {
        this.maskColor = { r, g, b, a };
        if (this.visible && this.currentMask) {
            this.renderMask();
        }
    }

    /**
     * Dispose of resources
     */
    dispose(): void {
        this.hide();
        if (this.canvas.parentElement) {
            this.canvas.parentElement.removeChild(this.canvas);
        }
    }
}

// CSS styles for the mask overlay
const samMaskOverlayStyles = `
.sam-mask-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 50;
}

.sam-mask-overlay.hidden {
    display: none;
}
`;

// Inject styles
const styleElement = document.createElement('style');
styleElement.textContent = samMaskOverlayStyles;
document.head.appendChild(styleElement);

export { SamMaskOverlay };
