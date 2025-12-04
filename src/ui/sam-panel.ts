import { Container, Button, Label, Progress } from '@playcanvas/pcui';
import { Events } from '../events';
import { SegmentationPoint, MaskCandidate } from '../segmentation/types';
import { WebGPUCapabilities, ProviderState, resizeMaskSmooth } from '../segmentation/webgpu';

/**
 * SAM Panel - Redesigned floating panel for SAM segmentation tool.
 * 
 * Features:
 * - Large layout with image preview canvas
 * - Automatic segmentation on each click (no manual "Select" button)
 * - Apply/Cancel buttons for mask confirmation
 * - WebGPU status badge
 * - Performance stats display
 * - Multi-mask selection UI showing all 3 mask candidates
 */
class SamPanel extends Container {
    private fgLabel: Label;
    private bgLabel: Label;
    private undoButton: Button;
    private clearButton: Button;
    private statusContainer: Container;
    private statusLabel: Label;
    private progressBar: Progress;
    private pointsContainer: Container;
    private buttonsContainer: Container;
    private previewButtonsContainer: Container;
    private applyButton: Button;
    private cancelButton: Button;
    private debugExportButton: Button;
    private instructions: Label;
    private warningContainer: Container;
    private warningLabel: Label;
    private events: Events;
    private providerReady = false;
    private hasPendingMask = false;
    
    // New UI elements for redesigned panel
    private headerContainer: Container;
    private webgpuBadge: Label;
    private imagePreviewContainer: Container;
    private imagePreviewCanvas: HTMLCanvasElement;
    private imagePreviewCtx: CanvasRenderingContext2D | null = null;
    private statsContainer: Container;
    private statsLabel: Label;
    private processingIndicator: Label;

    // Multi-mask selection UI elements
    private mainContentContainer: Container;
    private maskSelectorContainer: Container;
    private maskSelectorTitle: Label;
    private maskCandidates: Array<{
        container: Container;
        canvas: HTMLCanvasElement;
        ctx: CanvasRenderingContext2D | null;
        scoreLabel: Label;
        indexLabel: Label;
    }> = [];
    private selectedMaskIndex: number = 0;
    private allMaskCandidates: MaskCandidate[] = [];

    // Model Inputs panel UI elements (left side)
    private modelInputsContainer: Container;
    private modelInputsTitle: Label;
    private inputImageContainer: Container;
    private inputImageCanvas: HTMLCanvasElement;
    private inputImageCtx: CanvasRenderingContext2D | null = null;
    private inputImageLabel: Label;
    private prevMaskContainer: Container;
    private prevMaskCanvas: HTMLCanvasElement;
    private prevMaskCtx: CanvasRenderingContext2D | null = null;
    private prevMaskLabel: Label;
    private pointPromptsContainer: Container;
    private pointPromptsTitle: Label;
    private pointPromptsList: Container;

    // State for preview compositing
    private capturedImageData: ImageData | null = null;
    private capturedWidth: number = 0;
    private capturedHeight: number = 0;
    private currentMask: Float32Array | null = null;
    private currentMaskWidth: number = 0;
    private currentMaskHeight: number = 0;
    private currentPoints: SegmentationPoint[] = [];

    constructor(events: Events, args = {}) {
        args = {
            ...args,
            id: 'sam-panel',
            class: 'sam-panel',
            hidden: true
        };

        super(args);
        this.events = events;

        // Header with title and WebGPU badge
        this.headerContainer = new Container({
            class: 'sam-panel-header'
        });

        const titleLabel = new Label({
            class: 'sam-panel-title',
            text: 'SAM2 Segmentation Tool'
        });

        this.webgpuBadge = new Label({
            class: 'sam-panel-webgpu-badge',
            text: 'WebGPU'
        });

        this.headerContainer.append(titleLabel);
        this.headerContainer.append(this.webgpuBadge);

        // Status container (for model loading progress)
        this.statusContainer = new Container({
            class: 'sam-panel-status',
            hidden: true
        });

        this.statusLabel = new Label({
            class: 'sam-panel-status-label',
            text: 'Initializing SAM...'
        });

        this.progressBar = new Progress({
            class: 'sam-panel-progress'
        });
        this.progressBar.value = 0;

        this.statusContainer.append(this.statusLabel);
        this.statusContainer.append(this.progressBar);

        // Warning container (for low VRAM, WebGPU unavailable)
        this.warningContainer = new Container({
            class: 'sam-panel-warning',
            hidden: true
        });

        this.warningLabel = new Label({
            class: 'sam-panel-warning-label',
            text: ''
        });

        this.warningContainer.append(this.warningLabel);

        // Image preview container with canvas
        this.imagePreviewContainer = new Container({
            class: 'sam-panel-preview-container'
        });

        // Create native canvas element for image preview (larger for better visibility)
        this.imagePreviewCanvas = document.createElement('canvas');
        this.imagePreviewCanvas.className = 'sam-panel-preview-canvas';
        this.imagePreviewCanvas.width = 800;
        this.imagePreviewCanvas.height = 450;
        this.imagePreviewCtx = this.imagePreviewCanvas.getContext('2d');

        // Processing indicator overlay
        this.processingIndicator = new Label({
            class: 'sam-panel-processing',
            text: 'Processing...',
            hidden: true
        });

        // Add canvas to container's DOM element
        const previewElement = this.imagePreviewContainer.dom;
        previewElement.appendChild(this.imagePreviewCanvas);
        this.imagePreviewContainer.append(this.processingIndicator);

        // Add click handler to image preview canvas for adding points
        this.imagePreviewCanvas.addEventListener('click', (e: MouseEvent) => this.handlePreviewClick(e));
        this.imagePreviewCanvas.addEventListener('contextmenu', (e: MouseEvent) => {
            e.preventDefault();
            e.stopPropagation();
            this.handlePreviewClick(e, true); // Right-click = background point
        });
        this.imagePreviewCanvas.style.cursor = 'crosshair';

        // Create mask selector panel (right side - for multi-mask debug/selection)
        this.maskSelectorContainer = new Container({
            class: 'sam-panel-mask-selector',
            hidden: true
        });

        this.maskSelectorTitle = new Label({
            class: 'sam-panel-mask-selector-title',
            text: 'Mask Candidates'
        });
        this.maskSelectorContainer.append(this.maskSelectorTitle);

        // Create 3 mask candidate thumbnails
        const maskLabels = ['Tight (0)', 'Medium (1)', 'Broad (2)'];
        for (let i = 0; i < 3; i++) {
            const candidateContainer = new Container({
                class: 'sam-panel-mask-candidate'
            });
            candidateContainer.class.add(`sam-panel-mask-candidate-${i}`);

            const indexLabel = new Label({
                class: 'sam-panel-mask-candidate-index',
                text: maskLabels[i]
            });

            const canvas = document.createElement('canvas');
            canvas.className = 'sam-panel-mask-candidate-canvas';
            canvas.width = 128;
            canvas.height = 72;
            const ctx = canvas.getContext('2d');

            const scoreLabel = new Label({
                class: 'sam-panel-mask-candidate-score',
                text: 'IoU: --'
            });

            candidateContainer.append(indexLabel);
            candidateContainer.dom.appendChild(canvas);
            candidateContainer.append(scoreLabel);

            // Click handler to select this mask
            const maskIndex = i;
            canvas.addEventListener('click', () => {
                this.selectMaskCandidate(maskIndex);
            });
            candidateContainer.dom.addEventListener('click', (e) => {
                if (e.target !== canvas) {
                    this.selectMaskCandidate(maskIndex);
                }
            });

            this.maskCandidates.push({
                container: candidateContainer,
                canvas,
                ctx,
                scoreLabel,
                indexLabel
            });

            this.maskSelectorContainer.append(candidateContainer);
        }

        // Create Model Inputs panel (left side - shows what goes into the model)
        this.modelInputsContainer = new Container({
            class: 'sam-panel-model-inputs'
        });

        this.modelInputsTitle = new Label({
            class: 'sam-panel-model-inputs-title',
            text: 'Model Inputs'
        });
        this.modelInputsContainer.append(this.modelInputsTitle);

        // Input image thumbnail container
        this.inputImageContainer = new Container({
            class: 'sam-panel-input-image-container'
        });

        this.inputImageLabel = new Label({
            class: 'sam-panel-input-image-label',
            text: 'Input Image'
        });

        this.inputImageCanvas = document.createElement('canvas');
        this.inputImageCanvas.className = 'sam-panel-input-image-canvas';
        this.inputImageCanvas.width = 128;
        this.inputImageCanvas.height = 128;
        this.inputImageCtx = this.inputImageCanvas.getContext('2d');

        this.inputImageContainer.append(this.inputImageLabel);
        this.inputImageContainer.dom.appendChild(this.inputImageCanvas);
        this.modelInputsContainer.append(this.inputImageContainer);

        // Previous mask thumbnail container
        this.prevMaskContainer = new Container({
            class: 'sam-panel-prev-mask-container'
        });

        this.prevMaskLabel = new Label({
            class: 'sam-panel-prev-mask-label',
            text: 'Previous Mask'
        });

        this.prevMaskCanvas = document.createElement('canvas');
        this.prevMaskCanvas.className = 'sam-panel-prev-mask-canvas';
        this.prevMaskCanvas.width = 128;
        this.prevMaskCanvas.height = 72;
        this.prevMaskCtx = this.prevMaskCanvas.getContext('2d');

        this.prevMaskContainer.append(this.prevMaskLabel);
        this.prevMaskContainer.dom.appendChild(this.prevMaskCanvas);
        this.modelInputsContainer.append(this.prevMaskContainer);

        // Point prompts list container
        this.pointPromptsContainer = new Container({
            class: 'sam-panel-point-prompts-container'
        });

        this.pointPromptsTitle = new Label({
            class: 'sam-panel-point-prompts-title',
            text: 'Point Prompts'
        });

        this.pointPromptsList = new Container({
            class: 'sam-panel-point-prompts-list'
        });

        this.pointPromptsContainer.append(this.pointPromptsTitle);
        this.pointPromptsContainer.append(this.pointPromptsList);
        this.modelInputsContainer.append(this.pointPromptsContainer);

        // Main content container (holds inputs, preview, and mask selector side by side)
        this.mainContentContainer = new Container({
            class: 'sam-panel-main-content'
        });
        this.mainContentContainer.append(this.modelInputsContainer);
        this.mainContentContainer.append(this.imagePreviewContainer);
        this.mainContentContainer.append(this.maskSelectorContainer);

        // Point count labels
        this.pointsContainer = new Container({
            class: 'sam-panel-points'
        });

        this.fgLabel = new Label({
            class: 'sam-panel-fg',
            text: 'FG: 0'
        });

        this.bgLabel = new Label({
            class: 'sam-panel-bg',
            text: 'BG: 0'
        });

        this.pointsContainer.append(this.fgLabel);
        this.pointsContainer.append(this.bgLabel);

        // Stats container for timing info
        this.statsContainer = new Container({
            class: 'sam-panel-stats',
            hidden: true
        });

        this.statsLabel = new Label({
            class: 'sam-panel-stats-label',
            text: ''
        });

        this.statsContainer.append(this.statsLabel);

        // Instructions
        this.instructions = new Label({
            class: 'sam-panel-instructions',
            text: 'Click to segment • Left: foreground • Right: background'
        });

        // Action buttons (Clear, Undo only - no Select button)
        this.buttonsContainer = new Container({
            class: 'sam-panel-buttons'
        });

        this.clearButton = new Button({
            class: 'sam-panel-button',
            text: 'Clear',
            icon: 'E120'
        });

        this.undoButton = new Button({
            class: 'sam-panel-button',
            text: 'Undo',
            icon: 'E114'
        });

        this.buttonsContainer.append(this.clearButton);
        this.buttonsContainer.append(this.undoButton);

        // Preview buttons container (Apply/Cancel for mask preview)
        this.previewButtonsContainer = new Container({
            class: 'sam-panel-preview-buttons',
            hidden: true
        });

        this.applyButton = new Button({
            class: 'sam-panel-apply',
            text: 'Apply Selection',
            icon: 'E149'
        });

        this.cancelButton = new Button({
            class: 'sam-panel-cancel',
            text: 'Cancel',
            icon: 'E129'
        });

        // Debug export button
        this.debugExportButton = new Button({
            class: 'sam-panel-debug-export',
            text: 'Debug Export',
            icon: 'E228'
        });

        this.previewButtonsContainer.append(this.cancelButton);
        this.previewButtonsContainer.append(this.applyButton);
        this.previewButtonsContainer.append(this.debugExportButton);

        // Assemble panel
        this.append(this.headerContainer);
        this.append(this.statusContainer);
        this.append(this.warningContainer);
        this.append(this.mainContentContainer);
        this.append(this.pointsContainer);
        this.append(this.statsContainer);
        this.append(this.instructions);
        this.append(this.buttonsContainer);
        this.append(this.previewButtonsContainer);

        // Button event handlers
        this.clearButton.on('click', () => {
            events.fire('sam.clearPoints');
            this.clearImagePreview();
            this.statsContainer.hidden = true;
        });

        this.undoButton.on('click', () => {
            events.fire('sam.undoPoint');
        });

        // Preview button event handlers
        this.applyButton.on('click', () => {
            events.fire('sam.applyMask');
        });

        this.cancelButton.on('click', () => {
            events.fire('sam.cancelMask');
        });

        this.debugExportButton.on('click', () => {
            this.exportDebugData();
        });

        // Register all event handlers
        this.registerEventHandlers();
    }

    private registerEventHandlers(): void {
        // Update point counts when points change and redraw preview with points
        this.events.on('sam.pointsChanged', (points: SegmentationPoint[]) => {
            const fgCount = points.filter(p => p.type === 'fg').length;
            const bgCount = points.filter(p => p.type === 'bg').length;
            this.fgLabel.text = `FG: ${fgCount}`;
            this.bgLabel.text = `BG: ${bgCount}`;

            // Enable/disable buttons based on point count
            this.undoButton.enabled = points.length > 0;
            this.clearButton.enabled = points.length > 0;

            // Store points and redraw preview to show them
            this.currentPoints = [...points];
            if (this.capturedImageData) {
                this.redrawPreview();
            }

            // Update Model Inputs panel
            this.updateInputImageThumbnail();
            this.updatePointPromptsList();
        });

        // Show/hide panel based on SAM tool activation
        this.events.on('sam.activated', () => {
            this.hidden = false;
            // Reset button states
            this.undoButton.enabled = false;
            this.clearButton.enabled = false;
            this.fgLabel.text = 'FG: 0';
            this.bgLabel.text = 'BG: 0';
            this.clearImagePreview();
            this.statsContainer.hidden = true;

            // Show normal UI if provider is ready, otherwise show status
            if (this.providerReady) {
                this.showReadyState();
            }
        });

        this.events.on('sam.deactivated', () => {
            this.hidden = true;
        });

        this.events.on('sam.cancelled', () => {
            this.hidden = true;
        });

        // Handle image captured - display in preview canvas
        this.events.on('sam.imageCaptured', (data: { image: Uint8Array; width: number; height: number }) => {
            this.displayCapturedImage(data.image, data.width, data.height);
        });

        // Handle model loading progress
        this.events.on('sam.modelLoadProgress', (data: { loaded: number; total: number; stage: string }) => {
            this.showLoadingState(data.loaded, data.total, data.stage);
        });

        // Handle provider status changes
        this.events.on('sam.providerStatusChanged', (data: { state: ProviderState; details?: string }) => {
            this.handleProviderStateChange(data.state, data.details);
        });

        // Handle provider ready
        this.events.on('sam.providerReady', () => {
            this.providerReady = true;
            this.webgpuBadge.class.add('sam-panel-webgpu-ready');
            this.showReadyState();
        });

        // Handle capabilities check
        this.events.on('sam.capabilities', (capabilities: WebGPUCapabilities) => {
            if (!capabilities.available) {
                this.webgpuBadge.class.add('sam-panel-webgpu-unavailable');
                this.showWarning(`WebGPU not available: ${capabilities.unavailableReason || 'Unknown reason'}`);
            }
        });

        // Handle low VRAM warning
        this.events.on('sam.lowVramWarning', (data: { estimated: number; threshold: number }) => {
            const estimatedGB = (data.estimated / (1024 * 1024 * 1024)).toFixed(1);
            this.showWarning(`Low GPU memory (~${estimatedGB}GB). Performance may be affected.`);
        });

        // Handle initialization error
        this.events.on('sam.initError', (error: string) => {
            this.showError(error);
        });

        // Handle segmentation start
        this.events.on('sam.segmentStart', () => {
            this.processingIndicator.hidden = false;
            this.buttonsContainer.hidden = true;
        });

        // Handle segmentation complete with stats
        this.events.on('sam.segmentComplete', (data?: { 
            hasPendingMask?: boolean; 
            stats?: { totalTime: number; encodeTime: number; decodeTime: number } 
        }) => {
            this.processingIndicator.hidden = true;

            // Display stats if available
            if (data?.stats) {
                this.statsLabel.text = `Total: ${data.stats.totalTime}ms | Encode: ${data.stats.encodeTime}ms | Decode: ${data.stats.decodeTime}ms`;
                this.statsContainer.hidden = false;
            }

            // If there's a pending mask, show preview mode
            if (data?.hasPendingMask) {
                this.hasPendingMask = true;
                this.showPreviewMode();
            } else {
                this.buttonsContainer.hidden = false;
            }
        });

        // Handle mask ready - overlay on preview canvas and populate multi-mask selector
        // Note: segmentation service sends Uint8Array (binary 0/255), we convert to smooth Float32Array
        this.events.on('sam.maskReady', (data: { 
            mask: Uint8Array; 
            width: number; 
            height: number;
            allMasks?: MaskCandidate[];
            selectedMaskIndex?: number;
        }) => {
            // Convert binary Uint8Array mask to smooth Float32Array for visualization
            // Use resizeMaskSmooth to get smooth edges via bilinear interpolation
            const targetWidth = this.capturedWidth || data.width;
            const targetHeight = this.capturedHeight || data.height;
            const smoothMask = resizeMaskSmooth(data.mask, data.width, data.height, targetWidth, targetHeight);
            
            this.overlayMaskOnPreview(smoothMask, targetWidth, targetHeight);

            // Populate multi-mask selector if allMasks available
            if (data.allMasks && data.allMasks.length > 0) {
                this.allMaskCandidates = data.allMasks;
                this.selectedMaskIndex = data.selectedMaskIndex ?? 0;
                this.showMaskSelector(data.width, data.height);
            }
        });

        // Handle mask applied - return to normal state
        this.events.on('sam.maskApplied', () => {
            this.hasPendingMask = false;
            this.hidePreviewMode();
            this.clearImagePreview();
        });

        // Handle mask cancelled - return to normal state
        this.events.on('sam.maskCancelled', () => {
            this.hasPendingMask = false;
            this.hidePreviewMode();
        });

        this.events.on('sam.segmentError', (error: string) => {
            this.processingIndicator.hidden = true;
            this.buttonsContainer.hidden = false;
            this.showError(error);
            setTimeout(() => {
                if (!this.hidden) {
                    this.warningContainer.hidden = true;
                }
            }, 5000);
        });

        // Handle pre-encoding events (encoding happens automatically when panel opens)
        this.events.on('sam.encodingStart', () => {
            this.showEncodingState();
        });

        this.events.on('sam.encodingComplete', (data: { encodeTime: number }) => {
            this.showEncodingComplete(data.encodeTime);
        });

        this.events.on('sam.encodingError', (error: string) => {
            this.showError(`Encoding failed: ${error}`);
        });
    }

    /**
     * Display captured image in preview canvas and store for later redraws
     */
    private displayCapturedImage(image: Uint8Array, width: number, height: number): void {
        console.log('[SAM Panel] displayCapturedImage called:', { 
            imageLength: image.length, 
            width, 
            height,
            expectedLength: width * height * 4,
            hasCtx: !!this.imagePreviewCtx
        });

        if (!this.imagePreviewCtx) {
            console.warn('[SAM Panel] No canvas context available');
            return;
        }

        // Create ImageData from RGBA bytes - copy to new ArrayBuffer to avoid SharedArrayBuffer issues
        const rgbaData = new Uint8ClampedArray(image.length);
        rgbaData.set(image);
        const imageData = new ImageData(rgbaData, width, height);

        // Debug: Check if image has any non-zero pixels
        let nonZeroPixels = 0;
        for (let i = 0; i < Math.min(1000, image.length); i += 4) {
            if (image[i] > 0 || image[i + 1] > 0 || image[i + 2] > 0) {
                nonZeroPixels++;
            }
        }
        console.log('[SAM Panel] Image data check - non-zero pixels in first 250:', nonZeroPixels);

        // Store the captured image for later redraws
        this.capturedImageData = imageData;
        this.capturedWidth = width;
        this.capturedHeight = height;

        // Redraw the full preview (image + mask + points)
        this.redrawPreview();

        // Update Model Inputs panel with the captured image
        this.updateInputImageThumbnail();
    }

    /**
     * Overlay mask on preview canvas and store for later redraws
     */
    private overlayMaskOnPreview(mask: Float32Array, maskWidth: number, maskHeight: number): void {
        // Store mask data for later redraws
        this.currentMask = mask;
        this.currentMaskWidth = maskWidth;
        this.currentMaskHeight = maskHeight;

        // Redraw the full preview (image + mask + points)
        this.redrawPreview();

        // Update Model Inputs panel with the previous mask thumbnail
        this.updatePrevMaskThumbnail();
    }

    /**
     * Redraw the entire preview canvas with all layers:
     * 1. Base captured image
     * 2. Mask overlay (if any)
     * 3. Points (if any)
     */
    private redrawPreview(): void {
        if (!this.imagePreviewCtx) return;

        const canvas = this.imagePreviewCanvas;
        const ctx = this.imagePreviewCtx;

        // Clear canvas
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Layer 1: Draw captured image
        if (this.capturedImageData) {
            const width = this.capturedWidth;
            const height = this.capturedHeight;

            // Calculate scaling to fit
            const scale = Math.min(canvas.width / width, canvas.height / height);
            const scaledWidth = width * scale;
            const scaledHeight = height * scale;
            const offsetX = (canvas.width - scaledWidth) / 2;
            const offsetY = (canvas.height - scaledHeight) / 2;

            // Create temporary canvas to hold full-size image
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const tempCtx = tempCanvas.getContext('2d');
            if (tempCtx) {
                tempCtx.putImageData(this.capturedImageData, 0, 0);
                // Draw scaled to preview canvas
                ctx.drawImage(tempCanvas, offsetX, offsetY, scaledWidth, scaledHeight);
            }
            console.warn('[SAM Panel] rendered background image to preview');
        } else {
            console.warn('[SAM Panel] No captured image data to draw');
        }

        // Layer 2: Draw mask overlay (GREEN, using globalAlpha like original SAM2 repo)
        if (this.currentMask && this.currentMaskWidth > 0 && this.currentMaskHeight > 0) {
            const maskWidth = this.currentMaskWidth;
            const maskHeight = this.currentMaskHeight;

            // Calculate same scaling as image
            const scale = Math.min(canvas.width / maskWidth, canvas.height / maskHeight);
            const scaledWidth = maskWidth * scale;
            const scaledHeight = maskHeight * scale;
            const offsetX = (canvas.width - scaledWidth) / 2;
            const offsetY = (canvas.height - scaledHeight) / 2;

            // Create mask overlay image with GREEN color (matching original SAM2 repo)
            const maskImageData = new ImageData(maskWidth, maskHeight);
            const maskData = maskImageData.data;

            for (let i = 0; i < this.currentMask.length; i++) {
                const alpha = Math.max(0, Math.min(1, this.currentMask[i]));
                const idx = i * 4;
                // GREEN overlay for mask (matching original repo: #22c55e)
                maskData[idx] = 34;      // R
                maskData[idx + 1] = 197; // G  
                maskData[idx + 2] = 94;  // B
                maskData[idx + 3] = Math.floor(alpha * 255); // Full alpha where mask exists
            }

            // Create temp canvas for mask
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = maskWidth;
            tempCanvas.height = maskHeight;
            const tempCtx = tempCanvas.getContext('2d');
            if (tempCtx) {
                tempCtx.putImageData(maskImageData, 0, 0);
                // Overlay mask with 70% opacity (like original SAM2 repo)
                ctx.globalAlpha = 0.7;
                ctx.drawImage(tempCanvas, offsetX, offsetY, scaledWidth, scaledHeight);
                ctx.globalAlpha = 1.0;
            }
        }

        // Layer 3: Draw points
        this.drawPointsOnPreview();
    }

    /**
     * Draw selection points on the preview canvas
     * Scales point coordinates from viewport to preview canvas size
     */
    private drawPointsOnPreview(): void {
        if (!this.imagePreviewCtx || !this.capturedImageData || this.currentPoints.length === 0) return;

        const canvas = this.imagePreviewCanvas;
        const ctx = this.imagePreviewCtx;

        // Calculate scaling from viewport to preview canvas
        const viewportWidth = this.capturedWidth;
        const viewportHeight = this.capturedHeight;

        const scale = Math.min(canvas.width / viewportWidth, canvas.height / viewportHeight);
        const scaledWidth = viewportWidth * scale;
        const scaledHeight = viewportHeight * scale;
        const offsetX = (canvas.width - scaledWidth) / 2;
        const offsetY = (canvas.height - scaledHeight) / 2;

        const pointRadius = 6; // Smaller radius for preview

        this.currentPoints.forEach((point, index) => {
            // Scale point coordinates from viewport to preview canvas
            const x = point.x * scale + offsetX;
            const y = point.y * scale + offsetY;

            // Outer circle (white border)
            ctx.beginPath();
            ctx.arc(x, y, pointRadius + 2, 0, Math.PI * 2);
            ctx.fillStyle = 'white';
            ctx.fill();
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Inner circle (color indicates type: green = fg, red = bg)
            ctx.beginPath();
            ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
            ctx.fillStyle = point.type === 'fg' ? '#22c55e' : '#ef4444';
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Number label
            ctx.fillStyle = 'white';
            ctx.font = 'bold 8px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText((index + 1).toString(), x, y);
        });
    }

    /**
     * Draw selection points on any canvas context
     * Scales point coordinates from viewport to target canvas size
     * Used for drawing points on mask candidate thumbnails
     */
    private drawPointsOnCanvas(
        ctx: CanvasRenderingContext2D,
        canvasWidth: number,
        canvasHeight: number,
        viewportWidth: number,
        viewportHeight: number
    ): void {
        if (this.currentPoints.length === 0) return;

        // Calculate scaling from viewport to canvas
        const scale = Math.min(canvasWidth / viewportWidth, canvasHeight / viewportHeight);
        const scaledWidth = viewportWidth * scale;
        const scaledHeight = viewportHeight * scale;
        const offsetX = (canvasWidth - scaledWidth) / 2;
        const offsetY = (canvasHeight - scaledHeight) / 2;

        // Use smaller radius for thumbnails
        const pointRadius = 3;

        this.currentPoints.forEach((point, index) => {
            // Scale point coordinates from viewport to canvas
            const x = point.x * scale + offsetX;
            const y = point.y * scale + offsetY;

            // Outer circle (white border)
            ctx.beginPath();
            ctx.arc(x, y, pointRadius + 1.5, 0, Math.PI * 2);
            ctx.fillStyle = 'white';
            ctx.fill();
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 0.5;
            ctx.stroke();

            // Inner circle (color indicates type: green = fg, red = bg)
            ctx.beginPath();
            ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
            ctx.fillStyle = point.type === 'fg' ? '#22c55e' : '#ef4444';
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Number label (only show if radius is large enough)
            if (pointRadius >= 3) {
                ctx.fillStyle = 'white';
                ctx.font = 'bold 6px system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText((index + 1).toString(), x, y);
            }
        });
    }

    /**
     * Update the input image thumbnail in the Model Inputs panel
     * Draws the captured image scaled to 128×128 with points overlay
     */
    private updateInputImageThumbnail(): void {
        if (!this.inputImageCtx || !this.capturedImageData) {
            // Clear to placeholder state
            if (this.inputImageCtx) {
                const ctx = this.inputImageCtx;
                ctx.fillStyle = '#0a0a0a';
                ctx.fillRect(0, 0, 128, 128);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                ctx.font = '9px system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('No image', 64, 64);
            }
            return;
        }

        const ctx = this.inputImageCtx;
        const canvas = this.inputImageCanvas;
        const thumbSize = 128;

        // Clear canvas
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, thumbSize, thumbSize);

        // Create temp canvas for the captured image
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.capturedWidth;
        tempCanvas.height = this.capturedHeight;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) return;
        tempCtx.putImageData(this.capturedImageData, 0, 0);

        // Calculate scaling to fit in square thumbnail (letterbox/pillarbox)
        const scale = Math.min(thumbSize / this.capturedWidth, thumbSize / this.capturedHeight);
        const scaledW = this.capturedWidth * scale;
        const scaledH = this.capturedHeight * scale;
        const offsetX = (thumbSize - scaledW) / 2;
        const offsetY = (thumbSize - scaledH) / 2;

        // Draw scaled image
        ctx.drawImage(tempCanvas, offsetX, offsetY, scaledW, scaledH);

        // Draw points overlay on the thumbnail
        if (this.currentPoints.length > 0) {
            const pointRadius = 3;
            this.currentPoints.forEach((point, index) => {
                // Scale point from viewport coords to thumbnail coords
                const x = point.x * scale + offsetX;
                const y = point.y * scale + offsetY;

                // Outer circle
                ctx.beginPath();
                ctx.arc(x, y, pointRadius + 1.5, 0, Math.PI * 2);
                ctx.fillStyle = 'white';
                ctx.fill();

                // Inner circle
                ctx.beginPath();
                ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
                ctx.fillStyle = point.type === 'fg' ? '#22c55e' : '#ef4444';
                ctx.fill();

                // Number label
                ctx.fillStyle = 'white';
                ctx.font = 'bold 6px system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText((index + 1).toString(), x, y);
            });
        }

        // Update label to show dimensions
        this.inputImageLabel.text = `Input (${this.capturedWidth}×${this.capturedHeight})`;
    }

    /**
     * Update the previous mask thumbnail in the Model Inputs panel
     * Draws the current mask as a grayscale preview
     */
    private updatePrevMaskThumbnail(): void {
        if (!this.prevMaskCtx) return;

        const ctx = this.prevMaskCtx;
        const thumbWidth = 128;
        const thumbHeight = 72;

        // Clear canvas
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, thumbWidth, thumbHeight);

        if (!this.currentMask || this.currentMaskWidth === 0 || this.currentMaskHeight === 0) {
            // Show "No mask" placeholder
            ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.font = '9px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('No mask', thumbWidth / 2, thumbHeight / 2);
            this.prevMaskLabel.text = 'Previous Mask';
            return;
        }

        // Create grayscale mask image
        const maskImageData = new ImageData(this.currentMaskWidth, this.currentMaskHeight);
        const pixels = maskImageData.data;

        for (let i = 0; i < this.currentMask.length; i++) {
            const alpha = Math.max(0, Math.min(1, this.currentMask[i]));
            const value = Math.floor(alpha * 255);
            const idx = i * 4;
            pixels[idx] = value;     // R
            pixels[idx + 1] = value; // G
            pixels[idx + 2] = value; // B
            pixels[idx + 3] = 255;   // A (fully opaque)
        }

        // Create temp canvas for the mask
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.currentMaskWidth;
        tempCanvas.height = this.currentMaskHeight;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) return;
        tempCtx.putImageData(maskImageData, 0, 0);

        // Calculate scaling to fit in thumbnail (letterbox)
        const scale = Math.min(thumbWidth / this.currentMaskWidth, thumbHeight / this.currentMaskHeight);
        const scaledW = this.currentMaskWidth * scale;
        const scaledH = this.currentMaskHeight * scale;
        const offsetX = (thumbWidth - scaledW) / 2;
        const offsetY = (thumbHeight - scaledH) / 2;

        // Draw scaled mask
        ctx.drawImage(tempCanvas, offsetX, offsetY, scaledW, scaledH);

        // Update label to show dimensions
        this.prevMaskLabel.text = `Prev Mask (${this.currentMaskWidth}×${this.currentMaskHeight})`;
    }

    /**
     * Update the point prompts list in the Model Inputs panel
     * Shows each point with coordinates and type (fg/bg)
     */
    private updatePointPromptsList(): void {
        // Clear existing items
        const listDom = this.pointPromptsList.dom;
        while (listDom.firstChild) {
            listDom.removeChild(listDom.firstChild);
        }

        if (this.currentPoints.length === 0) {
            // Show "No points" placeholder
            const noPointsEl = document.createElement('div');
            noPointsEl.className = 'sam-panel-no-points';
            noPointsEl.textContent = 'Click to add points';
            listDom.appendChild(noPointsEl);
            return;
        }

        // Create item for each point
        this.currentPoints.forEach((point, index) => {
            const itemEl = document.createElement('div');
            itemEl.className = `sam-panel-point-prompt-item ${point.type}`;

            const indexEl = document.createElement('span');
            indexEl.className = 'sam-panel-point-prompt-index';
            indexEl.textContent = `#${index + 1}`;

            const coordsEl = document.createElement('span');
            coordsEl.className = 'sam-panel-point-prompt-coords';
            coordsEl.textContent = `(${Math.round(point.x)}, ${Math.round(point.y)})`;

            const typeEl = document.createElement('span');
            typeEl.className = `sam-panel-point-prompt-type ${point.type}`;
            typeEl.textContent = point.type;

            itemEl.appendChild(indexEl);
            itemEl.appendChild(coordsEl);
            itemEl.appendChild(typeEl);
            listDom.appendChild(itemEl);
        });
    }

    /**
     * Clear the Model Inputs panel to placeholder state
     */
    private clearModelInputsPanel(): void {
        this.updateInputImageThumbnail();
        this.updatePrevMaskThumbnail();
        this.updatePointPromptsList();
    }

    /**
     * Clear image preview canvas and reset all state
     */
    private clearImagePreview(): void {
        console.log('[SAM Panel] clearImagePreview called');
        // Reset all stored state
        this.capturedImageData = null;
        this.capturedWidth = 0;
        this.capturedHeight = 0;
        this.currentMask = null;
        this.currentMaskWidth = 0;
        this.currentMaskHeight = 0;
        this.currentPoints = [];

        if (!this.imagePreviewCtx) return;
        const ctx = this.imagePreviewCtx;
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, this.imagePreviewCanvas.width, this.imagePreviewCanvas.height);
        
        // Draw placeholder text
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.font = '14px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Click on viewport to segment', this.imagePreviewCanvas.width / 2, this.imagePreviewCanvas.height / 2);

        // Also clear the Model Inputs panel to placeholder state
        this.clearModelInputsPanel();
    }

    /**
     * Show loading state with progress bar
     */
    private showLoadingState(loaded: number, total: number, stage: string): void {
        this.statusContainer.hidden = false;
        this.imagePreviewContainer.hidden = true;
        this.pointsContainer.hidden = true;
        this.statsContainer.hidden = true;
        this.instructions.hidden = true;
        this.buttonsContainer.hidden = true;

        let stageText = 'Initializing...';
        if (stage === 'encoder') {
            stageText = 'Downloading SAM encoder...';
        } else if (stage === 'decoder') {
            stageText = 'Downloading SAM decoder...';
        } else if (stage === 'initializing') {
            stageText = 'Initializing SAM model...';
        }

        const progress = total > 0 ? (loaded / total) * 100 : 0;
        const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
        const totalMB = (total / (1024 * 1024)).toFixed(1);

        this.statusLabel.text = `${stageText} (${loadedMB}/${totalMB} MB)`;
        this.progressBar.value = progress;
    }

    /**
     * Show ready state (normal UI)
     * Note: Does NOT clear the image preview - that's handled separately by
     * explicit clear actions (Clear button, sam.activated, sam.maskApplied)
     */
    private showReadyState(): void {
        this.statusContainer.hidden = true;
        this.imagePreviewContainer.hidden = false;
        this.pointsContainer.hidden = false;
        this.instructions.hidden = false;
        this.buttonsContainer.hidden = false;
        // Don't clear image preview here - it gets called when transitioning
        // from 'processing' to 'ready' which would wipe the captured image
        // before the mask can be overlaid
    }

    /**
     * Show warning message
     */
    private showWarning(message: string): void {
        this.warningContainer.hidden = false;
        this.warningLabel.text = `⚠️ ${message}`;
        this.warningContainer.class.remove('sam-panel-error');
        this.warningContainer.class.add('sam-panel-warning-visible');
    }

    /**
     * Show error message
     */
    private showError(message: string): void {
        this.warningContainer.hidden = false;
        this.warningLabel.text = `❌ ${message}`;
        this.warningContainer.class.add('sam-panel-error');
        this.warningContainer.class.remove('sam-panel-warning-visible');

        this.statusContainer.hidden = true;
        this.imagePreviewContainer.hidden = false;
        this.pointsContainer.hidden = false;
        this.instructions.hidden = false;
        this.buttonsContainer.hidden = false;
    }

    /**
     * Handle provider state changes
     */
    private handleProviderStateChange(state: ProviderState, details?: string): void {
        switch (state) {
            case 'loading-models':
                this.statusLabel.text = details || 'Loading models...';
                break;
            case 'initializing':
                this.statusLabel.text = 'Initializing ONNX runtime...';
                this.progressBar.value = 100;
                break;
            case 'ready':
                this.providerReady = true;
                this.webgpuBadge.class.add('sam-panel-webgpu-ready');
                this.showReadyState();
                break;
            case 'processing':
                break;
            case 'error':
                this.showError(details || 'An error occurred');
                break;
        }
    }

    /**
     * Show encoding state - indicates that image encoding is in progress
     */
    private showEncodingState(): void {
        this.processingIndicator.text = 'Encoding image...';
        this.processingIndicator.hidden = false;
        this.instructions.text = 'Encoding image for segmentation...';
        // Disable clicks during encoding
        this.imagePreviewCanvas.style.pointerEvents = 'none';
        this.imagePreviewCanvas.style.cursor = 'wait';
    }

    /**
     * Show encoding complete - indicates ready for segmentation clicks
     */
    private showEncodingComplete(encodeTime: number): void {
        this.processingIndicator.hidden = true;
        this.processingIndicator.text = 'Processing...'; // Reset for future use
        this.instructions.text = `Ready! Click to segment (encoded in ${encodeTime}ms)`;
        // Re-enable clicks
        this.imagePreviewCanvas.style.pointerEvents = 'auto';
        this.imagePreviewCanvas.style.cursor = 'crosshair';
        // Show encode time in stats
        this.statsLabel.text = `Encode: ${encodeTime}ms`;
        this.statsContainer.hidden = false;
    }

    /**
     * Show preview mode - displays Apply/Cancel buttons for mask confirmation
     */
    private showPreviewMode(): void {
        this.instructions.text = 'Review mask preview. Apply to select splats or cancel.';
        this.buttonsContainer.hidden = true;
        this.previewButtonsContainer.hidden = false;
    }

    /**
     * Hide preview mode - returns to normal button state
     */
    private hidePreviewMode(): void {
        this.instructions.text = 'Click to segment • Left: foreground • Right: background';
        this.buttonsContainer.hidden = false;
        this.previewButtonsContainer.hidden = true;
        // Hide mask selector when exiting preview mode
        this.maskSelectorContainer.hidden = true;
        this.allMaskCandidates = [];
    }

    /**
     * Handle click on the image preview canvas to add a point
     * Converts preview canvas coordinates back to viewport coordinates
     */
    private handlePreviewClick(e: MouseEvent, isBackground: boolean = false): void {
        // Don't allow clicks if we don't have a captured image
        if (!this.capturedImageData || !this.providerReady) return;

        // Get click position relative to canvas
        const rect = this.imagePreviewCanvas.getBoundingClientRect();
        const canvasX = e.clientX - rect.left;
        const canvasY = e.clientY - rect.top;

        // Calculate the scale and offset used when drawing the image
        const canvas = this.imagePreviewCanvas;
        const viewportWidth = this.capturedWidth;
        const viewportHeight = this.capturedHeight;

        const scale = Math.min(canvas.width / viewportWidth, canvas.height / viewportHeight);
        const scaledWidth = viewportWidth * scale;
        const scaledHeight = viewportHeight * scale;
        const offsetX = (canvas.width - scaledWidth) / 2;
        const offsetY = (canvas.height - scaledHeight) / 2;

        // Account for CSS scaling (canvas is displayed at different size than internal resolution)
        const cssScale = canvas.width / rect.width;
        const internalX = canvasX * cssScale;
        const internalY = canvasY * cssScale;

        // Check if click is within the image bounds
        if (internalX < offsetX || internalX > offsetX + scaledWidth ||
            internalY < offsetY || internalY > offsetY + scaledHeight) {
            return; // Click outside image area
        }

        // Convert back to viewport coordinates
        const viewportX = (internalX - offsetX) / scale;
        const viewportY = (internalY - offsetY) / scale;

        // Create the point
        const newPoint: SegmentationPoint = {
            x: viewportX,
            y: viewportY,
            type: isBackground ? 'bg' : 'fg'
        };

        // Add point via event (this will update the SAM selection tool's points array too)
        this.events.fire('sam.addPoint', newPoint);

        // Get all current points and trigger segmentation
        const allPoints = this.events.invoke('sam.getPoints') as SegmentationPoint[] || [];
        if (allPoints.length > 0) {
            this.events.fire('sam.segment', allPoints);
        }
    }

    /**
     * Show the mask selector panel and populate it with all mask candidates
     */
    private showMaskSelector(width: number, height: number): void {
        if (this.allMaskCandidates.length === 0) return;

        // Show the mask selector container
        this.maskSelectorContainer.hidden = false;

        // Calculate scaling for thumbnails
        const thumbWidth = 128;
        const thumbHeight = 72;

        // Render each mask candidate to its thumbnail canvas
        this.allMaskCandidates.forEach((candidate, index) => {
            if (index >= this.maskCandidates.length) return;

            const { canvas, ctx, scoreLabel, container } = this.maskCandidates[index];
            if (!ctx) return;

            // Update IoU score label
            const iouPercent = (candidate.iouScore * 100).toFixed(1);
            scoreLabel.text = `IoU: ${iouPercent}%`;

            // Clear canvas
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, thumbWidth, thumbHeight);

            // Draw the captured image as background (if available)
            if (this.capturedImageData) {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = this.capturedWidth;
                tempCanvas.height = this.capturedHeight;
                const tempCtx = tempCanvas.getContext('2d');
                if (tempCtx) {
                    tempCtx.putImageData(this.capturedImageData, 0, 0);
                    
                    // Scale and center on thumbnail
                    const scale = Math.min(thumbWidth / this.capturedWidth, thumbHeight / this.capturedHeight);
                    const scaledW = this.capturedWidth * scale;
                    const scaledH = this.capturedHeight * scale;
                    const offsetX = (thumbWidth - scaledW) / 2;
                    const offsetY = (thumbHeight - scaledH) / 2;
                    
                    ctx.drawImage(tempCanvas, offsetX, offsetY, scaledW, scaledH);
                }
            }

            // Create mask overlay from the candidate's mask (Uint8Array at original resolution)
            const maskData = candidate.mask;
            const maskImageData = new ImageData(width, height);
            const pixels = maskImageData.data;

            for (let i = 0; i < maskData.length; i++) {
                const alpha = maskData[i] / 255; // Uint8Array 0-255 -> 0-1
                const idx = i * 4;
                // GREEN semi-transparent overlay (matching main preview: #22c55e)
                pixels[idx] = 34;       // R
                pixels[idx + 1] = 197;  // G
                pixels[idx + 2] = 94;   // B
                pixels[idx + 3] = Math.floor(alpha * 178); // A (~70% opacity)
            }

            // Draw mask overlay on thumbnail
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = width;
            maskCanvas.height = height;
            const maskCtx = maskCanvas.getContext('2d');
            if (maskCtx) {
                maskCtx.putImageData(maskImageData, 0, 0);
                
                // Scale and center on thumbnail
                const scale = Math.min(thumbWidth / width, thumbHeight / height);
                const scaledW = width * scale;
                const scaledH = height * scale;
                const offsetX = (thumbWidth - scaledW) / 2;
                const offsetY = (thumbHeight - scaledH) / 2;
                ctx.globalAlpha = 0.5;
                ctx.drawImage(maskCanvas, offsetX, offsetY, scaledW, scaledH);
                ctx.globalAlpha = 1.0;
            }

            // Draw points on thumbnail
            this.drawPointsOnCanvas(ctx, thumbWidth, thumbHeight, width, height);

            // Update selection state visual
            if (index === this.selectedMaskIndex) {
                container.class.add('sam-panel-mask-candidate-selected');
            } else {
                container.class.remove('sam-panel-mask-candidate-selected');
            }
        });
    }

    /**
     * Select a different mask candidate and update the main preview
     */
    private selectMaskCandidate(index: number): void {
        if (index < 0 || index >= this.allMaskCandidates.length) return;
        if (index === this.selectedMaskIndex) return;

        // Update selected index
        this.selectedMaskIndex = index;
        const candidate = this.allMaskCandidates[index];

        // Update selection visual on thumbnails
        this.maskCandidates.forEach((mc, i) => {
            if (i === index) {
                mc.container.class.add('sam-panel-mask-candidate-selected');
            } else {
                mc.container.class.remove('sam-panel-mask-candidate-selected');
            }
        });

        // Convert Uint8Array mask to smooth Float32Array for visualization
        // Use resizeMaskSmooth to get smooth edges and resize to viewport dimensions
        // candidate.mask is Uint8Array at original image resolution (width x height)
        const maskWidth = candidate.width;
        const maskHeight = candidate.height;
        const targetWidth = this.capturedWidth || maskWidth;
        const targetHeight = this.capturedHeight || maskHeight;
        
        const smoothMask = resizeMaskSmooth(
            candidate.mask, 
            maskWidth, 
            maskHeight, 
            targetWidth, 
            targetHeight
        );

        // Update the current mask dimensions and data, then redraw preview
        this.currentMask = smoothMask;
        this.currentMaskWidth = targetWidth;
        this.currentMaskHeight = targetHeight;
        this.redrawPreview();

        // Fire event to update the segmentation service with the selected mask
        this.events.fire('sam.maskSelected', {
            index: index,
            mask: candidate.mask,
            logits: candidate.logits,
            iouScore: candidate.iouScore
        });
    }

    /**
     * Export all debug data as downloadable files:
     * - original.png: The captured viewport image
     * - mask_X_iou_XX.X.png: Each mask candidate with IoU score in filename
     * - composite.png: Image with selected mask overlay
     * - points_overlay.png: Image with just the points drawn
     * - debug_info.json: Metadata about the segmentation
     */
    private async exportDebugData(): Promise<void> {
        if (!this.capturedImageData) {
            console.warn('[SAM Debug Export] No captured image available');
            return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const prefix = `sam_debug_${timestamp}`;

        // Helper function to download a file
        const downloadFile = (blob: Blob, filename: string) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        };

        // Helper to convert canvas to blob
        const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => {
            return new Promise((resolve, reject) => {
                canvas.toBlob((blob) => {
                    if (blob) {
                        resolve(blob);
                    } else {
                        reject(new Error('Failed to create blob from canvas'));
                    }
                }, 'image/png');
            });
        };

        try {
            // 1. Export original image
            const originalCanvas = document.createElement('canvas');
            originalCanvas.width = this.capturedWidth;
            originalCanvas.height = this.capturedHeight;
            const originalCtx = originalCanvas.getContext('2d');
            if (originalCtx) {
                originalCtx.putImageData(this.capturedImageData, 0, 0);
                const originalBlob = await canvasToBlob(originalCanvas);
                downloadFile(originalBlob, `${prefix}_original.png`);
            }

            // 2. Export each mask candidate with IoU score in filename
            for (let i = 0; i < this.allMaskCandidates.length; i++) {
                const candidate = this.allMaskCandidates[i];
                const maskCanvas = document.createElement('canvas');
                
                // Get mask dimensions from the candidate (now properly tracked)
                const maskWidth = candidate.width;
                const maskHeight = candidate.height;
                maskCanvas.width = maskWidth;
                maskCanvas.height = maskHeight;
                
                const maskCtx = maskCanvas.getContext('2d');
                if (maskCtx) {
                    // Create grayscale mask image (white = mask, black = background)
                    const maskImageData = new ImageData(maskWidth, maskHeight);
                    const pixels = maskImageData.data;
                    
                    for (let j = 0; j < candidate.mask.length; j++) {
                        const value = candidate.mask[j]; // 0 or 255
                        const idx = j * 4;
                        pixels[idx] = value;     // R
                        pixels[idx + 1] = value; // G
                        pixels[idx + 2] = value; // B
                        pixels[idx + 3] = 255;   // A (fully opaque)
                    }
                    
                    maskCtx.putImageData(maskImageData, 0, 0);
                    
                    const iouStr = (candidate.iouScore * 100).toFixed(1).replace('.', '_');
                    const maskBlob = await canvasToBlob(maskCanvas);
                    downloadFile(maskBlob, `${prefix}_mask_${i}_iou_${iouStr}.png`);
                }
            }

            // 3. Export composite preview (image + selected mask overlay)
            const compositeCanvas = document.createElement('canvas');
            compositeCanvas.width = this.capturedWidth;
            compositeCanvas.height = this.capturedHeight;
            const compositeCtx = compositeCanvas.getContext('2d');
            if (compositeCtx && this.currentMask) {
                // Draw original image
                compositeCtx.putImageData(this.capturedImageData, 0, 0);
                
                // Create and overlay mask with green tint
                const maskWidth = this.currentMaskWidth;
                const maskHeight = this.currentMaskHeight;
                const maskOverlay = new ImageData(maskWidth, maskHeight);
                const overlayPixels = maskOverlay.data;
                
                for (let i = 0; i < this.currentMask.length; i++) {
                    const alpha = Math.max(0, Math.min(1, this.currentMask[i]));
                    const idx = i * 4;
                    // GREEN overlay (matching main preview)
                    overlayPixels[idx] = 34;      // R
                    overlayPixels[idx + 1] = 197; // G
                    overlayPixels[idx + 2] = 94;  // B
                    overlayPixels[idx + 3] = Math.floor(alpha * 178); // ~70% opacity
                }
                
                const overlayCanvas = document.createElement('canvas');
                overlayCanvas.width = maskWidth;
                overlayCanvas.height = maskHeight;
                const overlayCtx = overlayCanvas.getContext('2d');
                if (overlayCtx) {
                    overlayCtx.putImageData(maskOverlay, 0, 0);
                    compositeCtx.drawImage(overlayCanvas, 0, 0, this.capturedWidth, this.capturedHeight);
                }
                
                const compositeBlob = await canvasToBlob(compositeCanvas);
                downloadFile(compositeBlob, `${prefix}_composite.png`);
            }

            // 4. Export points overlay (image with points drawn on it)
            const pointsCanvas = document.createElement('canvas');
            pointsCanvas.width = this.capturedWidth;
            pointsCanvas.height = this.capturedHeight;
            const pointsCtx = pointsCanvas.getContext('2d');
            if (pointsCtx) {
                // Draw original image
                pointsCtx.putImageData(this.capturedImageData, 0, 0);
                
                // Draw points using same style as preview
                const pointRadius = 10;
                this.currentPoints.forEach((point, index) => {
                    const x = point.x;
                    const y = point.y;

                    // Outer circle (white border)
                    pointsCtx.beginPath();
                    pointsCtx.arc(x, y, pointRadius + 3, 0, Math.PI * 2);
                    pointsCtx.fillStyle = 'white';
                    pointsCtx.fill();
                    pointsCtx.strokeStyle = 'black';
                    pointsCtx.lineWidth = 2;
                    pointsCtx.stroke();

                    // Inner circle (color indicates type)
                    pointsCtx.beginPath();
                    pointsCtx.arc(x, y, pointRadius, 0, Math.PI * 2);
                    pointsCtx.fillStyle = point.type === 'fg' ? '#22c55e' : '#ef4444';
                    pointsCtx.fill();
                    pointsCtx.strokeStyle = 'white';
                    pointsCtx.lineWidth = 2;
                    pointsCtx.stroke();

                    // Number label
                    pointsCtx.fillStyle = 'white';
                    pointsCtx.font = 'bold 12px system-ui, sans-serif';
                    pointsCtx.textAlign = 'center';
                    pointsCtx.textBaseline = 'middle';
                    pointsCtx.fillText((index + 1).toString(), x, y);
                });
                
                const pointsBlob = await canvasToBlob(pointsCanvas);
                downloadFile(pointsBlob, `${prefix}_points_overlay.png`);
            }

            // 5. Export debug_info.json with metadata
            const debugInfo = {
                timestamp: new Date().toISOString(),
                imageDimensions: {
                    width: this.capturedWidth,
                    height: this.capturedHeight
                },
                points: this.currentPoints.map((p, i) => ({
                    index: i + 1,
                    x: p.x,
                    y: p.y,
                    type: p.type,
                    normalizedX: p.x / this.capturedWidth,
                    normalizedY: p.y / this.capturedHeight
                })),
                maskCandidates: this.allMaskCandidates.map((c, i) => ({
                    index: i,
                    iouScore: c.iouScore,
                    iouPercent: (c.iouScore * 100).toFixed(2) + '%',
                    isSelected: i === this.selectedMaskIndex,
                    maskSize: Math.sqrt(c.mask.length)
                })),
                selectedMaskIndex: this.selectedMaskIndex,
                stats: this.statsLabel.text || 'N/A',
                providerReady: this.providerReady,
                hasPendingMask: this.hasPendingMask
            };
            
            const jsonBlob = new Blob([JSON.stringify(debugInfo, null, 2)], { type: 'application/json' });
            downloadFile(jsonBlob, `${prefix}_debug_info.json`);

            console.log('[SAM Debug Export] Successfully exported all debug files with prefix:', prefix);
        } catch (error) {
            console.error('[SAM Debug Export] Error exporting debug data:', error);
        }
    }
}

// CSS styles for the redesigned SAM panel
const samPanelStyles = `
.sam-panel {
    position: absolute;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(26, 26, 26, 0.98);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    z-index: 100;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    min-width: 1150px;
    max-width: 1280px;
}

.sam-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.sam-panel-title {
    font-size: 14px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.95);
}

.sam-panel-webgpu-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.sam-panel-webgpu-ready {
    background: rgba(34, 197, 94, 0.2) !important;
    color: #22c55e !important;
}

.sam-panel-webgpu-unavailable {
    background: rgba(239, 68, 68, 0.2) !important;
    color: #ef4444 !important;
}

.sam-panel-status {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: center;
    padding: 20px;
}

.sam-panel-status-label {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.8);
    text-align: center;
}

.sam-panel-progress {
    width: 100%;
    height: 6px;
    border-radius: 3px;
}

.sam-panel-progress .pcui-progress-bar {
    background: #3b82f6;
    border-radius: 3px;
    transition: width 0.2s ease;
}

.sam-panel-warning {
    padding: 8px 12px;
    border-radius: 6px;
    background: rgba(234, 179, 8, 0.15);
    border: 1px solid rgba(234, 179, 8, 0.3);
}

.sam-panel-warning.sam-panel-error {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.sam-panel-warning-label {
    font-size: 11px;
    color: #fbbf24;
    line-height: 1.4;
}

.sam-panel-error .sam-panel-warning-label {
    color: #ef4444;
}

.sam-panel-preview-container {
    position: relative;
    background: #1a1a1a;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.sam-panel-preview-canvas {
    display: block;
    width: 100%;
    height: auto;
    aspect-ratio: 2/1;
}

.sam-panel-processing {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0, 0, 0, 0.8);
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 13px;
    color: white;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sam-panel-processing::before {
    content: '';
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: sam-spin 0.8s linear infinite;
}

.sam-panel-points {
    display: flex;
    gap: 12px;
    justify-content: center;
}

.sam-panel-fg {
    font-size: 12px;
    padding: 6px 12px;
    border-radius: 6px;
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    font-weight: 500;
}

.sam-panel-bg {
    font-size: 12px;
    padding: 6px 12px;
    border-radius: 6px;
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    font-weight: 500;
}

.sam-panel-stats {
    display: flex;
    justify-content: center;
}

.sam-panel-stats-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
}

.sam-panel-instructions {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
    text-align: center;
}

.sam-panel-buttons {
    display: flex;
    gap: 8px;
    justify-content: center;
}

.sam-panel-button {
    min-width: 80px;
}

.sam-panel-preview-buttons {
    display: flex;
    gap: 12px;
    justify-content: center;
}

.sam-panel-apply {
    min-width: 140px;
    background: #22c55e !important;
    font-weight: 500;
}

.sam-panel-apply:hover:not(:disabled) {
    background: #16a34a !important;
}

.sam-panel-cancel {
    min-width: 100px;
    background: rgba(255, 255, 255, 0.1) !important;
}

.sam-panel-cancel:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.2) !important;
}

.sam-panel-debug-export {
    min-width: 110px;
    background: rgba(59, 130, 246, 0.2) !important;
    color: #3b82f6 !important;
    font-size: 11px !important;
}

.sam-panel-debug-export:hover:not(:disabled) {
    background: rgba(59, 130, 246, 0.35) !important;
}

@keyframes sam-spin {
    to {
        transform: rotate(360deg);
    }
}

/* Multi-mask selector styles */
.sam-panel-main-content {
    display: flex;
    gap: 12px;
    align-items: flex-start;
}

.sam-panel-mask-selector {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    min-width: 140px;
}

.sam-panel-mask-selector-title {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.7);
    text-align: center;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 4px;
}

.sam-panel-mask-candidate {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 6px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.03);
    border: 2px solid transparent;
    cursor: pointer;
    transition: all 0.15s ease;
}

.sam-panel-mask-candidate:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(59, 130, 246, 0.3);
}

.sam-panel-mask-candidate-selected {
    background: rgba(59, 130, 246, 0.15) !important;
    border-color: #3b82f6 !important;
}

.sam-panel-mask-candidate-index {
    font-size: 10px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.6);
}

.sam-panel-mask-candidate-canvas {
    display: block;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.sam-panel-mask-candidate-score {
    font-size: 10px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
}

.sam-panel-mask-candidate-selected .sam-panel-mask-candidate-score {
    color: #3b82f6;
}

/* Model Inputs panel styles (left side) */
.sam-panel-model-inputs {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    min-width: 150px;
    max-width: 160px;
}

.sam-panel-model-inputs-title {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.7);
    text-align: center;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 4px;
}

.sam-panel-input-image-container,
.sam-panel-prev-mask-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}

.sam-panel-input-image-label,
.sam-panel-prev-mask-label {
    font-size: 9px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.sam-panel-input-image-canvas {
    display: block;
    width: 128px;
    height: 128px;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: #0a0a0a;
}

.sam-panel-prev-mask-canvas {
    display: block;
    width: 128px;
    height: 72px;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: #0a0a0a;
}

.sam-panel-point-prompts-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 4px;
}

.sam-panel-point-prompts-title {
    font-size: 9px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.3px;
    text-align: center;
}

.sam-panel-point-prompts-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
    max-height: 100px;
    overflow-y: auto;
}

.sam-panel-point-prompt-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 6px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.03);
    font-size: 9px;
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
}

.sam-panel-point-prompt-item.fg {
    border-left: 2px solid #22c55e;
}

.sam-panel-point-prompt-item.bg {
    border-left: 2px solid #ef4444;
}

.sam-panel-point-prompt-index {
    font-weight: 600;
    color: rgba(255, 255, 255, 0.7);
    min-width: 14px;
}

.sam-panel-point-prompt-coords {
    color: rgba(255, 255, 255, 0.5);
}

.sam-panel-point-prompt-type {
    margin-left: auto;
    font-weight: 500;
    text-transform: uppercase;
    font-size: 8px;
}

.sam-panel-point-prompt-type.fg {
    color: #22c55e;
}

.sam-panel-point-prompt-type.bg {
    color: #ef4444;
}

.sam-panel-no-points {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.3);
    text-align: center;
    padding: 8px 0;
    font-style: italic;
}
`;

// Inject styles
const styleElement = document.createElement('style');
styleElement.textContent = samPanelStyles;
document.head.appendChild(styleElement);

export { SamPanel };
