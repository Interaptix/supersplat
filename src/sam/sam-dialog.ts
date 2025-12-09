/**
 * SAM2 Dialog
 * PCUI dialog for SAM2 image segmentation.
 * Ported from next-sam React implementation.
 */

import { Button, Container, Element, Label, Spinner } from '@playcanvas/pcui';

import { Events } from '../events';
import {
    resizeCanvas,
    canvasToFloat32Array,
    float32ArrayToCanvas,
    sliceTensor
} from './image-utils';
import { localize } from '../ui/localization';

// Configuration: Set to false for automatic mode (production), true for debug mode (manual controls)
const SAM_DEBUG_MODE = false;

// SAM2 constants
const IMAGE_SIZE = { w: 1024, h: 1024 };
const MASK_SIZE = { w: 256, h: 256 };

interface SAM2Point {
    x: number;
    y: number;
    label: number;
}

interface AllMasks {
    canvases: HTMLCanvasElement[];
    arrays: Float32Array[];
    scores: number[];
    selectedIdx: number;
}

interface Stats {
    device: string;
    downloadModelsTime: number[];
    encodeImageTimes: number[];
    decodeTimes: number[];
}

// Selection result returned when user applies mask
interface SAMSelectionResult {
    mask: HTMLCanvasElement;
    cameraPose: {
        position: { x: number; y: number; z: number };
        target: { x: number; y: number; z: number };
    };
    originalWidth: number;
    originalHeight: number;
    operation: 'add' | 'remove' | 'set';
}

class SAMDialog extends Container {
    show: (capturedImage?: HTMLCanvasElement) => Promise<SAMSelectionResult | null>;
    hide: () => void;
    destroy: () => void;

    constructor(events: Events, args = {}) {
        args = {
            ...args,
            id: 'sam-dialog',
            class: 'settings-dialog',
            hidden: true,
            tabIndex: -1
        };

        super(args);

        // State
        let worker: Worker | null = null;
        let device: string | null = null;
        let loading = false;
        let imageEncoded = false;
        let status = 'Initializing...';
        let image: HTMLCanvasElement | null = null;
        let mask: HTMLCanvasElement | null = null;
        let prevMaskArray: Float32Array | null = null;
        let allMasks: AllMasks | null = null;
        let points: SAM2Point[] = [];
        let stats: Stats | null = null;

        // Automatic mode state (used when SAM_DEBUG_MODE = false)
        let workerReady = false;
        let pendingAutoEncode = false;

        // Store camera pose and original dimensions when capturing screen
        let capturedCameraPose: {
            position: { x: number; y: number; z: number };
            target: { x: number; y: number; z: number };
        } | null = null;
        let capturedOriginalWidth = 0;
        let capturedOriginalHeight = 0;

        // Dialog container
        const dialog = new Container({ id: 'dialog', class: 'sam-dialog-inner' });

        // Header
        const headerText = new Label({ id: 'text', text: 'SEGMENTATION' });
        const header = new Container({ id: 'header' });
        header.append(headerText);

        // Status row
        const statusRow = new Container({ class: 'sam-status-row' });
        const statusLabel = new Label({ class: 'sam-status-label', text: status });
        const statusSpinner = new Spinner({ class: 'sam-spinner', hidden: true, size: 16 });
        statusRow.append(statusLabel);
        statusRow.append(statusSpinner);

        // Device info
        const deviceLabel = new Label({ class: 'sam-device-label', text: '' });

        // Canvas container
        const canvasContainer = new Container({ class: 'sam-canvas-container' });
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        canvas.className = 'sam-canvas';
        canvasContainer.dom.appendChild(canvas);

        // Button row
        const buttonRow = new Container({ class: 'sam-button-row' });

        const encodeButton = new Button({
            class: 'sam-button',
            text: 'Encode Image',
            enabled: false
        });

        const uploadButton = new Button({
            class: 'sam-button',
            text: 'Upload'
        });

        const captureButton = new Button({
            class: 'sam-button',
            text: 'Capture Screen'
        });

        const clearButton = new Button({
            class: 'sam-button',
            text: 'Clear Points',
            enabled: false
        });

        buttonRow.append(encodeButton);
        buttonRow.append(uploadButton);
        buttonRow.append(captureButton);
        buttonRow.append(clearButton);

        // Points info
        const pointsLabel = new Label({ class: 'sam-points-label', text: 'Points: 0 (Left=Add, Right=Remove)' });

        // Mask selection row (buttons will be wired up after selectMask is defined)
        const maskRow = new Container({ class: 'sam-mask-row', hidden: true });
        const maskLabel = new Label({ class: 'label', text: 'Select Mask:' });
        const maskButtons: Button[] = [];
        for (let i = 0; i < 3; i++) {
            const btn = new Button({
                class: 'sam-mask-button',
                text: `Mask ${i}`
            });
            maskButtons.push(btn);
        }
        maskRow.append(maskLabel);
        maskButtons.forEach(btn => maskRow.append(btn));

        // Footer
        const footer = new Container({ id: 'footer', class: 'sam-footer' });

        const cancelButton = new Button({
            class: 'button',
            text: 'Cancel'
        });

        // Selection operation buttons
        const setButton = new Button({
            class: ['button', 'sam-action-button'],
            text: 'Set Selection',
            enabled: false
        });

        const addButton = new Button({
            class: ['button', 'sam-action-button'],
            text: 'Add to Selection',
            enabled: false
        });

        const removeButton = new Button({
            class: ['button', 'sam-action-button'],
            text: 'Remove from Selection',
            enabled: false
        });

        footer.append(cancelButton);
        footer.append(setButton);
        footer.append(addButton);
        footer.append(removeButton);

        // Assemble dialog
        const content = new Container({ id: 'content' });
        content.append(statusRow);
        content.append(deviceLabel);
        content.append(canvasContainer);
        content.append(buttonRow);
        content.append(pointsLabel);
        content.append(maskRow);

        dialog.append(header);
        dialog.append(content);
        dialog.append(footer);

        this.append(dialog);

        // Hidden file input
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        document.body.appendChild(fileInput);

        // Helper functions
        const updateStatus = (newStatus: string, isLoading = false) => {
            status = newStatus;
            loading = isLoading;
            statusLabel.text = status;
            statusSpinner.hidden = !isLoading;
            encodeButton.enabled = !isLoading && !imageEncoded && image !== null;
            clearButton.enabled = !isLoading && points.length > 0;
            const maskReady = !isLoading && mask !== null;
            setButton.enabled = maskReady;
            addButton.enabled = maskReady;
            removeButton.enabled = maskReady;
        };

        const updateDeviceLabel = () => {
            if (device) {
                deviceLabel.text = `Running on: ${device}`;
            }
        };

        const updatePointsLabel = () => {
            pointsLabel.text = `Points: ${points.length} (Left=Add, Right=Remove)`;
        };

        const updateMaskButtons = () => {
            if (allMasks) {
                maskRow.hidden = false;
                maskButtons.forEach((btn, i) => {
                    const score = allMasks.scores[i];
                    const isBest = i === allMasks.scores.indexOf(Math.max(...allMasks.scores));
                    const isSelected = i === allMasks.selectedIdx;
                    btn.text = `Mask ${i} (${score.toFixed(3)})${isBest ? ' ★' : ''}`;
                    btn.class[isSelected ? 'add' : 'remove']('active');
                });
            } else {
                maskRow.hidden = true;
            }
        };

        const drawCanvas = () => {
            const ctx = canvas.getContext('2d')!;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (image) {
                ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvas.width, canvas.height);

                if (mask) {
                    ctx.globalAlpha = 0.7;
                    ctx.drawImage(mask, 0, 0, mask.width, mask.height, 0, 0, canvas.width, canvas.height);
                    ctx.globalAlpha = 1;
                }

                // Draw points
                points.forEach((point) => {
                    const x = (point.x / IMAGE_SIZE.w) * canvas.width;
                    const y = (point.y / IMAGE_SIZE.h) * canvas.height;

                    ctx.beginPath();
                    ctx.arc(x, y, 8, 0, 2 * Math.PI);
                    ctx.fillStyle = point.label === 1 ? '#22c55e' : '#ef4444';
                    ctx.fill();
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                });
            }
        };

        // Clear segmentation results only (keeps image AND encoding)
        // Used by: Clear Points button
        const clearSegmentation = () => {
            points = [];
            mask = null;
            prevMaskArray = null;
            allMasks = null;
            updatePointsLabel();
            updateMaskButtons();
            drawCanvas();
            setButton.enabled = false;
            addButton.enabled = false;
            removeButton.enabled = false;
        };

        // Full reset for opening dialog (clears everything for fresh session)
        // Used by: show() to ensure clean slate
        const resetSession = () => {
            image = null;
            mask = null;
            prevMaskArray = null;
            allMasks = null;
            points = [];
            imageEncoded = false;
            capturedCameraPose = null;
            capturedOriginalWidth = 0;
            capturedOriginalHeight = 0;
            stats = null;

            // Reset automatic mode state
            workerReady = false;
            pendingAutoEncode = false;

            // Reset UI
            updateStatus('Initializing...');
            updatePointsLabel();
            updateMaskButtons();
            encodeButton.enabled = false;
            clearButton.enabled = false;
            setButton.enabled = false;
            addButton.enabled = false;
            removeButton.enabled = false;
            drawCanvas();
        };

        const selectMask = (idx: number) => {
            if (!allMasks) return;
            allMasks.selectedIdx = idx;
            mask = allMasks.canvases[idx];
            prevMaskArray = allMasks.arrays[idx];
            updateMaskButtons();
            drawCanvas();
        };

        // Wire up mask button click handlers now that selectMask is defined
        maskButtons.forEach((btn, i) => {
            btn.dom.addEventListener('click', () => selectMask(i));
        });

        // Auto-encode image (used in automatic mode)
        const autoEncodeImage = () => {
            if (!image || !worker) return;

            const resizedCanvas = resizeCanvas(image, IMAGE_SIZE);
            const tensorData = canvasToFloat32Array(resizedCanvas);

            worker.postMessage({
                type: 'encodeImage',
                data: tensorData
            });

            updateStatus('Encoding image...', true);
        };

        // Auto-capture screen (used in automatic mode)
        const autoCaptureScreen = async () => {
            try {
                updateStatus('Capturing screen...', true);

                // Invoke the capture.screen event to get the current canvas and camera pose
                const captureData = await events.invoke('capture.screen');

                if (captureData && captureData.image) {
                    // Store the camera pose and original dimensions for later use
                    capturedCameraPose = captureData.cameraPose;
                    capturedOriginalWidth = captureData.canvasWidth;
                    capturedOriginalHeight = captureData.canvasHeight;

                    // Get the captured canvas
                    const capturedCanvas = captureData.image;
                    const width = capturedCanvas.width;
                    const height = capturedCanvas.height;

                    // Calculate padding to make square (same logic as loadImage)
                    const largestDim = Math.max(width, height);
                    const padX = (largestDim - width) / 2;
                    const padY = (largestDim - height) / 2;

                    // Create a square canvas with the captured image centered
                    const squareCanvas = document.createElement('canvas');
                    squareCanvas.width = largestDim;
                    squareCanvas.height = largestDim;

                    const ctx = squareCanvas.getContext('2d')!;
                    ctx.fillStyle = '#000000';
                    ctx.fillRect(0, 0, largestDim, largestDim);
                    ctx.drawImage(capturedCanvas, padX, padY, width, height);

                    // Set as current image
                    image = squareCanvas;
                    clearSegmentation();
                    imageEncoded = false;
                    drawCanvas();

                    console.log('[SAM2] Auto-captured screen with camera pose:', capturedCameraPose);

                    // In automatic mode, either encode immediately if worker ready, or set pending flag
                    if (workerReady) {
                        autoEncodeImage();
                    } else {
                        pendingAutoEncode = true;
                        updateStatus('Waiting for model...', true);
                    }
                } else {
                    updateStatus('Failed to capture screen');
                }
            } catch (error) {
                console.error('[SAM2] Error auto-capturing screen:', error);
                updateStatus(`Error capturing screen: ${(error as Error).message}`);
            }
        };

        const handleDecodingResults = (decodingResults: any) => {
            const maskTensors = decodingResults.masks;
            const [, noMasks, width, height] = maskTensors.dims;
            const maskScores = Array.from(decodingResults.iou_predictions.data) as number[];

            const maskArrays: Float32Array[] = [];
            const maskCanvases: HTMLCanvasElement[] = [];

            for (let i = 0; i < noMasks; i++) {
                const maskArray = sliceTensor(maskTensors, i);
                maskArrays.push(maskArray);
                let maskCanvas = float32ArrayToCanvas(maskArray, width, height);
                maskCanvas = resizeCanvas(maskCanvas, IMAGE_SIZE);
                maskCanvases.push(maskCanvas);
            }

            const bestMaskIdx = maskScores.indexOf(Math.max(...maskScores));

            allMasks = {
                canvases: maskCanvases,
                arrays: maskArrays,
                scores: maskScores,
                selectedIdx: bestMaskIdx
            };

            mask = maskCanvases[bestMaskIdx];
            prevMaskArray = maskArrays[bestMaskIdx];

            updateMaskButtons();
            drawCanvas();
            updateStatus('Ready. Click on image to refine.');
            setButton.enabled = true;
            addButton.enabled = true;
            removeButton.enabled = true;
        };

        // Worker message handler
        const onWorkerMessage = (event: MessageEvent) => {
            const { type, data } = event.data;

            if (type === 'modelReady') {
                const { success, device: dev } = data;
                if (success) {
                    device = dev;
                    updateDeviceLabel();
                    workerReady = true;

                    // In automatic mode, check if we have a pending image to encode
                    if (!SAM_DEBUG_MODE && pendingAutoEncode && image) {
                        pendingAutoEncode = false;
                        autoEncodeImage();
                    } else {
                        updateStatus('Ready. Encode image to start.');
                        encodeButton.enabled = image !== null;
                    }
                } else {
                    updateStatus('Error loading model (check console)');
                }
            } else if (type === 'downloadInProgress') {
                updateStatus('Downloading model...', true);
            } else if (type === 'loadingInProgress') {
                updateStatus('Loading model...', true);
            } else if (type === 'encodeImageDone') {
                imageEncoded = true;
                updateStatus('Ready. Click on image to segment.');
            } else if (type === 'decodeMaskResult') {
                handleDecodingResults(data);
            } else if (type === 'stats') {
                stats = data;
            }
        };

        // Initialize worker
        const initWorker = () => {
            if (worker) return;

            // Create worker from the bundled worker file (built by rollup to dist/sam-worker.js)
            worker = new Worker('./sam-worker.js', {
                type: 'module'
            });
            worker.addEventListener('message', onWorkerMessage);
            worker.addEventListener('error', (e) => {
                console.error('[SAM2] Worker error:', e);
                updateStatus(`Worker error: ${e.message}`);
            });
            worker.postMessage({ type: 'initModel' });
            updateStatus('Initializing...', true);
        };

        // Event handlers
        const handleCanvasClick = (event: MouseEvent) => {
            if (!imageEncoded || loading) return;

            event.preventDefault();

            const rect = canvas.getBoundingClientRect();
            const point: SAM2Point = {
                x: ((event.clientX - rect.left) / canvas.width) * IMAGE_SIZE.w,
                y: ((event.clientY - rect.top) / canvas.height) * IMAGE_SIZE.h,
                label: event.button === 0 ? 1 : 0
            };

            points.push(point);
            updatePointsLabel();
            drawCanvas();

            updateStatus('Decoding...', true);

            if (prevMaskArray) {
                worker!.postMessage({
                    type: 'decodeMask',
                    data: {
                        points,
                        maskArray: prevMaskArray,
                        maskShape: [1, 1, MASK_SIZE.w, MASK_SIZE.h]
                    }
                });
            } else {
                worker!.postMessage({
                    type: 'decodeMask',
                    data: {
                        points,
                        maskArray: null,
                        maskShape: null
                    }
                });
            }
        };

        canvas.addEventListener('click', handleCanvasClick);
        canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            handleCanvasClick(e);
        });

        encodeButton.on('click', () => {
            if (!image || !worker) return;

            const resizedCanvas = resizeCanvas(image, IMAGE_SIZE);
            const tensorData = canvasToFloat32Array(resizedCanvas);

            worker.postMessage({
                type: 'encodeImage',
                data: tensorData
            });

            updateStatus('Encoding image...', true);
        });

        const loadImage = (url: string) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.src = url;
            img.onload = () => {
                const largestDim = Math.max(img.naturalWidth, img.naturalHeight);

                // Calculate padding to make square
                const padX = (largestDim - img.naturalWidth) / 2;
                const padY = (largestDim - img.naturalHeight) / 2;

                const offscreenCanvas = document.createElement('canvas');
                offscreenCanvas.width = largestDim;
                offscreenCanvas.height = largestDim;

                const ctx = offscreenCanvas.getContext('2d')!;
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, largestDim, largestDim);
                ctx.drawImage(img, padX, padY, img.naturalWidth, img.naturalHeight);

                image = offscreenCanvas;
                clearSegmentation();
                imageEncoded = false;
                drawCanvas();
                updateStatus('Ready. Encode image to start.');
                encodeButton.enabled = true;
            };
        };

        uploadButton.on('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            const file = (e.target as HTMLInputElement).files?.[0];
            if (!file) return;

            const url = URL.createObjectURL(file);
            loadImage(url);
            (e.target as HTMLInputElement).value = '';
        });

        clearButton.on('click', () => {
            clearSegmentation();
            if (imageEncoded) {
                updateStatus('Ready. Click on image to segment.');
            }
        });

        // Capture Screen button handler
        captureButton.on('click', async () => {
            try {
                updateStatus('Capturing screen...', true);

                // Invoke the capture.screen event to get the current canvas and camera pose
                const captureData = await events.invoke('capture.screen');

                if (captureData && captureData.image) {
                    // Store the camera pose and original dimensions for later use
                    capturedCameraPose = captureData.cameraPose;
                    capturedOriginalWidth = captureData.canvasWidth;
                    capturedOriginalHeight = captureData.canvasHeight;

                    // Get the captured canvas
                    const capturedCanvas = captureData.image;
                    const width = capturedCanvas.width;
                    const height = capturedCanvas.height;

                    // Calculate padding to make square (same logic as loadImage)
                    const largestDim = Math.max(width, height);
                    const padX = (largestDim - width) / 2;
                    const padY = (largestDim - height) / 2;

                    // Create a square canvas with the captured image centered
                    const squareCanvas = document.createElement('canvas');
                    squareCanvas.width = largestDim;
                    squareCanvas.height = largestDim;

                    const ctx = squareCanvas.getContext('2d')!;
                    ctx.fillStyle = '#000000';
                    ctx.fillRect(0, 0, largestDim, largestDim);
                    ctx.drawImage(capturedCanvas, padX, padY, width, height);

                    // Set as current image and reset segmentation state
                    image = squareCanvas;
                    clearSegmentation();
                    imageEncoded = false;
                    drawCanvas();

                    console.log('[SAM2] Screen captured with camera pose:', capturedCameraPose);
                    updateStatus('Ready. Encode image to start.');
                    encodeButton.enabled = true;
                } else {
                    updateStatus('Failed to capture screen');
                }
            } catch (error) {
                console.error('[SAM2] Error capturing screen:', error);
                updateStatus(`Error capturing screen: ${(error as Error).message}`);
            }
        });

        // Keyboard handler
        let onCancel: () => void;
        let onApply: (operation: 'add' | 'remove' | 'set') => void;

        const keydown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                onCancel();
            }
        };

        cancelButton.on('click', () => onCancel());
        setButton.on('click', () => onApply('set'));
        addButton.on('click', () => onApply('add'));
        removeButton.on('click', () => onApply('remove'));

        // Public methods
        this.show = (capturedImage?: HTMLCanvasElement) => {
            console.log('SAMDialog.show() called, setting hidden = false');
            this.hidden = false;
            console.log('SAMDialog hidden state:', this.hidden, 'DOM display:', window.getComputedStyle(this.dom).display);
            document.addEventListener('keydown', keydown);
            this.dom.focus();

            // Reset session state for fresh start (clears all previous session data)
            resetSession();

            // Initialize worker if needed
            initWorker();

            // Always send initModel to the worker to update status (worker may already be initialized from previous open)
            if (worker) {
                worker.postMessage({ type: 'initModel' });
                updateStatus('Initializing...', true);
            }

            // Mode-specific behavior
            if (SAM_DEBUG_MODE) {
                // Debug mode: Show all manual controls, optionally use provided image
                encodeButton.hidden = false;
                uploadButton.hidden = false;
                captureButton.hidden = false;

                if (capturedImage) {
                    image = capturedImage;
                    clearSegmentation();
                    imageEncoded = false;
                    drawCanvas();
                    updateStatus('Ready. Encode image to start.');
                    encodeButton.enabled = true;
                }
            } else {
                // Automatic mode: Hide manual controls, auto-capture and encode
                encodeButton.hidden = true;
                uploadButton.hidden = true;
                captureButton.hidden = true;

                // Auto-capture screen and encode (handles async timing internally)
                autoCaptureScreen();
            }

            return new Promise<SAMSelectionResult | null>((resolve) => {
                onCancel = () => {
                    resolve(null);
                };

                onApply = (operation: 'add' | 'remove' | 'set') => {
                    if (mask && capturedCameraPose && capturedOriginalWidth > 0) {
                        // Return the selection result with mask, camera pose, original dimensions, and operation
                        resolve({
                            mask,
                            cameraPose: capturedCameraPose,
                            originalWidth: capturedOriginalWidth,
                            originalHeight: capturedOriginalHeight,
                            operation
                        });
                    } else {
                        resolve(null);
                    }
                };
            }).finally(() => {
                document.removeEventListener('keydown', keydown);
                this.hide();
            });
        };

        this.hide = () => {
            this.hidden = true;
        };

        this.destroy = () => {
            this.hide();
            if (worker) {
                worker.terminate();
                worker = null;
            }
            if (fileInput.parentNode) {
                fileInput.parentNode.removeChild(fileInput);
            }
            super.destroy();
        };
    }
}

export { SAMDialog };
