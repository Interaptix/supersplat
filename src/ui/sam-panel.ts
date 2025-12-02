import { Container, Button, Label } from '@playcanvas/pcui';
import { Events } from '../events';
import { SegmentationPoint } from '../segmentation/types';

/**
 * SAM Panel - Floating action panel for SAM segmentation tool.
 * Shows point counts and provides action buttons.
 */
class SamPanel extends Container {
    constructor(events: Events, args = {}) {
        args = {
            ...args,
            id: 'sam-panel',
            class: 'sam-panel',
            hidden: true
        };

        super(args);

        // Point count labels
        const pointsContainer = new Container({
            class: 'sam-panel-points'
        });

        const fgLabel = new Label({
            class: 'sam-panel-fg',
            text: 'Foreground: 0'
        });

        const bgLabel = new Label({
            class: 'sam-panel-bg',
            text: 'Background: 0'
        });

        pointsContainer.append(fgLabel);
        pointsContainer.append(bgLabel);

        // Instructions
        const instructions = new Label({
            class: 'sam-panel-instructions',
            text: 'Left-click: foreground | Right-click: background'
        });

        // Action buttons
        const buttonsContainer = new Container({
            class: 'sam-panel-buttons'
        });

        const clearButton = new Button({
            class: 'sam-panel-button',
            text: 'Clear',
            icon: 'E120' // refresh icon
        });

        const undoButton = new Button({
            class: 'sam-panel-button',
            text: 'Undo',
            icon: 'E114' // undo icon
        });

        const segmentButton = new Button({
            class: 'sam-panel-segment',
            text: 'Select',
            icon: 'E401' // play/process icon
        });

        buttonsContainer.append(clearButton);
        buttonsContainer.append(undoButton);
        buttonsContainer.append(segmentButton);

        // Assemble panel
        this.append(pointsContainer);
        this.append(instructions);
        this.append(buttonsContainer);

        // Button event handlers
        clearButton.on('click', () => {
            events.fire('sam.clearPoints');
        });

        undoButton.on('click', () => {
            events.fire('sam.undoPoint');
        });

        segmentButton.on('click', () => {
            const points = events.invoke('sam.getPoints') as SegmentationPoint[];
            if (points && points.length > 0) {
                events.fire('sam.segment', points);
            }
        });

        // Update point counts when points change
        events.on('sam.pointsChanged', (points: SegmentationPoint[]) => {
            const fgCount = points.filter(p => p.type === 'fg').length;
            const bgCount = points.filter(p => p.type === 'bg').length;
            fgLabel.text = `Foreground: ${fgCount}`;
            bgLabel.text = `Background: ${bgCount}`;

            // Enable/disable segment button based on point count
            segmentButton.enabled = points.length > 0;
            undoButton.enabled = points.length > 0;
        });

        // Show/hide panel based on SAM tool activation
        events.on('sam.activated', () => {
            this.hidden = false;
            // Reset button states
            segmentButton.enabled = false;
            undoButton.enabled = false;
            fgLabel.text = 'Foreground: 0';
            bgLabel.text = 'Background: 0';
        });

        events.on('sam.deactivated', () => {
            this.hidden = true;
        });

        events.on('sam.cancelled', () => {
            this.hidden = true;
        });

        // Handle segmentation start/complete for loading state
        events.on('sam.segmentStart', () => {
            segmentButton.enabled = false;
            segmentButton.text = 'Selecting...';
        });

        events.on('sam.segmentComplete', () => {
            segmentButton.enabled = true;
            segmentButton.text = 'Select';
        });

        events.on('sam.segmentError', (error: string) => {
            segmentButton.enabled = true;
            segmentButton.text = 'Select';
            // Could show error toast here
            console.error('Segmentation error:', error);
        });
    }
}

// CSS styles for the SAM panel
const samPanelStyles = `
.sam-panel {
    position: absolute;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(30, 30, 30, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    min-width: 280px;
}

.sam-panel-points {
    display: flex;
    gap: 16px;
    justify-content: center;
}

.sam-panel-fg {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
}

.sam-panel-bg {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

.sam-panel-instructions {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    text-align: center;
}

.sam-panel-buttons {
    display: flex;
    gap: 8px;
    justify-content: center;
}

.sam-panel-button {
    min-width: 70px;
}

.sam-panel-segment {
    min-width: 70px;
    background: #3b82f6 !important;
}

.sam-panel-segment:hover:not(:disabled) {
    background: #2563eb !important;
}

.sam-panel-segment:disabled {
    background: rgba(59, 130, 246, 0.5) !important;
}
`;

// Inject styles
const styleElement = document.createElement('style');
styleElement.textContent = samPanelStyles;
document.head.appendChild(styleElement);

export { SamPanel };
