import { Events } from '../events';
import { SegmentationPoint } from '../segmentation/types';

/**
 * SAM Selection Tool - allows users to pick foreground/background points
 * for AI-powered segmentation.
 *
 * Usage:
 * - Left click: Add foreground point (include in selection)
 * - Right click / Ctrl+click: Add background point (exclude from selection)
 * - Press Enter or click "Segment" button: Trigger segmentation
 * - Press Escape or click "Cancel": Cancel and clear points
 */
class SamSelection {
    activate: () => void;
    deactivate: () => void;

    constructor(events: Events, parent: HTMLElement) {
        // Create SVG overlay for point visualization
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.classList.add('tool-svg', 'hidden');
        svg.id = 'sam-select-svg';
        parent.appendChild(svg);

        // Points collected for segmentation
        let points: SegmentationPoint[] = [];

        // Point radius for visualization
        const pointRadius = 8;

        /**
         * Render all points on the SVG
         */
        const paint = () => {
            // Clear existing elements
            while (svg.firstChild) {
                svg.removeChild(svg.firstChild);
            }

            // Draw each point
            points.forEach((point, index) => {
                const group = document.createElementNS(svg.namespaceURI, 'g');

                // Outer circle (border)
                const outerCircle = document.createElementNS(svg.namespaceURI, 'circle');
                outerCircle.setAttribute('cx', point.x.toString());
                outerCircle.setAttribute('cy', point.y.toString());
                outerCircle.setAttribute('r', (pointRadius + 2).toString());
                outerCircle.setAttribute('fill', 'white');
                outerCircle.setAttribute('stroke', 'black');
                outerCircle.setAttribute('stroke-width', '1');
                group.appendChild(outerCircle);

                // Inner circle (color indicates type)
                const innerCircle = document.createElementNS(svg.namespaceURI, 'circle');
                innerCircle.setAttribute('cx', point.x.toString());
                innerCircle.setAttribute('cy', point.y.toString());
                innerCircle.setAttribute('r', pointRadius.toString());
                innerCircle.setAttribute('fill', point.type === 'fg' ? '#22c55e' : '#ef4444'); // green for fg, red for bg
                innerCircle.setAttribute('stroke', 'white');
                innerCircle.setAttribute('stroke-width', '2');
                group.appendChild(innerCircle);

                // Number label
                const text = document.createElementNS(svg.namespaceURI, 'text');
                text.setAttribute('x', point.x.toString());
                text.setAttribute('y', (point.y + 4).toString());
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('fill', 'white');
                text.setAttribute('font-size', '10');
                text.setAttribute('font-weight', 'bold');
                text.setAttribute('pointer-events', 'none');
                text.textContent = (index + 1).toString();
                group.appendChild(text);

                svg.appendChild(group);
            });

            // Notify listeners of point change
            events.fire('sam.pointsChanged', [...points]);
        };

        /**
         * Clear all points
         */
        const clearPoints = () => {
            points = [];
            paint();
        };

        /**
         * Remove the last added point
         */
        const undoLastPoint = () => {
            if (points.length > 0) {
                points.pop();
                paint();
            }
        };

        /**
         * Handle pointer down - add a point and trigger immediate segmentation
         */
        const pointerdown = (e: PointerEvent) => {
            // Only handle primary button (left click) or right click
            if (e.pointerType === 'mouse' && e.button !== 0 && e.button !== 2) {
                return;
            }

            e.preventDefault();
            e.stopPropagation();

            // Determine point type: right click or ctrl+click = background, otherwise foreground
            const isBackground = e.button === 2 || e.ctrlKey;

            const newPoint: SegmentationPoint = {
                x: e.offsetX,
                y: e.offsetY,
                type: isBackground ? 'bg' : 'fg'
            };

            points.push(newPoint);
            paint();

            // Trigger immediate segmentation like the original SAM2 repo
            events.fire('sam.segment', [...points]);
        };

        /**
         * Prevent context menu on right click
         */
        const contextmenu = (e: Event) => {
            e.preventDefault();
            e.stopPropagation();
        };

        /**
         * Handle keyboard shortcuts
         */
        const keydown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                // Cancel - clear points and deactivate
                clearPoints();
                events.fire('sam.cancelled');
            } else if (e.key === 'Enter') {
                // Trigger segmentation
                if (points.length > 0) {
                    events.fire('sam.segment', [...points]);
                }
            } else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
                // Undo last point
                e.preventDefault();
                undoLastPoint();
            }
        };

        this.activate = () => {
            // Option B: Hide viewport overlay entirely - all clicks happen on the panel's preview canvas
            // Keep SVG hidden since we won't draw points on the main viewport
            svg.classList.add('hidden');
            parent.style.display = 'none';
            
            // Don't add viewport pointerdown handlers - clicks go to panel instead
            // But keep keyboard shortcuts active
            document.addEventListener('keydown', keydown);

            // Clear any previous points
            clearPoints();

            // Notify that SAM tool is active
            events.fire('sam.activated');

            // Immediately capture viewport preview when tool opens
            // This shows the current view in the panel and starts pre-encoding
            events.fire('sam.capturePreview');
        };

        this.deactivate = () => {
            svg.classList.add('hidden');
            parent.style.display = 'none';
            document.removeEventListener('keydown', keydown);

            // Clear points on deactivation
            clearPoints();

            // Notify that SAM tool is deactivated
            events.fire('sam.deactivated');
        };

        // Listen for external commands
        events.on('sam.clearPoints', () => {
            clearPoints();
        });

        events.on('sam.undoPoint', () => {
            undoLastPoint();
        });

        // Allow adding points programmatically
        events.on('sam.addPoint', (point: SegmentationPoint) => {
            points.push(point);
            paint();
        });

        // Get current points
        events.function('sam.getPoints', () => {
            return [...points];
        });
    }
}

export { SamSelection };
