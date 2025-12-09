import { Container, Label } from '@playcanvas/pcui';
import { Mat4, path, Vec3 } from 'playcanvas';

import { DataPanel } from './data-panel';
import { Events } from '../events';
import { BottomToolbar } from './bottom-toolbar';
import { ColorPanel } from './color-panel';
import { ExportPopup } from './export-popup';
import { ImageSettingsDialog } from './image-settings-dialog';
import { localize, localizeInit } from './localization';
import { Menu } from './menu';
import { ModeToggle } from './mode-toggle';
import logo from './playcanvas-logo.png';
import { Popup, ShowOptions } from './popup';
import { Progress } from './progress';
import { PublishSettingsDialog } from './publish-settings-dialog';
import { RightToolbar } from './right-toolbar';
import { ScenePanel } from './scene-panel';
import { ShortcutsPopup } from './shortcuts-popup';
import { Spinner } from './spinner';
import { TimelinePanel } from './timeline-panel';
import { Tooltips } from './tooltips';
import { VideoSettingsDialog } from './video-settings-dialog';
import { ViewCube } from './view-cube';
import { SAMDialog } from '../sam/sam-dialog';
import { ViewPanel } from './view-panel';
import { version } from '../../package.json';

// ts compiler and vscode find this type, but eslint does not
type FilePickerAcceptType = unknown;

const removeExtension = (filename: string) => {
    return filename.substring(0, filename.length - path.getExtension(filename).length);
};

class EditorUI {
    appContainer: Container;
    topContainer: Container;
    canvasContainer: Container;
    toolsContainer: Container;
    canvas: HTMLCanvasElement;
    popup: Popup;

    constructor(events: Events) {
        // favicon
        const link = document.createElement('link');
        link.rel = 'icon';
        link.href = logo;
        document.head.appendChild(link);

        // app
        const appContainer = new Container({
            id: 'app-container'
        });

        // editor
        const editorContainer = new Container({
            id: 'editor-container'
        });

        // tooltips container
        const tooltipsContainer = new Container({
            id: 'tooltips-container'
        });

        // top container
        const topContainer = new Container({
            id: 'top-container'
        });

        // canvas
        const canvas = document.createElement('canvas');
        canvas.id = 'canvas';

        // app label
        const appLabel = new Label({
            id: 'app-label',
            text: `SUPERSPLAT v${version}`
        });

        // cursor label
        const cursorLabel = new Label({
            id: 'cursor-label'
        });

        let fullprecision = '';

        events.on('camera.focalPointPicked', (details: { position: Vec3 }) => {
            cursorLabel.text = `${details.position.x.toFixed(2)}, ${details.position.y.toFixed(2)}, ${details.position.z.toFixed(2)}`;
            fullprecision = `${details.position.x}, ${details.position.y}, ${details.position.z}`;
        });

        ['pointerdown', 'pointerup', 'pointermove', 'wheel', 'dblclick'].forEach((eventName) => {
            cursorLabel.dom.addEventListener(eventName, (event: Event) => event.stopPropagation());
        });

        cursorLabel.dom.addEventListener('pointerdown', () => {
            navigator.clipboard.writeText(fullprecision);

            const orig = cursorLabel.text;
            cursorLabel.text = localize('cursor.copied');
            setTimeout(() => {
                cursorLabel.text = orig;
            }, 1000);
        });

        // canvas container
        const canvasContainer = new Container({
            id: 'canvas-container'
        });

        // tools container
        const toolsContainer = new Container({
            id: 'tools-container'
        });

        // tooltips
        const tooltips = new Tooltips();
        tooltipsContainer.append(tooltips);

        // bottom toolbar
        const scenePanel = new ScenePanel(events, tooltips);
        const viewPanel = new ViewPanel(events, tooltips);
        const colorPanel = new ColorPanel(events, tooltips);
        const bottomToolbar = new BottomToolbar(events, tooltips);
        const rightToolbar = new RightToolbar(events, tooltips);
        const modeToggle = new ModeToggle(events, tooltips);
        const menu = new Menu(events);

        canvasContainer.dom.appendChild(canvas);
        canvasContainer.append(appLabel);
        canvasContainer.append(cursorLabel);
        canvasContainer.append(toolsContainer);
        canvasContainer.append(scenePanel);
        canvasContainer.append(viewPanel);
        canvasContainer.append(colorPanel);
        canvasContainer.append(bottomToolbar);
        canvasContainer.append(rightToolbar);
        canvasContainer.append(modeToggle);
        canvasContainer.append(menu);

        // view axes container
        const viewCube = new ViewCube(events);
        canvasContainer.append(viewCube);
        events.on('prerender', (cameraMatrix: Mat4) => {
            viewCube.update(cameraMatrix);
        });

        // main container
        const mainContainer = new Container({
            id: 'main-container'
        });

        const timelinePanel = new TimelinePanel(events, tooltips);
        const dataPanel = new DataPanel(events);

        mainContainer.append(canvasContainer);
        mainContainer.append(timelinePanel);
        mainContainer.append(dataPanel);

        editorContainer.append(mainContainer);

        tooltips.register(cursorLabel, localize('cursor.click-to-copy'), 'top');

        // message popup
        const popup = new Popup(tooltips);

        // shortcuts popup
        const shortcutsPopup = new ShortcutsPopup();

        // export popup
        const exportPopup = new ExportPopup(events);

        // publish settings
        const publishSettingsDialog = new PublishSettingsDialog(events);

        // image settings
        const imageSettingsDialog = new ImageSettingsDialog(events);

        // video settings
        const videoSettingsDialog = new VideoSettingsDialog(events);

        // SAM2 dialog
        const samDialog = new SAMDialog(events);

        topContainer.append(popup);
        topContainer.append(exportPopup);
        topContainer.append(publishSettingsDialog);
        topContainer.append(imageSettingsDialog);
        topContainer.append(videoSettingsDialog);
        topContainer.append(samDialog);

        appContainer.append(editorContainer);
        appContainer.append(topContainer);
        appContainer.append(tooltipsContainer);
        appContainer.append(shortcutsPopup);

        this.appContainer = appContainer;
        this.topContainer = topContainer;
        this.canvasContainer = canvasContainer;
        this.toolsContainer = toolsContainer;
        this.canvas = canvas;
        this.popup = popup;

        document.body.appendChild(appContainer.dom);
        document.body.setAttribute('tabIndex', '-1');

        events.on('show.shortcuts', () => {
            shortcutsPopup.hidden = false;
        });

        events.function('show.exportPopup', (exportType, splatNames: [string], showFilenameEdit: boolean) => {
            return exportPopup.show(exportType, splatNames, showFilenameEdit);
        });

        events.function('show.publishSettingsDialog', async () => {
            // show popup if user isn't logged in
            const userStatus = await events.invoke('publish.userStatus');
            if (!userStatus) {
                await events.invoke('showPopup', {
                    type: 'error',
                    header: localize('popup.error'),
                    message: localize('popup.publish.please-log-in')
                });
                return false;
            }

            // get user publish settings
            const publishSettings = await publishSettingsDialog.show(userStatus);

            // do publish
            if (publishSettings) {
                await events.invoke('scene.publish', publishSettings);
            }
        });

        events.function('show.imageSettingsDialog', async () => {
            const imageSettings = await imageSettingsDialog.show();

            if (imageSettings) {
                await events.invoke('render.image', imageSettings);
            }
        });

        events.function('show.videoSettingsDialog', async () => {
            const videoSettings = await videoSettingsDialog.show();

            if (videoSettings) {

                try {
                    const docName = events.invoke('doc.name');

                    // Determine file extension and mime type based on format
                    let fileExtension: string;
                    let filePickerTypes: FilePickerAcceptType[];

                    // Codec name mapping for display
                    const codecNames: Record<string, string> = {
                        'h264': 'H.264',
                        'h265': 'H.265',
                        'vp9': 'VP9',
                        'av1': 'AV1'
                    };
                    const codecName = codecNames[videoSettings.codec] || videoSettings.codec.toUpperCase();

                    if (videoSettings.format === 'webm') {
                        fileExtension = '.webm';
                        filePickerTypes = [{
                            description: `WebM Video (${codecName})`,
                            accept: { 'video/webm': ['.webm'] }
                        }];
                    } else if (videoSettings.format === 'mov') {
                        fileExtension = '.mov';
                        filePickerTypes = [{
                            description: `MOV Video (${codecName})`,
                            accept: { 'video/quicktime': ['.mov'] }
                        }];
                    } else if (videoSettings.format === 'mkv') {
                        fileExtension = '.mkv';
                        filePickerTypes = [{
                            description: `MKV Video (${codecName})`,
                            accept: { 'video/x-matroska': ['.mkv'] }
                        }];
                    } else {
                        fileExtension = '.mp4';
                        filePickerTypes = [{
                            description: `MP4 Video (${codecName})`,
                            accept: { 'video/mp4': ['.mp4'] }
                        }];
                    }

                    const suggested = `${removeExtension(docName ?? 'supersplat')}${fileExtension}`;

                    let writable;

                    if (window.showSaveFilePicker) {
                        const fileHandle = await window.showSaveFilePicker({
                            id: 'SuperSplatVideoFileExport',
                            types: filePickerTypes,
                            suggestedName: suggested
                        });

                        writable = await fileHandle.createWritable();
                    }

                    await events.invoke('render.video', videoSettings, writable);
                } catch (error) {
                    if (error instanceof DOMException && error.name === 'AbortError') {
                        // user cancelled save dialog
                        return;
                    }

                    await events.invoke('showPopup', {
                        type: 'error',
                        header: 'Failed to render video',
                        message: `'${error.message ?? error}'`
                    });
                }
            }
        });

        events.function('show.about', () => {
            return this.popup.show({
                type: 'info',
                header: 'About',
                message: `SUPERSPLAT v${version}`
            });
        });

        events.function('showPopup', (options: ShowOptions) => {
            return this.popup.show(options);
        });

        // spinner

        const spinner = new Spinner();

        topContainer.append(spinner);

        events.on('startSpinner', () => {
            spinner.hidden = false;
        });

        events.on('stopSpinner', () => {
            spinner.hidden = true;
        });

        // progress

        const progress = new Progress();

        topContainer.append(progress);

        events.on('progressStart', (header: string) => {
            progress.hidden = false;
            progress.setHeader(header);
        });

        events.on('progressUpdate', (options: { text: string, progress: number }) => {
            progress.setText(options.text);
            progress.setProgress(options.progress);
        });

        events.on('progressEnd', () => {
            progress.hidden = true;
        });

        // SAM2 tool event
        events.on('tool.sam2', async () => {
            console.log('SAM2 tool event fired!');
            const result = await samDialog.show();
            if (result) {
                console.log('[SAM2] Selection result received:', result);

                // Restore the camera pose to match the captured screenshot
                const { cameraPose, mask, originalWidth, originalHeight } = result;
                events.fire('camera.setPose', cameraPose);

                // SAM processes images at 1024x1024 and pads non-square images to square
                // We need to transform the mask from SAM's padded 1024x1024 space to screen coordinates
                const SAM_IMAGE_SIZE = 1024;

                // Calculate padding that was applied to make the original image square
                const largestDim = Math.max(originalWidth, originalHeight);
                const padX = (largestDim - originalWidth) / 2;
                const padY = (largestDim - originalHeight) / 2;

                // Create a canvas at screen resolution and transform the mask
                const screenMaskCanvas = document.createElement('canvas');
                screenMaskCanvas.width = originalWidth;
                screenMaskCanvas.height = originalHeight;
                const ctx = screenMaskCanvas.getContext('2d')!;

                // The mask is 1024x1024 but the actual image content is centered within it
                // We need to extract just the portion that corresponds to the original image
                // Scale factor from SAM space to original square space
                const scale = SAM_IMAGE_SIZE / largestDim;

                // Source rectangle in SAM 1024x1024 space (where the actual image content is)
                const srcX = padX * scale;
                const srcY = padY * scale;
                const srcW = originalWidth * scale;
                const srcH = originalHeight * scale;

                // Draw the relevant portion of the mask to screen size
                ctx.drawImage(
                    mask,
                    srcX, srcY, srcW, srcH,  // Source: portion of 1024x1024 mask that contains actual image
                    0, 0, originalWidth, originalHeight  // Dest: full screen canvas
                );

                // Convert mask format: SAM uses red=50 (0x32) for green visualization,
                // but select.byMask expects red=255 for selection detection
                const imageData = ctx.getImageData(0, 0, screenMaskCanvas.width, screenMaskCanvas.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    if (imageData.data[i + 3] > 0) {  // If alpha > 0 (masked pixel)
                        imageData.data[i] = 255;       // Set red to 255
                    }
                }
                ctx.putImageData(imageData, 0, 0);

                console.log('[SAM2] Mask transformed to screen coordinates:', {
                    originalWidth,
                    originalHeight,
                    largestDim,
                    padX,
                    padY,
                    scale,
                    srcX,
                    srcY,
                    srcW,
                    srcH
                });

                // Fire the select.byMask event with the transformed mask
                // The selection system expects: op ('set', 'add', 'remove'), canvas, context
                events.fire('select.byMask', 'set', screenMaskCanvas, ctx);

                console.log('[SAM2] select.byMask event fired');
            }
        });

        // initialize canvas to correct size before creating graphics device etc
        const pixelRatio = window.devicePixelRatio;
        canvas.width = Math.ceil(canvasContainer.dom.offsetWidth * pixelRatio);
        canvas.height = Math.ceil(canvasContainer.dom.offsetHeight * pixelRatio);

        ['contextmenu', 'gesturestart', 'gesturechange', 'gestureend'].forEach((event) => {
            document.addEventListener(event, (e) => {
                e.preventDefault();
            }, true);
        });

        // whenever the canvas container is clicked, set keyboard focus on the body
        canvasContainer.dom.addEventListener('pointerdown', (event: PointerEvent) => {
            // set focus on the body if user is busy pressing on the canvas or a child of the tools
            // element
            if (event.target === canvas || toolsContainer.dom.contains(event.target as Node)) {
                document.body.focus();
            }
        }, true);
    }
}

export { EditorUI };
