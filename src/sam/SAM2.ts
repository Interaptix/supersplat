import * as ort from 'onnxruntime-web';

import {
    visualizeEncoderEmbedding,
    visualizeModelOutput,
    visualizeDecoderInput,
    extractBestMask,
    DEBUG_SAM2,
    debugLog
} from './debug-utils';

const ENCODER_URL =
    'https://huggingface.co/g-ronimo/sam2-tiny/resolve/main/sam2_hiera_tiny_encoder.with_runtime_opt.ort';
const DECODER_URL =
    'https://huggingface.co/g-ronimo/sam2-tiny/resolve/main/sam2_hiera_tiny_decoder_pr1.onnx';

interface Point {
    x: number;
    y: number;
    label: number;
}

interface ImageEncoded {
    high_res_feats_0: ort.Tensor;
    high_res_feats_1: ort.Tensor;
    image_embed: ort.Tensor;
}

type SessionInfo = [ort.InferenceSession, string];

/**
 * Get filename from URL (replacement for Node.js path.basename)
 * @param {string} url - The URL to extract filename from
 * @returns {string} The filename extracted from the URL
 */
function getFilenameFromUrl(url: string): string {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;
    return pathname.substring(pathname.lastIndexOf('/') + 1);
}

export class SAM2 {
    bufferEncoder: ArrayBuffer | null = null;
    bufferDecoder: ArrayBuffer | null = null;
    sessionEncoder: SessionInfo | null = null;
    sessionDecoder: SessionInfo | null = null;
    image_encoded: ImageEncoded | null = null;


    async downloadModels(): Promise<void> {
        this.bufferEncoder = await this.downloadModel(ENCODER_URL);
        this.bufferDecoder = await this.downloadModel(DECODER_URL);
    }

    async downloadModel(url: string): Promise<ArrayBuffer | null> {
        // step 1: check if cached
        const root = await navigator.storage.getDirectory();
        const filename = getFilenameFromUrl(url);

        let fileHandle: FileSystemFileHandle | undefined;
        try {
            fileHandle = await root.getFileHandle(filename);
        } catch (e) {
            console.error('File does not exist:', filename, e);
        }

        if (fileHandle) {
            const file = await fileHandle.getFile();
            if (file.size > 0) return await file.arrayBuffer();
        }

        // step 2: download if not cached
        console.log(`File not in cache, downloading from ${url}`);
        let buffer: ArrayBuffer | null = null;
        try {
            buffer = await fetch(url, {
                headers: new Headers({
                    Origin: location.origin
                }),
                mode: 'cors'
            }).then(response => response.arrayBuffer());
        } catch (e) {
            console.error(`Download of ${url} failed: `, e);
            return null;
        }

        // step 3: store
        try {
            const newFileHandle = await root.getFileHandle(filename, { create: true });
            const writable = await newFileHandle.createWritable();
            await writable.write(buffer);
            await writable.close();

            console.log(`Stored ${filename}`);
        } catch (e) {
            console.error(`Storage of ${filename} failed: `, e);
        }

        return buffer;
    }

    async createSessions(): Promise<{ success: boolean; device: string | null }> {
        const success =
            (await this.getEncoderSession()) && (await this.getDecoderSession());

        return {
            success: !!success,
            device: success ? this.sessionEncoder![1] : null
        };
    }

    async getORTSession(model: ArrayBuffer): Promise<SessionInfo | null> {
        // Creating a session with executionProviders: {"webgpu", "cpu"} fails
        // => "Error: multiple calls to 'initWasm()' detected."
        // but ONLY in Safari and Firefox (wtf)
        // seems to be related to web worker, see https://github.com/microsoft/onnxruntime/issues/22113
        // => loop through each ep, catch e if not available and move on
        let session: ort.InferenceSession | null = null;
        for (const ep of ['webgpu', 'cpu'] as const) {
            try {
                session = await ort.InferenceSession.create(model, {
                    executionProviders: [ep]
                });
            } catch (e) {
                console.error(e);
                continue;
            }

            return [session, ep];
        }
        return null;
    }

    async getEncoderSession(): Promise<SessionInfo | null> {
        if (!this.sessionEncoder && this.bufferEncoder) {
            this.sessionEncoder = await this.getORTSession(this.bufferEncoder);
        }
        return this.sessionEncoder;
    }

    async getDecoderSession(): Promise<SessionInfo | null> {
        if (!this.sessionDecoder && this.bufferDecoder) {
            this.sessionDecoder = await this.getORTSession(this.bufferDecoder);
        }
        return this.sessionDecoder;
    }

    async encodeImage(inputTensor: ort.Tensor): Promise<void> {
        const sessionInfo = await this.getEncoderSession();
        if (!sessionInfo) throw new Error('Encoder session not available');

        const [session] = sessionInfo;
        const results = await session.run({ image: inputTensor });

        this.image_encoded = {
            high_res_feats_0: results[session.outputNames[0]],
            high_res_feats_1: results[session.outputNames[1]],
            image_embed: results[session.outputNames[2]]
        };

        // Debug visualization: show encoder embedding
        if (DEBUG_SAM2) {
            const embed = this.image_encoded.image_embed;
            await visualizeEncoderEmbedding(embed.data as Float32Array, embed.dims as number[]);
        }
    }

    async decode(points: Point[], masks?: ort.Tensor): Promise<ort.InferenceSession.OnnxValueMapType> {
        const sessionInfo = await this.getDecoderSession();
        if (!sessionInfo) throw new Error('Decoder session not available');
        if (!this.image_encoded) throw new Error('Image not encoded');

        const [session] = sessionInfo;

        const flatPoints = points.map((point) => {
            return [point.x, point.y];
        });

        const flatLabels = points.map((point) => {
            return point.label;
        });

        debugLog({
            flatPoints,
            flatLabels,
            masks
        });

        let mask_input: ort.Tensor;
        let has_mask_input: ort.Tensor;
        if (masks) {
            mask_input = masks;
            has_mask_input = new ort.Tensor('float32', [1], [1]);
        } else {
            // dummy data
            mask_input = new ort.Tensor(
                'float32',
                new Float32Array(256 * 256),
                [1, 1, 256, 256]
            );
            has_mask_input = new ort.Tensor('float32', [0], [1]);
        }

        const inputs = {
            image_embed: this.image_encoded.image_embed,
            high_res_feats_0: this.image_encoded.high_res_feats_0,
            high_res_feats_1: this.image_encoded.high_res_feats_1,
            point_coords: new ort.Tensor('float32', flatPoints.flat(), [
                1,
                flatPoints.length,
                2
            ]),
            point_labels: new ort.Tensor('float32', flatLabels, [
                1,
                flatLabels.length
            ]),
            mask_input: mask_input,
            has_mask_input: has_mask_input
        };

        debugLog('Decoder inputs:', inputs);

        // DEBUG: Log detailed tensor stats for comparison (no-op when DEBUG_SAM2 is false)
        const logTensorStats = DEBUG_SAM2 ?
            (name: string, tensor: ort.Tensor) => {
                const data = tensor.data as Float32Array;
                let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
                let zeroCount = 0;
                for (let i = 0; i < data.length; i++) {
                    const v = data[i];
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sum += v;
                    sumSq += v * v;
                    if (v === 0) zeroCount++;
                }
                const mean = sum / data.length;
                const variance = (sumSq / data.length) - (mean * mean);
                const std = Math.sqrt(Math.max(0, variance));
                console.log(`[SAM2 TENSOR] ${name}: shape=[${tensor.dims.join(',')}], min=${min.toFixed(6)}, max=${max.toFixed(6)}, mean=${mean.toFixed(6)}, std=${std.toFixed(6)}, zeros=${zeroCount}/${data.length}`);

                // Log first few values for exact matching
                const firstFew = Array.from(data.slice(0, 10)).map(v => v.toFixed(6)).join(', ');
                console.log(`[SAM2 TENSOR] ${name} data[0..9]: [${firstFew}]`);
            } :
            () => {};

        logTensorStats('image_embed', this.image_encoded.image_embed);
        logTensorStats('point_coords', inputs.point_coords);
        logTensorStats('point_labels', inputs.point_labels);
        logTensorStats('mask_input', inputs.mask_input);
        logTensorStats('has_mask_input', inputs.has_mask_input);
        logTensorStats('high_res_feats_0', this.image_encoded.high_res_feats_0);
        logTensorStats('high_res_feats_1', this.image_encoded.high_res_feats_1);

        // Debug visualization: show decoder inputs (mask_input + points)
        if (DEBUG_SAM2) {
            await visualizeDecoderInput(mask_input.data as Float32Array, (has_mask_input.data as Float32Array)[0] === 1, points);
        }

        const results = await session.run(inputs);

        // Debug visualization: show decoder output mask
        if (DEBUG_SAM2) {
            const { mask, logits, width, height, iouScore } = extractBestMask(results);
            console.log(`[SAM2 DEBUG] Decoder output - IoU score: ${iouScore?.toFixed(4) || 'N/A'}`);
            await visualizeModelOutput(mask, width, height, logits, points);
        }

        return results;
    }
}
