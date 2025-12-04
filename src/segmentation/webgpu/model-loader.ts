/**
 * SAM2 model loading with IndexedDB caching for persistence.
 */

/** Model configuration */
export interface ModelConfig {
    /** Human-readable name */
    name: string;
    /** URL to download the model */
    url: string;
    /** Expected file size in bytes (for progress calculation) */
    expectedSize: number;
}

/** Download progress callback */
export type ProgressCallback = (loaded: number, total: number, modelName: string) => void;

/** Default SAM2 tiny model URLs from HuggingFace */
export const SAM2_MODELS: Record<string, ModelConfig> = {
    encoder: {
        name: 'SAM2 Encoder',
        url: 'https://huggingface.co/g-ronimo/sam2-tiny/resolve/main/sam2_hiera_tiny_encoder.with_runtime_opt.ort',
        expectedSize: 42 * 1024 * 1024 // ~42MB
    },
    decoder: {
        name: 'SAM2 Decoder',
        url: 'https://huggingface.co/g-ronimo/sam2-tiny/resolve/main/sam2_hiera_tiny_decoder_pr1.onnx',
        expectedSize: 15 * 1024 * 1024 // ~15MB
    }
};

/** IndexedDB database name */
const DB_NAME = 'supersplat-sam2-models';
const DB_VERSION = 1;
const STORE_NAME = 'models';

/**
 * Open IndexedDB database for model storage.
 */
function openDatabase(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => {
            reject(new Error(`Failed to open IndexedDB: ${request.error?.message}`));
        };

        request.onsuccess = () => {
            resolve(request.result);
        };

        request.onupgradeneeded = (event) => {
            const db = (event.target as IDBOpenDBRequest).result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };
    });
}

/**
 * Get model from IndexedDB cache.
 */
async function getCachedModel(modelKey: string): Promise<ArrayBuffer | null> {
    try {
        const db = await openDatabase();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(STORE_NAME, 'readonly');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.get(modelKey);

            request.onerror = () => {
                reject(new Error(`Failed to read from cache: ${request.error?.message}`));
            };

            request.onsuccess = () => {
                resolve(request.result || null);
            };

            transaction.oncomplete = () => {
                db.close();
            };
        });
    } catch (error) {
        console.warn('IndexedDB cache read failed:', error);
        return null;
    }
}

/**
 * Store model in IndexedDB cache.
 */
async function cacheModel(modelKey: string, data: ArrayBuffer): Promise<void> {
    try {
        const db = await openDatabase();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(STORE_NAME, 'readwrite');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.put(data, modelKey);

            request.onerror = () => {
                reject(new Error(`Failed to write to cache: ${request.error?.message}`));
            };

            request.onsuccess = () => {
                resolve();
            };

            transaction.oncomplete = () => {
                db.close();
            };
        });
    } catch (error) {
        console.warn('IndexedDB cache write failed:', error);
        // Don't throw - caching is optional
    }
}

/**
 * Check if models are cached in IndexedDB.
 */
export async function areModelsCached(): Promise<boolean> {
    try {
        const encoder = await getCachedModel('encoder');
        const decoder = await getCachedModel('decoder');
        return encoder !== null && decoder !== null;
    } catch {
        return false;
    }
}

/**
 * Get cached model sizes (for UI display).
 */
export async function getCachedModelInfo(): Promise<{ encoder: number; decoder: number } | null> {
    try {
        const encoder = await getCachedModel('encoder');
        const decoder = await getCachedModel('decoder');
        if (encoder && decoder) {
            return {
                encoder: encoder.byteLength,
                decoder: decoder.byteLength
            };
        }
        return null;
    } catch {
        return null;
    }
}

/**
 * Clear cached models from IndexedDB.
 */
export async function clearModelCache(): Promise<void> {
    try {
        const db = await openDatabase();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(STORE_NAME, 'readwrite');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.clear();

            request.onerror = () => {
                reject(new Error(`Failed to clear cache: ${request.error?.message}`));
            };

            request.onsuccess = () => {
                resolve();
            };

            transaction.oncomplete = () => {
                db.close();
            };
        });
    } catch (error) {
        console.warn('Failed to clear model cache:', error);
    }
}

/**
 * Download a model with progress tracking.
 */
async function downloadModel(
    config: ModelConfig,
    onProgress?: ProgressCallback,
    signal?: AbortSignal
): Promise<ArrayBuffer> {
    const response = await fetch(config.url, { signal });

    if (!response.ok) {
        throw new Error(`Failed to download ${config.name}: ${response.status} ${response.statusText}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : config.expectedSize;

    if (!response.body) {
        // Fallback for browsers without streaming support
        const buffer = await response.arrayBuffer();
        onProgress?.(buffer.byteLength, buffer.byteLength, config.name);
        return buffer;
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;

    while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        chunks.push(value);
        loaded += value.length;
        onProgress?.(loaded, total, config.name);
    }

    // Combine chunks into single ArrayBuffer
    const buffer = new ArrayBuffer(loaded);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) {
        view.set(chunk, offset);
        offset += chunk.length;
    }

    return buffer;
}

/**
 * Load a single model (from cache or download).
 */
export async function loadModel(
    modelKey: 'encoder' | 'decoder',
    onProgress?: ProgressCallback,
    signal?: AbortSignal
): Promise<ArrayBuffer> {
    // Try cache first
    const cached = await getCachedModel(modelKey);
    if (cached) {
        const config = SAM2_MODELS[modelKey];
        onProgress?.(cached.byteLength, cached.byteLength, config.name);
        return cached;
    }

    // Download and cache
    const config = SAM2_MODELS[modelKey];
    const data = await downloadModel(config, onProgress, signal);
    await cacheModel(modelKey, data);
    return data;
}

/**
 * Load both encoder and decoder models.
 */
export async function loadAllModels(
    onProgress?: (loaded: number, total: number, stage: string) => void,
    signal?: AbortSignal
): Promise<{ encoder: ArrayBuffer; decoder: ArrayBuffer }> {
    const totalSize = SAM2_MODELS.encoder.expectedSize + SAM2_MODELS.decoder.expectedSize;
    let totalLoaded = 0;
    let currentModel = '';

    const trackProgress: ProgressCallback = (loaded, total, modelName) => {
        if (modelName !== currentModel) {
            // Starting new model - add previous model's size
            if (currentModel === SAM2_MODELS.encoder.name) {
                totalLoaded = SAM2_MODELS.encoder.expectedSize;
            }
            currentModel = modelName;
        }

        const overallLoaded = currentModel === SAM2_MODELS.encoder.name
            ? loaded
            : SAM2_MODELS.encoder.expectedSize + loaded;

        onProgress?.(overallLoaded, totalSize, modelName);
    };

    currentModel = SAM2_MODELS.encoder.name;
    const encoder = await loadModel('encoder', trackProgress, signal);

    currentModel = SAM2_MODELS.decoder.name;
    const decoder = await loadModel('decoder', trackProgress, signal);

    return { encoder, decoder };
}

/**
 * Format bytes for display.
 */
export function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Get total model size for display.
 */
export function getTotalModelSize(): number {
    return SAM2_MODELS.encoder.expectedSize + SAM2_MODELS.decoder.expectedSize;
}
