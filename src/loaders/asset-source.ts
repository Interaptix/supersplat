interface AssetSource {
    filename?: string;
    url?: string;
    contents?: File;
    animationFrame?: boolean;                                   // animations disable morton re-ordering at load time for faster loading
    mapUrl?: (name: string) => string;                          // function to map texture names to URLs
    mapFile?: (name: string) => AssetSource | null;             // function to map names to files
}

const fetchRequest = async (assetSource: AssetSource): Promise<Response | File> => {
    if (assetSource.contents) {
        return assetSource.contents;
    }
    
    const url = assetSource.url || assetSource.filename;
    
    try {
        const response = await fetch(url);
        return response;
    } catch (error) {
        // Network errors, CORS errors, etc.
        throw new Error(`Network error: ${error.message || 'Failed to fetch resource'}`);
    }
};

const fetchArrayBuffer = async (assetSource: AssetSource): Promise<ArrayBuffer> => {
    const response = await fetchRequest(assetSource);

    if (response instanceof Response) {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.arrayBuffer();
    }

    if (response instanceof File) {
        return await response.arrayBuffer();
    }

    throw new Error('Invalid response type');
};

const fetchText = async (assetSource: AssetSource): Promise<string> => {
    const response = await fetchRequest(assetSource);

    if (response instanceof Response) {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.text();
    }

    if (response instanceof File) {
        return await response.text();
    }

    throw new Error('Invalid response type');
};

export type { AssetSource };

export { fetchArrayBuffer, fetchText };
