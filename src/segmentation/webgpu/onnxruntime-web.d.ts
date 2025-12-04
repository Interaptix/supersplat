/**
 * Type declarations for onnxruntime-web.
 * 
 * The onnxruntime-web package has a known issue where its types.d.ts
 * isn't properly exported via package.json "exports" field.
 * This declaration file provides the necessary types for our usage.
 */

declare module 'onnxruntime-web' {
    /**
     * Tensor class for creating and manipulating tensors.
     */
    export class Tensor {
        constructor(
            type: Tensor.Type,
            data: Tensor.DataType | readonly number[] | readonly bigint[] | readonly boolean[],
            dims?: readonly number[]
        );
        
        readonly type: Tensor.Type;
        readonly dims: readonly number[];
        readonly data: Tensor.DataType;
        readonly size: number;
        
        reshape(dims: readonly number[]): Tensor;
        toDataURL(options?: Tensor.DataUrlOptions): string;
        dispose(): void;
    }

    export namespace Tensor {
        type Type = 
            | 'float32' | 'float64' 
            | 'int8' | 'int16' | 'int32' | 'int64' 
            | 'uint8' | 'uint16' | 'uint32' | 'uint64'
            | 'bool' | 'string';
        
        type DataType = 
            | Float32Array | Float64Array
            | Int8Array | Int16Array | Int32Array | BigInt64Array
            | Uint8Array | Uint16Array | Uint32Array | BigUint64Array
            | string[];

        interface DataUrlOptions {
            format?: 'RGB' | 'BGR' | 'RBG' | 'GRB' | 'GBR' | 'BRG' | 'RGBA' | 'BGRA';
            tensorLayout?: 'NHWC' | 'NCHW';
        }
    }

    /**
     * Inference Session for running ONNX models.
     */
    export class InferenceSession {
        static create(
            uriOrBuffer: string | ArrayBufferLike | Uint8Array,
            options?: InferenceSession.SessionOptions
        ): Promise<InferenceSession>;

        run(
            feeds: InferenceSession.OnnxValueMapType,
            options?: InferenceSession.RunOptions
        ): Promise<InferenceSession.OnnxValueMapType>;

        run(
            feeds: InferenceSession.OnnxValueMapType,
            fetches: InferenceSession.FetchesType,
            options?: InferenceSession.RunOptions
        ): Promise<InferenceSession.OnnxValueMapType>;

        release(): Promise<void>;

        readonly inputNames: readonly string[];
        readonly outputNames: readonly string[];
    }

    export namespace InferenceSession {
        interface SessionOptions {
            executionProviders?: readonly ExecutionProviderConfig[];
            graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
            enableCpuMemArena?: boolean;
            enableMemPattern?: boolean;
            executionMode?: 'sequential' | 'parallel';
            logId?: string;
            logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
            logVerbosityLevel?: number;
            extra?: Record<string, unknown>;
        }

        interface RunOptions {
            logId?: string;
            logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
            logVerbosityLevel?: number;
            terminate?: boolean;
            extra?: Record<string, unknown>;
        }

        type ExecutionProviderConfig = 
            | 'webgpu' 
            | 'wasm' 
            | 'webgl' 
            | 'cpu' 
            | 'webnn'
            | WebGPUExecutionProviderConfig
            | WebNNExecutionProviderConfig
            | WasmExecutionProviderConfig;

        interface WebGPUExecutionProviderConfig {
            name: 'webgpu';
            preferredLayout?: 'NCHW' | 'NHWC';
            device?: GPUDevice;
        }

        interface WebNNExecutionProviderConfig {
            name: 'webnn';
            deviceType?: 'cpu' | 'gpu' | 'npu';
            powerPreference?: 'default' | 'low-power' | 'high-performance';
        }

        interface WasmExecutionProviderConfig {
            name: 'wasm';
        }

        type OnnxValueMapType = { readonly [name: string]: Tensor };
        type FetchesType = readonly string[] | { readonly [name: string]: Tensor | null };
    }

    /**
     * Environment configuration for ONNX Runtime Web.
     */
    export const env: Env;

    interface Env {
        wasm: {
            wasmPaths?: string | { [key: string]: string };
            numThreads?: number;
            simd?: boolean;
            proxy?: boolean;
        };
        webgpu: {
            powerPreference?: 'default' | 'low-power' | 'high-performance';
            profilingMode?: 'default' | 'off';
        };
        logLevel?: 'verbose' | 'info' | 'warning' | 'error' | 'fatal';
        debug?: boolean;
    }
}
