import os
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import time
import argparse


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
INPUT_SIZE = 224

# logger to sourceture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TensorRTInference:
    def __init__(self, onnx_file_path, engine_file_path,
                 workspace=1 << 30, batch_size=1, fp16=False):

        # Build Engine
        self.engine = self.get_engine(
            onnx_file_path, engine_file_path, workspace, batch_size, fp16)

        # Set Context
        self.context = self.engine.create_execution_context()
        # Create stream to copy inputs/outputs
        self.stream = cuda.Stream()

    def get_engine(self, onnx_file_path, engine_file_path,
                   workspace, batch_size, fp16):

        precision = 'fp32'
        if fp16:
            precision = 'fp16'

        def build_engine():

            builder = trt.Builder(TRT_LOGGER)
            config = builder.create_builder_config()
            network = builder.create_network(EXPLICIT_BATCH)
            parser = trt.OnnxParser(network, TRT_LOGGER)
            runtime = trt.Runtime(TRT_LOGGER)

            # allow TensorRT to use up to 1GB of GPU memory for tactic selection
            config.max_workspace_size = workspace
            # we have only one image in batch
            builder.max_batch_size = batch_size
            # use FP16 mode if possible
            if precision == 'fp16' and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            # parse ONNX
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('[Engine] ERROR: Failed to parse ONNX file')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
                else:
                    print('[Engine] Completed parsing of ONNX file')
            profile = builder.create_optimization_profile()     
            profile.set_shape('data', (batch_size, 3, INPUT_SIZE, INPUT_SIZE), (batch_size, 3, INPUT_SIZE, INPUT_SIZE), (batch_size, 3, INPUT_SIZE, INPUT_SIZE))
            config.add_optimization_profile(profile)
            # generate TensorRT engine optimized for the target platform
            print('[Engine] Building an engine')
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            # engine = builder.build_cuda_engine(network, config)

            print('[Engine] Completed creating Engine')

            with open(engine_file_path.format(batch_size, precision), 'wb') as f:
                f.write(engine.serialize())

            print('[Engine] Engine burned on disk as {}'.format(
                engine_file_path.format(batch_size, precision)))

            return engine

        if os.path.exists(engine_file_path.format(batch_size, precision)):
            # If a serialized engine exists, use it instead of building an engine.
            print("[Engine] Reading engine from file {}".format(
                engine_file_path.format(batch_size, precision)))

            with open(engine_file_path.format(batch_size, precision), "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

        else:
            return build_engine()

    def allocate_buffers(self, engine):
        inputs = list()
        outputs = list()
        bindings = list()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * \
                engine.max_batch_size * np.dtype(np.float32).itemsize
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    def run_inference(self, host_input):

        inputs, outputs, bindings = self.allocate_buffers(self.engine)
        inputs[0].host = host_input

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
         for inp in inputs]
        # Run inference.
        self.context.execute_async_v2(
            bindings=bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
         for out in outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.

        outputs = [out.host for out in outputs]

        return np.argmax(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorRT Inference')
    parser.add_argument('--onnx', type=str, default='models/resnet18-v2.onnx',
                        help='Path to ONNX file')
    parser.add_argument('--engine', type=str, default='models/resnet18-v2-python.engine',
                        help='Path to serialized engine file')
    parser.add_argument('--workspace', type=int, default=1 << 30,
                        help='Workspace size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16')
    parser.add_argument('--input', type=str, default='',
                        help='Path to input image')
    args = parser.parse_args()

    trt_inference = TensorRTInference(
        args.onnx, args.engine, args.workspace, args.batch_size, args.fp16)

    # image = cv2.imread(args.input)
    # blob = cv2.dnn.blobFromImage(
    #     image, 1.0, (224, 224), (0, 0, 0), False, False)
    

    # blob = np.transpose(blob, (1, 0, 2, 3))
    # blob = np.expand_dims(blob, axis=0)
    # blob = np.ascontiguousarray(blob)
    # random input 
    blob = np.random.rand(1, 3, 224, 224)
    mesurement_res = []
    for i in range(600):
        start = time.time()
        result = trt_inference.run_inference(blob)
        mesurement_res.append(time.time() - start)
    print(f'Average inference time: {np.mean(mesurement_res)*1000} ms')
