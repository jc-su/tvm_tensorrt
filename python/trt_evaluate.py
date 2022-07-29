import argparse
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit


TRT_LOGGER = trt.Logger()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    
def allocate_buffers(engine):
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


def evaluate(engine, host_input):
    cuda.init()
    device = cuda.Device(0)  # enter your Gpu id here
    ctx = device.make_context()
    stream = cuda.Stream()
    inputs, outputs, bindings = allocate_buffers(engine)
    ctx.pop()
    inputs[0].host = host_input
    context = engine.create_execution_context()
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream)
        for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream)
        for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    outputs = [out.host for out in outputs]
    return np.argmax(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorRT Inference')
    parser.add_argument('--engine', type=str, default='resnet-18-fp32.engine',
                        help='Path to serialized engine file')

    args = parser.parse_args()

    engine = load_engine(args.engine)
    # input_shape = engine.get_binding_shape()
    # input_shape = (1, 3, input_shape[2], input_shape[3])
    # print(input_shape)
    input_shape = (1, 3, 224, 224)
    input_data = np.random.rand(*input_shape)
    print(evaluate(engine, input_data))

    