import depthai as dai
import numpy as np

print(f"DepthAI version: {dai.__version__}")

IP = "10.12.121.169"
info = dai.DeviceInfo(IP)
info.protocol = dai.X_LINK_TCP_IP
info.state = dai.X_LINK_GATE
info.platform = dai.X_LINK_RVC4
device = dai.Device(info)
pipeline = dai.Pipeline(device)

network = pipeline.create(dai.node.NeuralNetwork)
# network.setModelPath("float16_cached.dlc")

in_queue = network.input.createInputQueue()
out_queue = network.out.createOutputQueue()

pipeline.start()

input_data = dai.NNData()
features = np.random.rand(1, 10, 30000, 20)
input_data.addTensor("input.1", features)
indices = np.random.rand(1, 30000, 2)
input_data.addTensor("indices_input", indices)
in_queue.send(input_data)  # <-- here is where the error occurs

print(out_queue.get().getFirstLayerInt32())
