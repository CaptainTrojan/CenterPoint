import onnx
import onnx_graphsurgeon as gs
import argparse
import onnxruntime as ort
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, default='onnx_model/pointpillars.onnx', help='Path to the ONNX model')
args = parser.parse_args()

model_path = args.model_path
model = onnx.load(model_path)

# Check the model
try:
    onnx.checker.check_model(model)
    print("The model is valid.")
except onnx.checker.ValidationError as e:
    print("Model validation error:", e)

# Try to do inference with the model
try:
    session = ort.InferenceSession(model_path)
    example_inputs = {input_name: np.random.rand(*input.shape).astype(np.float32) for input_name, input in zip(input_names, session.get_inputs())}
    outputs = session.run(None, example_inputs)
    print("Inference successful.")
except Exception as e:
    print("Inference failed:", e)

# Graph Surgeon to inspect nodes
graph = gs.import_onnx(model)
graph.toposort()
node_names = {node.name for node in graph.nodes}
input_names = {input.name for input in graph.inputs}
output_names = {output.name for output in graph.outputs}

# Check for any node inputs not produced by preceding nodes
# for node in graph.nodes:
#     for input_tensor in node.inputs:
#         if input_tensor.name not in node_names and input_tensor.name not in input_names:
#             print(f"Node {node.name} has an input {input_tensor.name} that is not produced by any preceding node.")

# Optionally, print graph nodes for further inspection
for node in graph.nodes:
    print(node)

# Save the potentially fixed model
onnx.save(gs.export_onnx(graph), 'fixed_model.onnx')
