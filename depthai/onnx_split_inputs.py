import onnx
import onnx_graphsurgeon as gs
import numpy as np

# Load the ONNX model
model = onnx.load("onnx_model/pointpillars.onnx")
graph = gs.import_onnx(model)

# Identify the first input (feats_input)
feats_input = graph.inputs[0]

# Create 10 new input variables (for splitting the input into smaller parts)
new_inputs_feats = [
    gs.Variable(name=f"feats_{i}", dtype=feats_input.dtype, shape=(1, 10, 3000, 20)) for i in range(10)
]

# Update the graph inputs: use the 10 new inputs and keep the second input unchanged
graph.inputs = new_inputs_feats + [graph.inputs[1]]

# Create a Concat node to concatenate the split parts along axis 2
concat_feats = gs.Node(op="Concat", attrs={'axis': 2}, inputs=new_inputs_feats)

# Define the output of the Concat node (shape matches the original feats_input)
concat_feats_output = gs.Variable(name="concat_feats_output", dtype=feats_input.dtype, shape=(1, 10, 30000, 20))

# Set the output of the Concat node
concat_feats.outputs = [concat_feats_output]

# Insert the Concat node earlier in the graph to ensure topological order
graph.nodes.insert(0, concat_feats)

# Replace the original input (feats_input) in the graph with the Concat output globally
for node in graph.nodes:
    for i, inp in enumerate(node.inputs):
        if inp == feats_input:
            node.inputs[i] = concat_feats_output  # Replace with the Concat output

# Topologically sort the graph
graph.toposort()
onnx_model = gs.export_onnx(graph)

# Save the updated ONNX model
onnx.save(onnx_model, "onnx_model/pointpillars_split.onnx")

# Load and check the model
model = onnx.load("onnx_model/pointpillars_split.onnx")
try:
    onnx.checker.check_model(model)
    print("The model is valid.")
except onnx.checker.ValidationError as e:
    print("Model validation error:", e)
