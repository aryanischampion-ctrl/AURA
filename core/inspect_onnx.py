import os
import onnx


MODEL_PATH = "models/aura_intent_model_int8.onnx"


model = onnx.load(MODEL_PATH)

onnx.checker.check_model(model)


print("AURA ONNX MODEL CHECK")
print("=" * 40)

print("Model:", MODEL_PATH)
print(
    "Size:",
    round(os.path.getsize(MODEL_PATH) / 1024, 2),
    "KB"
)

print("\nInputs:")

for tensor in model.graph.input:
    print(
        " -",
        tensor.name,
        tensor.type.tensor_type.shape
    )

print("\nOutputs:")

for tensor in model.graph.output:
    print(
        " -",
        tensor.name,
        tensor.type.tensor_type.shape
    )

print("\nOperators:")

operators = sorted(
    set(node.op_type for node in model.graph.node)
)

for operator in operators:
    print(" -", operator)

print("\nTotal graph nodes:", len(model.graph.node))

print("\nONNX model is valid.")