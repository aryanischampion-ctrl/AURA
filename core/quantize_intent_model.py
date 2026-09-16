import os
from onnxruntime.quantization import quantize_dynamic, QuantType

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "aura_intent_model.onnx"
)

OUTPUT_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "aura_intent_model_int8.onnx"
)

print("AURA INT8 QUANTIZATION")
print("=" * 40)

quantize_dynamic(
    INPUT_MODEL,
    OUTPUT_MODEL,
    weight_type=QuantType.QInt8
)

print("\nINT8 quantization completed.")
print(f"Original model: {INPUT_MODEL}")
print(f"INT8 model:     {OUTPUT_MODEL}")