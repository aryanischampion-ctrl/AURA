import os
import numpy as np
import onnxruntime as ort


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FP32_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "aura_intent_model.onnx"
)

INT8_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "aura_intent_model_int8.onnx"
)


print("AURA FP32 vs INT8 COMPARISON")
print("=" * 45)


# Load both models
fp32_session = ort.InferenceSession(
    FP32_MODEL,
    providers=["CPUExecutionProvider"]
)

int8_session = ort.InferenceSession(
    INT8_MODEL,
    providers=["CPUExecutionProvider"]
)


# Get input name
fp32_input = fp32_session.get_inputs()[0].name
int8_input = int8_session.get_inputs()[0].name


# ---------------------------------------------------------
# Test inputs
# ---------------------------------------------------------
# 482 features are required by the AURA model.
# We use deterministic random inputs for the first test.
np.random.seed(42)

test_input = np.random.randn(
    1, 482
).astype(np.float32)


# Run FP32
fp32_output = fp32_session.run(
    None,
    {fp32_input: test_input}
)[0]


# Run INT8
int8_output = int8_session.run(
    None,
    {int8_input: test_input}
)[0]


print("\nFP32 output:")
print(fp32_output)

print("\nINT8 output:")
print(int8_output)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

absolute_difference = np.abs(
    fp32_output - int8_output
)

max_difference = np.max(
    absolute_difference
)

mean_difference = np.mean(
    absolute_difference
)


fp32_prediction = np.argmax(fp32_output, axis=1)
int8_prediction = np.argmax(int8_output, axis=1)


print("\nComparison")
print("-" * 45)

print(
    "Maximum absolute difference:",
    max_difference
)

print(
    "Mean absolute difference:",
    mean_difference
)

print(
    "FP32 predicted class:",
    fp32_prediction
)

print(
    "INT8 predicted class:",
    int8_prediction
)


if np.array_equal(
    fp32_prediction,
    int8_prediction
):
    print("\nPrediction MATCH: PASS")
else:
    print("\nPrediction MATCH: WARNING")