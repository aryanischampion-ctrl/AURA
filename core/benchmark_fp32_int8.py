import time
import pickle
import numpy as np
import onnxruntime as ort


# =========================================================
# Configuration
# =========================================================

ITERATIONS = 1000
WARMUP = 20

FP32_MODEL = "models/aura_intent_model.onnx"
INT8_MODEL = "models/aura_intent_model_int8.onnx"


# =========================================================
# Load Vectorizer
# =========================================================

with open(
    "models/neural_intent_vectorizer.pkl",
    "rb"
) as f:
    vectorizer = pickle.load(f)


# =========================================================
# Real AURA Input
# =========================================================

text = "I am about to share my API key with an external AI tool"

features = vectorizer.transform(
    [text]
).toarray().astype(np.float32)


# =========================================================
# Load Models
# =========================================================

fp32_session = ort.InferenceSession(
    FP32_MODEL,
    providers=["CPUExecutionProvider"]
)

int8_session = ort.InferenceSession(
    INT8_MODEL,
    providers=["CPUExecutionProvider"]
)

fp32_input = fp32_session.get_inputs()[0].name
int8_input = int8_session.get_inputs()[0].name


# =========================================================
# Warm-up
# =========================================================

for _ in range(WARMUP):

    fp32_session.run(
        None,
        {fp32_input: features}
    )

    int8_session.run(
        None,
        {int8_input: features}
    )


# =========================================================
# Benchmark FP32
# =========================================================

start = time.perf_counter()

for _ in range(ITERATIONS):

    fp32_session.run(
        None,
        {fp32_input: features}
    )

end = time.perf_counter()

fp32_total = end - start

fp32_average = (
    fp32_total / ITERATIONS
) * 1000


# =========================================================
# Benchmark INT8
# =========================================================

start = time.perf_counter()

for _ in range(ITERATIONS):

    int8_session.run(
        None,
        {int8_input: features}
    )

end = time.perf_counter()

int8_total = end - start

int8_average = (
    int8_total / ITERATIONS
) * 1000


# =========================================================
# Calculate Speedup
# =========================================================

speedup = fp32_average / int8_average


fp32_ips = 1000 / fp32_average
int8_ips = 1000 / int8_average


# =========================================================
# Results
# =========================================================

print("\nAURA FP32 vs INT8 CPU BENCHMARK")
print("=" * 50)

print("Iterations:", ITERATIONS)

print("\nFP32")
print("-" * 30)
print(
    "Total time:",
    round(fp32_total, 4),
    "seconds"
)
print(
    "Average latency:",
    round(fp32_average, 4),
    "ms"
)
print(
    "Inferences/sec:",
    round(fp32_ips, 2)
)

print("\nINT8")
print("-" * 30)
print(
    "Total time:",
    round(int8_total, 4),
    "seconds"
)
print(
    "Average latency:",
    round(int8_average, 4),
    "ms"
)
print(
    "Inferences/sec:",
    round(int8_ips, 2)
)

print("\nComparison")
print("-" * 30)
print(
    "INT8 speedup:",
    round(speedup, 2),
    "x"
)