import time
import pickle
import numpy as np
import onnxruntime as ort


# =========================================================
# Load Preprocessing
# =========================================================

with open(
    "models/neural_intent_vectorizer.pkl",
    "rb"
) as f:
    vectorizer = pickle.load(f)


# =========================================================
# Load ONNX Model
# =========================================================

session = ort.InferenceSession(
    "models/aura_intent_model.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# =========================================================
# Test Input
# =========================================================

text = "I am about to share my API key with an external AI tool"

features = vectorizer.transform(
    [text]
).toarray().astype(np.float32)


# =========================================================
# Warm-up
# =========================================================

for _ in range(20):
    session.run(
        None,
        {input_name: features}
    )


# =========================================================
# Benchmark
# =========================================================

iterations = 1000

start = time.perf_counter()

for _ in range(iterations):
    session.run(
        None,
        {input_name: features}
    )

end = time.perf_counter()


total_time = end - start

average_ms = (
    total_time / iterations
) * 1000


# =========================================================
# Results
# =========================================================

print("\nAURA ONNX CPU BENCHMARK")
print("=" * 40)

print("Iterations:", iterations)

print(
    "Total time:",
    round(total_time, 4),
    "seconds"
)

print(
    "Average inference latency:",
    round(average_ms, 4),
    "ms"
)

print(
    "Approximate inferences/second:",
    round(1000 / average_ms, 2)
)