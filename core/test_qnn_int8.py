import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn
import joblib

MODEL_PATH = "models/aura_intent_model_int8.onnx"
VECTORIZER_PATH = "models/neural_intent_vectorizer.pkl"

print("=" * 60)
print("AURA QNN INT8 TEST")
print("=" * 60)

# Register QNN plugin
qnn_library = qnn.get_library_path()

print("\nQNN library:")
print(qnn_library)

ort.register_execution_provider_library(
    "QNNExecutionProvider",
    qnn_library
)

print("\nAvailable providers:")
print(ort.get_available_providers())

# Create QNN session
print("\nCreating QNN inference session...")

ort.set_default_logger_severity(0)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "QNNExecutionProvider",
        "CPUExecutionProvider"
    ]
)

print("Session created successfully!")

print("\nExecution providers:")
print(session.get_providers())

# Load vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)

text = "I am about to share my API key with an external AI tool"

print("\nInput:")
print(text)

# Vectorize
X = vectorizer.transform([text]).toarray().astype(np.float32)

print("\nInput shape:")
print(X.shape)

# Run inference
input_name = session.get_inputs()[0].name

output = session.run(
    None,
    {input_name: X}
)

logits = output[0]

print("\nRaw output:")
print(logits)

prediction = int(np.argmax(logits[0]))

print("\nPredicted class index:")
print(prediction)

print("\n" + "=" * 60)
print("QNN TEST COMPLETED")
print("=" * 60)
