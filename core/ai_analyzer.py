import os
import pickle
import numpy as np
import onnxruntime as ort


# =========================================================
# AURA INT8 ONNX INTENT MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_FILE = os.path.join(
    BASE_DIR,
    "models",
    "neural_intent_vectorizer.pkl"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "aura_intent_model_int8.onnx"
)


# Load vectorizer
with open(VECTOR_FILE, "rb") as f:
    vectorizer = pickle.load(f)


# Load INT8 ONNX model
session = ort.InferenceSession(
    MODEL_FILE,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# Intent labels
INTENT_LABELS = [
    "asking a question",
    "private personal use",
    "sharing normal information",
    "sharing sensitive information",
    "writing or editing code"
]


# =========================================================
# Analyze User Context
# =========================================================

def analyze_context(text):

    # Convert text to the same 482-feature representation
    X = vectorizer.transform([text]).toarray().astype(np.float32)

    # Run INT8 ONNX model
    outputs = session.run(
        None,
        {input_name: X}
    )

    logits = outputs[0][0]

    # Softmax
    exp_values = np.exp(logits - np.max(logits))
    probabilities = exp_values / np.sum(exp_values)

    prediction_index = int(np.argmax(probabilities))

    intent = INTENT_LABELS[prediction_index]

    confidence = float(probabilities[prediction_index]) * 100
    if intent == "sharing sensitive information" and confidence < 80:
        intent = "sharing normal information"
    return {
        "intent": intent,
        "confidence": round(confidence, 2)
    }