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


with open(
    "models/neural_intent_label_encoder.pkl",
    "rb"
) as f:
    label_encoder = pickle.load(f)


# =========================================================
# Load ONNX Model
# =========================================================

session = ort.InferenceSession(
    "models/aura_intent_model.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# =========================================================
# Test Cases
# =========================================================

test_cases = [
    "I am about to share my API key with an external AI tool",
    "Please send my password to this website",
    "I need to upload my customer's private information",
    "What is machine learning?",
    "How does encryption work?",
    "Fix this Python code",
    "Write a Flask API for me",
    "Here is my public project description",
    "This is information already available publicly",
    "Help me organize my private notes",
    "Create a personal study plan",
]


# =========================================================
# ONNX Prediction
# =========================================================

print("\nAURA ONNX INTENT MODEL TEST")
print("=" * 50)

for text in test_cases:

    features = vectorizer.transform(
        [text]
    ).toarray().astype(np.float32)

    outputs = session.run(
        None,
        {
            input_name: features
        }
    )

    logits = outputs[0][0]

    probabilities = np.exp(
        logits - np.max(logits)
    )

    probabilities = (
        probabilities / probabilities.sum()
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    prediction = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = (
        probabilities[predicted_index] * 100
    )

    print("\nText:", text)
    print("Intent:", prediction)
    print(
        "Confidence:",
        round(float(confidence), 2),
        "%"
    )