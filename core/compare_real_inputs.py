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
# Load FP32 Model
# =========================================================

fp32_session = ort.InferenceSession(
    "models/aura_intent_model.onnx",
    providers=["CPUExecutionProvider"]
)

fp32_input_name = fp32_session.get_inputs()[0].name


# =========================================================
# Load INT8 Model
# =========================================================

int8_session = ort.InferenceSession(
    "models/aura_intent_model_int8.onnx",
    providers=["CPUExecutionProvider"]
)

int8_input_name = int8_session.get_inputs()[0].name


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
# Comparison
# =========================================================

print("\nAURA FP32 vs INT8 REAL INPUT TEST")
print("=" * 65)


matches = 0
total = len(test_cases)


for text in test_cases:

    # -----------------------------------------------------
    # Convert text → 482 TF-IDF features
    # -----------------------------------------------------

    features = vectorizer.transform(
        [text]
    ).toarray().astype(np.float32)


    # -----------------------------------------------------
    # FP32 inference
    # -----------------------------------------------------

    fp32_logits = fp32_session.run(
        None,
        {
            fp32_input_name: features
        }
    )[0][0]


    fp32_probabilities = np.exp(
        fp32_logits - np.max(fp32_logits)
    )

    fp32_probabilities /= fp32_probabilities.sum()

    fp32_index = int(
        np.argmax(fp32_probabilities)
    )

    fp32_prediction = label_encoder.inverse_transform(
        [fp32_index]
    )[0]

    fp32_confidence = (
        fp32_probabilities[fp32_index] * 100
    )


    # -----------------------------------------------------
    # INT8 inference
    # -----------------------------------------------------

    int8_logits = int8_session.run(
        None,
        {
            int8_input_name: features
        }
    )[0][0]


    int8_probabilities = np.exp(
        int8_logits - np.max(int8_logits)
    )

    int8_probabilities /= int8_probabilities.sum()

    int8_index = int(
        np.argmax(int8_probabilities)
    )

    int8_prediction = label_encoder.inverse_transform(
        [int8_index]
    )[0]

    int8_confidence = (
        int8_probabilities[int8_index] * 100
    )


    # -----------------------------------------------------
    # Compare
    # -----------------------------------------------------

    difference = np.max(
        np.abs(fp32_logits - int8_logits)
    )

    match = fp32_prediction == int8_prediction

    if match:
        matches += 1


    print("\nText:", text)

    print(
        "FP32 :",
        fp32_prediction,
        "| Confidence:",
        round(float(fp32_confidence), 2),
        "%"
    )

    print(
        "INT8 :",
        int8_prediction,
        "| Confidence:",
        round(float(int8_confidence), 2),
        "%"
    )

    print(
        "Max logit difference:",
        round(float(difference), 6)
    )

    print(
        "Prediction:",
        "MATCH" if match else "MISMATCH"
    )


# =========================================================
# Final Result
# =========================================================

accuracy = (
    matches / total
) * 100


print("\n" + "=" * 65)

print(
    f"Prediction matches: {matches}/{total}"
)

print(
    f"Consistency: {accuracy:.2f}%"
)


if matches == total:
    print(
        "\nRESULT: INT8 VALIDATION PASS"
    )
else:
    print(
        "\nRESULT: INT8 VALIDATION WARNING"
    )