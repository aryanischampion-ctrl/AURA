
import os
import pickle
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Load preprocessing
with open(
    os.path.join(BASE_DIR, "models", "neural_intent_vectorizer.pkl"),
    "rb"
) as f:
    vectorizer = pickle.load(f)

with open(
    os.path.join(BASE_DIR, "models", "neural_intent_label_encoder.pkl"),
    "rb"
) as f:
    label_encoder = pickle.load(f)

# Load both models
fp32_session = ort.InferenceSession(
    os.path.join(BASE_DIR, "models", "aura_intent_model.onnx"),
    providers=["CPUExecutionProvider"]
)

int8_session = ort.InferenceSession(
    os.path.join(BASE_DIR, "models", "aura_intent_model_int8.onnx"),
    providers=["CPUExecutionProvider"]
)

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

def predict(session, features):
    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: features}
    )

    logits = outputs[0][0]

    probabilities = np.exp(
        logits - np.max(logits)
    )

    probabilities /= probabilities.sum()

    index = int(np.argmax(probabilities))

    intent = label_encoder.inverse_transform(
        [index]
    )[0]

    confidence = probabilities[index] * 100

    return intent, confidence


print("\nAURA FP32 vs INT8 COMPARISON")
print("=" * 60)

matches = 0

for text in test_cases:

    features = vectorizer.transform(
        [text]
    ).toarray().astype(np.float32)

    fp32_intent, fp32_conf = predict(
        fp32_session, features
    )

    int8_intent, int8_conf = predict(
        int8_session, features
    )

    match = fp32_intent == int8_intent

    if match:
        matches += 1

    print("\nText:", text)
    print("FP32:", fp32_intent, round(float(fp32_conf), 2), "%")
    print("INT8:", int8_intent, round(float(int8_conf), 2), "%")
    print("Match:", match)

print("\n" + "=" * 60)
print(f"Matching predictions: {matches}/{len(test_cases)}")
print("Comparison completed.")