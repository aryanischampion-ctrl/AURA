import os
import pickle
import torch
import torch.nn as nn


# =========================================================
# Model Definition
# =========================================================

class AURAIntentModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# Load Files
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


checkpoint = torch.load(
    "models/neural_intent_model.pth",
    map_location="cpu"
)


model = AURAIntentModel(
    checkpoint["input_size"],
    checkpoint["hidden_size"],
    checkpoint["output_size"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


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
# Prediction
# =========================================================

print("\nAURA NEURAL INTENT MODEL TEST")
print("=" * 50)


for text in test_cases:

    X = vectorizer.transform([text]).toarray()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = model(X_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_index = torch.argmax(
            probabilities
        ).item()

    prediction = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = (
        probabilities[predicted_index].item() * 100
    )

    print("\nText:", text)
    print("Intent:", prediction)
    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )