import os
import pickle
import torch
import torch.nn as nn


# =========================================================
# AURA Neural Intent Model
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
# Load Vectorizer
# =========================================================

with open(
    "models/neural_intent_vectorizer.pkl",
    "rb"
) as f:
    vectorizer = pickle.load(f)


# =========================================================
# Load PyTorch Model
# =========================================================

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
# Create Example Input
# =========================================================

example_text = "I want to share my API key"

example_features = vectorizer.transform(
    [example_text]
).toarray()

example_input = torch.tensor(
    example_features,
    dtype=torch.float32
)


# =========================================================
# Export to ONNX
# =========================================================

os.makedirs("models", exist_ok=True)

onnx_path = "models/aura_intent_model.onnx"


torch.onnx.export(
    model,
    example_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamo=False
)


print("\nAURA neural model exported successfully.")
print("ONNX model:")
print(onnx_path)