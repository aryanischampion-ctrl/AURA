import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from data.intent_training import TRAINING_DATA


# =========================================================
# Configuration
# =========================================================

EPOCHS = 100
LEARNING_RATE = 0.01
HIDDEN_SIZE = 64


# =========================================================
# Prepare Training Data
# =========================================================

texts = [item[0] for item in TRAINING_DATA]
labels = [item[1] for item in TRAINING_DATA]


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(texts).toarray()


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)


dataset = TensorDataset(X_tensor, y_tensor)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# =========================================================
# Neural Network
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


input_size = X.shape[1]
output_size = len(label_encoder.classes_)


model = AURAIntentModel(
    input_size,
    HIDDEN_SIZE,
    output_size
)


# =========================================================
# Training
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("Training AURA neural intent model...")
print("Training examples:", len(TRAINING_DATA))
print("Input features:", input_size)
print("Intents:", output_size)


for epoch in range(EPOCHS):

    total_loss = 0

    for batch_X, batch_y in loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {total_loss:.4f}"
        )


# =========================================================
# Save Model
# =========================================================

os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "output_size": output_size,
        "classes": label_encoder.classes_.tolist()
    },
    "models/neural_intent_model.pth"
)


# Save vectorizer and label encoder

import pickle

with open(
    "models/neural_intent_vectorizer.pkl",
    "wb"
) as f:
    pickle.dump(vectorizer, f)


with open(
    "models/neural_intent_label_encoder.pkl",
    "wb"
) as f:
    pickle.dump(label_encoder, f)


print("\nAURA neural intent model trained successfully.")
print("Model saved to:")
print("models/neural_intent_model.pth")