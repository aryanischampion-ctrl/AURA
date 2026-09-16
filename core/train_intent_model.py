import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.intent_training import TRAINING_DATA


# Separate text and labels
texts = [item[0] for item in TRAINING_DATA]
labels = [item[1] for item in TRAINING_DATA]


# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(texts)


# Train classifier
classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(X, labels)


# Create models directory
os.makedirs("models", exist_ok=True)


# Save vectorizer
with open("models/intent_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# Save classifier
with open("models/intent_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)


print("AURA intent model trained successfully.")
print("Training examples:", len(TRAINING_DATA))
print("Intents:", len(classifier.classes_))
print("Classes:")

for intent in classifier.classes_:
    print(" -", intent)