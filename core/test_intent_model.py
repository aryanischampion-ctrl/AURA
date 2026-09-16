import pickle

with open("models/intent_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/intent_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)


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


for text in test_cases:

    X = vectorizer.transform([text])

    prediction = classifier.predict(X)[0]
    probabilities = classifier.predict_proba(X)[0]

    confidence = max(probabilities) * 100

    print("\nText:", text)
    print("Intent:", prediction)
    print("Confidence:", round(confidence, 2), "%")