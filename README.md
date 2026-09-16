# AURA — Adaptive User Risk Awareness

AURA is an AI-powered privacy assistant that detects sensitive information, understands the user's sharing intent, assesses exposure risk, and recommends safer ways to share content with external AI tools.

## Problem

Users increasingly paste personal, confidential, and security-sensitive information into AI tools without realizing the potential exposure risk.

AURA provides a privacy layer between the user and external AI tools by analyzing content **before it is shared**.

## Solution

AURA combines rule-based sensitive-data detection with an INT8 ONNX intent-classification model to understand both **what the content contains** and **why the user is sharing it**.

The system then calculates a risk score based on:

* Detected sensitive information
* Destination
* User sharing intent

When sensitive information is detected, AURA can generate a redacted safe version of the content.

## How AURA Works

```text
User Input
    ↓
Sensitive Data Detection
    ↓
INT8 ONNX Intent Analysis
    ↓
Confidence Guard
    ↓
Risk Assessment
    ↓
Security Recommendation
    ↓
Safe Redaction
```

## Key Features

* API key detection
* Password detection
* Email detection
* Phone-number detection
* AI-based sharing-intent classification
* Destination-aware risk assessment
* Risk score from 0–100
* CRITICAL / HIGH / MEDIUM / LOW risk levels
* Safe content redaction
* Local-first processing
* INT8 ONNX inference
* Snapdragon X Elite NPU validation through Qualcomm AI Hub

## AI Intent Classification

AURA's intent model classifies user input into five categories:

1. Asking a question
2. Private personal use
3. Sharing normal information
4. Sharing sensitive information
5. Writing or editing code

The deployed model uses an INT8 ONNX representation with **482 input features** and **5 output classes**.

## Model Optimization

The INT8 ONNX model was optimized for efficient edge deployment.

| Model     |      Size |
| --------- | --------: |
| FP32 ONNX | 122.62 KB |
| INT8 ONNX |  33.38 KB |
| Reduction |    ~72.8% |

Validation testing showed matching predictions between the FP32 and INT8 models across **11/11 tested cases**.

This demonstrates prediction agreement on the tested validation inputs; it is not a claim of 100% overall model accuracy.

## Snapdragon NPU Validation

The INT8 ONNX model was successfully compiled and profiled using Qualcomm AI Hub for a **Snapdragon X Elite CRD** target.

The Qualcomm AI Hub profile reported:

* Target: Snapdragon X Elite CRD
* Compute unit: Qualcomm NPU
* Model: AURA INT8 ONNX
* Input: `1 × 482`
* Output: 5 intent classes
* Estimated inference time: **31 µs**
* Compile job: `jpyo62m05`
* Profile job: `jp0ml9j0g`

The profile execution details show the model's major operations executing on the **NPU**, including Gemm and ReLU operations.

> Note: AURA was developed and tested locally on an AMD64 Windows development machine. The Snapdragon NPU result above comes from Qualcomm AI Hub profiling on the Snapdragon X Elite target and should not be interpreted as local NPU execution on the development PC.

## Example

Input:

```text
My API key is sk-abcdefghijklmnopqrstuvwxyz
```

AURA can identify:

```text
Risk Level: CRITICAL
Risk Score: 100/100
Detected: API Key (HIGH)
```

It can then generate:

```text
My API key is [API KEY REDACTED]
```

## Destinations

AURA can evaluate content against different destinations:

* External AI Tool
* Public Website
* Email
* Cloud Storage
* Private / Local

The destination affects the calculated exposure risk.

## Technology Stack

* Python
* PySide6
* NumPy
* Scikit-learn
* ONNX Runtime
* ONNX
* INT8 Quantization
* Qualcomm AI Hub
* Snapdragon X Elite NPU

## Project Structure

```text
AURA/
├── app.py
├── core/
│   ├── ai_analyzer.py
│   ├── detector.py
│   ├── recommender.py
│   ├── risk_engine.py
│   ├── sanitizer.py
│   ├── benchmark_fp32_int8.py
│   ├── benchmark_onnx.py
│   ├── compare_fp32_int8.py
│   ├── compare_models.py
│   ├── compare_real_inputs.py
│   ├── export_intent_onnx.py
│   ├── inspect_onnx.py
│   ├── inspect_qnn.py
│   ├── quantize_intent_model.py
│   ├── test_intent_model.py
│   ├── test_neural_intent_model.py
│   ├── test_onnx_model.py
│   ├── test_qnn_int8.py
│   ├── train_intent_model.py
│   └── train_neural_intent_model.py
├── data/
│   └── intent_training.py
├── models/
│   ├── aura_intent_model.onnx
│   ├── aura_intent_model_int8.onnx
│   ├── neural_intent_label_encoder.pkl
│   ├── neural_intent_model.pth
│   └── neural_intent_vectorizer.pkl
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/aryanischampion-ctrl/AURA.git
cd AURA
```

Create a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run AURA

```powershell
python app.py
```

The AURA desktop application will open locally.

## Privacy

AURA is designed around a local-first privacy model. User text is analyzed locally by the application during normal development execution rather than being sent to an external AI service for intent analysis.

## Validation

AURA has been validated through:

* Local application testing
* Sensitive-data detection tests
* FP32 vs INT8 comparison
* ONNX model testing
* Real-input intent testing
* Qualcomm AI Hub compilation
* Qualcomm AI Hub Snapdragon X Elite profiling

## Project Goal

AURA aims to make privacy-aware AI usage practical by helping users understand the risk of sharing information **before** sending it to an external AI system.

## License

This project is provided for educational and demonstration purposes.
