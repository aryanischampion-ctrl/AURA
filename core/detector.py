import re


def detect_sensitive_data(text):
    """
    Detect potentially sensitive information in text.

    Returns a list of findings.
    """

    findings = []

    # Possible OpenAI-style API key
    if re.search(r"sk-[A-Za-z0-9_-]{10,}", text):

        findings.append({
            "type": "API Key",
            "severity": "high",
            "message": "Possible API key detected."
        })

    # Email address
    if re.search(
        r"[\w.-]+@[\w.-]+\.\w+",
        text
    ):

        findings.append({
            "type": "Email",
            "severity": "medium",
            "message": "Email address detected."
        })

    # Ten-digit phone number
    if re.search(
        r"\b\d{10}\b",
        text
    ):

        findings.append({
            "type": "Phone Number",
            "severity": "medium",
            "message": "Possible phone number detected."
        })

    # Password-like assignment
    if re.search(
        r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+",
        text
    ):

        findings.append({
            "type": "Password",
            "severity": "high",
            "message": "Possible password detected."
        })

    return findings