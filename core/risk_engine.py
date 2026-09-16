def calculate_risk(findings, destination, ai_result=None):
    """
    Calculate risk using:
    1. Sensitive information
    2. Destination
    3. AI-understood intent
    """

    score = 0

    # Sensitive information
    for finding in findings:
        if finding["severity"] == "high":
            score += 70
        elif finding["severity"] == "medium":
            score += 20

    # Destination risk
    if destination == "External AI Tool":
        if any(f["severity"] == "high" for f in findings):
            score += 30

    elif destination == "Public Website":
        if any(f["severity"] == "high" for f in findings):
            score += 30
        elif any(f["severity"] == "medium" for f in findings):
            score += 15

    elif destination == "Email":
        if any(f["severity"] == "high" for f in findings):
            score += 20

    elif destination == "Cloud Storage":
        if any(f["severity"] == "high" for f in findings):
            score += 10

    # AI intent
    if ai_result:
        intent = ai_result["intent"]

        if intent == "sharing sensitive information":
            score += 20

        elif intent == "private personal use":
            score -= 10

    # Keep score between 0 and 100
    score = max(0, min(score, 100))

    if score >= 80:
        level = "CRITICAL"
    elif score >= 60:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "score": score
    }