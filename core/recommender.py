def generate_recommendation(findings, risk_level, destination):
    """
    Generate a context-aware safety recommendation.
    """

    if risk_level == "CRITICAL":
        return (
            f"DO NOT SHARE this content with {destination}. "
            "Highly sensitive information was detected and "
            "the destination increases the exposure risk. "
            "Remove or redact the sensitive information first."
        )

    if risk_level == "HIGH":
        return (
            f"Sharing this content with {destination} may expose "
            "sensitive information. Remove secrets or personal "
            "information before continuing."
        )

    if risk_level == "MEDIUM":
        return (
            f"Review this content before sharing it with "
            f"{destination}. Consider removing unnecessary "
            "personal information."
        )

    return (
        f"Low risk detected for {destination}. "
        "No major sensitive-data indicators were found. "
        "Review the content before sharing."
    )