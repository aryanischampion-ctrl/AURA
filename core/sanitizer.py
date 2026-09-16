import re


def sanitize_text(text):
    """
    Remove or mask sensitive information before sharing.
    """

    sanitized = text

    # API keys
    sanitized = re.sub(
        r"sk-[A-Za-z0-9_-]{10,}",
        "[API KEY REDACTED]",
        sanitized
    )

    # Passwords
    sanitized = re.sub(
        r"(?i)((password|passwd|pwd)\s*[:=]\s*)\S+",
        r"\1[PASSWORD REDACTED]",
        sanitized
    )

    # Email addresses
    sanitized = re.sub(
        r"[\w.-]+@[\w.-]+\.\w+",
        "[EMAIL REDACTED]",
        sanitized
    )

    # 10-digit phone numbers
    sanitized = re.sub(
        r"\b\d{10}\b",
        "[PHONE REDACTED]",
        sanitized
    )

    return sanitized