from __future__ import annotations

import re


def parse_accept_reject(text: str | None) -> tuple[str, str]:
    """Paper-v1 Accept/Reject parser, preserved from the original scripts."""
    if not text:
        return "Unknown", text or ""
    clean_text = re.sub(r"(?si)<think>.*?</think>", "", text)
    match = re.search(r"\[Answer\]:?\s*(Accept|Reject)", clean_text, re.IGNORECASE)
    if match:
        return match.group(1).title(), clean_text
    if "Reject" in clean_text and "Accept" not in clean_text:
        return "Reject", clean_text
    if "Accept" in clean_text and "Reject" not in clean_text:
        return "Accept", clean_text
    return "Unknown", clean_text


def parse_cpa_choice(text: str | None) -> tuple[str | None, str | None]:
    """Paper-v1 CPA parser, including the original [Answer] fallback."""
    if not text:
        return None, None
    match = re.search(r"\\boxed\{(Image\s*[AB])\}", text, re.IGNORECASE)
    choice = match.group(1).title() if match else None
    if not choice:
        match = re.search(r"\[Answer\]:?\s*(Image\s*[AB])", text, re.IGNORECASE)
        choice = match.group(1).title() if match else None
    reason_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    return choice, reason


def cpa_status(forward: str | None, reverse: str | None) -> str:
    if forward == "Image A" and reverse == "Image B":
        return "Consistent_Correct"
    if forward == "Image B" and reverse == "Image A":
        return "Consistent_Wrong"
    if forward == "Image B" and reverse == "Image B":
        return "Bias_Position_B"
    if forward == "Image A" and reverse == "Image A":
        return "Bias_Position_A"
    return "Parse_Fail"

