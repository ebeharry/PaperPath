from __future__ import annotations
import json
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# Shared truncation limit for paper abstracts in LLM prompts.
ABSTRACT_LIMIT = 300


def parse_json_from_response(response: str) -> dict:
    """Extract a JSON object from an LLM response, tolerating code fences."""
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse LLM JSON response; using raw text fallback")
    return {}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalise a 2-D array. Zero-norm rows are left as zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def l2_normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2 normalise a 1-D vector. Zero-norm vectors are left as zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec
