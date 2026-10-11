"""LLM model constants shared across the package.

These mirror the two ``ChatGoogleGenerativeAI`` configurations found in the
notebooks: a generator model for generation tasks and a distinct evaluator
model for the feedback (critic) loop.  Keeping them in a dedicated module lets
any file import the model identifiers and hyper-parameters without importing
any heavyweight SDK.
"""

from __future__ import annotations

import os

# The API key used by both the generator and the evaluator models (Gemini).
GOOGLE_API_KEY = os.getenv("GOOGLE_AI_STUDIO_GENERATIVE_MODEL_API_KEY")

# The generator model used for history / achievements / article generation.
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

# The evaluator model used by the LLM critic loop (feedback variants).
GOOGLE_EVALUATOR_MODEL = os.getenv("GOOGLE_EVALUATOR_MODEL", "gemini-3.7-flash")

# Shared generation hyper-parameters applied to every model.
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 4096