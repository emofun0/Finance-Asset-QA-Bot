from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError


TModel = TypeVar("TModel", bound=BaseModel)


def parse_structured_output(raw_text: str, schema: type[TModel]) -> TModel:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    try:
        return schema.model_validate_json(cleaned)
    except ValidationError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
        return schema.model_validate(parsed)
