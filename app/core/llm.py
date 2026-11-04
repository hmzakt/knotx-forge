import json
import os
import re
from typing import TypeVar, Type

from litellm import completion
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def _extract_json(content: str) -> str:
    """Accept plain JSON or JSON wrapped in Markdown fences."""
    content = content.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else content


class LLMGateway:
    """
    Small provider-neutral gateway.
    Configure model names using LiteLLM's provider/model syntax.
    Example: gemini/gemini-2.5-flash or groq/openai/gpt-oss-120b.
    """

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.model = model or os.getenv("GENERATOR_MODEL", "")
        self.temperature = (
            temperature if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0.2"))
        )
        if not self.model:
            raise ValueError("Set GENERATOR_MODEL in the environment.")

    def structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        model: str | None = None,
        temperature: float | None = None,
    ) -> T:
        """Request JSON and validate it against a Pydantic v2 model."""
        selected_model = model or self.model
        response = completion(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature if temperature is None else temperature,
        )
        content = response.choices[0].message.content # type: ignore
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{selected_model} returned an empty response.")

        try:
            return schema.model_validate_json(_extract_json(content))
        except (ValidationError, ValueError) as exc:
            raise ValueError(
                f"Could not validate structured response from {selected_model}: {exc}"
            ) from exc
