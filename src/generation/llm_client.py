"""
Google Gemini API wrapper using the new google-genai SDK.

Uses gemini-2.5-flash for text generation and vision (image descriptions).
Free tier: 15 RPM, 1,500 req/day, 1M TPM.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash"


class LLMClient:
    """
    Thin wrapper around the google-genai Python SDK.

    Exposes:
    - generate(): text-only generation with a system prompt
    - describe_image(): vision-based image description
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model_name = model
        logger.info("LLMClient initialised with model=%s.", model)

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a text response from Gemini.

        Args:
            system_prompt: Instructions / persona (system_instruction).
            user_message: The user's question or task.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Upper bound on response length.

        Returns:
            The model's response text.
        """
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text

    def describe_image(
        self,
        image_bytes: bytes,
        system_prompt: str,
        media_type: str = "image/png",
    ) -> str:
        """
        Generate a textual description of an image using Gemini Vision.

        Args:
            image_bytes: Raw image bytes (PNG or JPEG).
            system_prompt: Instructions for describing the image.
            media_type: Ignored (PIL handles format detection).

        Returns:
            Textual description produced by Gemini.
        """
        import PIL.Image
        from google.genai import types

        image = PIL.Image.open(io.BytesIO(image_bytes))

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[
                "Please describe this figure from the research paper.",
                image,
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
        return response.text
