from __future__ import annotations

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import Settings
from app.models.domain import ModelCallMetadata, ModelProfile


def build_model(settings: Settings, profile: ModelProfile | str = ModelProfile.BALANCED) -> OpenAIChatModel:
    profile_value = ModelProfile(profile).value if not isinstance(profile, ModelProfile) else profile.value
    return OpenAIChatModel(
        settings.model_for_profile(profile_value),
        provider=OpenAIProvider(base_url=settings.litellm_api_base, api_key=settings.litellm_api_key),
    )


def build_named_model(settings: Settings, model_name: str) -> OpenAIChatModel:
    """Build an OpenAI-compatible model without assigning it a research profile."""
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=settings.litellm_api_base, api_key=settings.litellm_api_key),
    )


def model_metadata(settings: Settings, profile: ModelProfile | str, retry_number: int = 0) -> ModelCallMetadata:
    p = ModelProfile(profile)
    return ModelCallMetadata(
        model=settings.model_for_profile(p.value), profile=p, temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens, retry_number=retry_number,
    )
