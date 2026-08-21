from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    The names intentionally match the thesis deployment `.env`. Provider secrets are
    consumed by LiteLLM; this application only needs the OpenAI-compatible gateway.
    """

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore", case_sensitive=False)

    # Core
    app_env: str = "development"
    frontend_origin: str = "http://localhost:3000"
    data_dir: Path = Path("./data")
    seed_ontology_path: Path = Path("./app/seed/dblp_schema_2024-06-14.ttl")

    # LiteLLM gateway
    litellm_api_base: str = "http://127.0.0.1:4000/v1"
    litellm_api_key: str = ""
    litellm_model: str = "qwen-chat"
    litellm_fast_model: str = "research-fast"
    # Optional explicit profiles. If absent they deliberately collapse onto the one
    # gateway model the thesis deployment exposes.
    model_balanced: str | None = None
    model_reasoning: str | None = None
    model_critic: str | None = None
    model_extractor: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_output_retries: int = 3

    # Agentic harness budgets
    max_agent_steps: int = 28
    max_subagent_calls: int = 18
    max_run_seconds: int = 900
    max_evidence_spans: int = 24
    max_evidence_chars: int = 45_000
    max_tool_output_chars: int = 20_000
    context_warn_fraction: float = 0.82
    context_clear_fraction: float = 0.70

    # Harness capabilities
    enable_agent_shell: bool = True
    enable_agent_bash: bool = False
    enable_dynamic_workflow: bool = True
    enable_code_mode: bool = False
    enable_step_persistence: bool = True
    agent_shell_timeout_seconds: float = 20.0
    agent_tool_extensions: str = ""
    shell_allowed_commands: list[str] = Field(default_factory=lambda: [
        "pwd", "ls", "cat", "rg", "grep", "find", "sed", "head", "tail", "wc", "sort", "uniq", "cut", "jq"
    ])

    # Retrieval: no embeddings/vector DB
    lexical_top_k: int = 14
    context_expansion_paragraphs: int = 2
    retrieval_allow_shell_search: bool = True
    retrieval_allow_graph_search: bool = True
    pdf_max_pages: int = 300

    # Parser adapters; optional services, never mandatory
    grobid_url: str | None = None
    docling_url: str | None = None

    # Ontology / validation
    allow_ontology_extensions: bool = True
    shacl_strict: bool = False
    edc_upstream_path: str | None = None
    enable_embedding_canonicalization_ablation: bool = False

    # Optional durable execution
    durability_backend: Literal["none", "dbos", "prefect", "temporal"] = "none"
    database_url: str = "sqlite:///./data/research.db"

    # Observability
    enable_logfire: bool = False
    logfire_token: str | None = None
    enable_langtrace: bool = True
    # Phase 1 exports application spans only. Phase 2 can opt into SDK vendor
    # auto-instrumentation after duplicate-span behavior is evaluated.
    enable_langtrace_auto_instrumentation: bool = False
    langtrace_api_key: str | None = None
    langtrace_api_host: str = "https://langtrace.ai/"
    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_headers: str | None = None
    otel_service_name: str = "thesis-research-harness"
    enable_llm_eval: bool = False

    # Frontend/backend
    next_public_api_base: str = "http://localhost:8000"
    backend_internal_url: str = "http://127.0.0.1:8000"

    @computed_field
    @property
    def frontend_origins(self) -> list[str]:
        return [x.strip() for x in self.frontend_origin.split(",") if x.strip()]

    def model_for_profile(self, profile: str) -> str:
        mapping = {
            "fast": self.litellm_fast_model,
            "balanced": self.model_balanced or self.litellm_model,
            "reasoning": self.model_reasoning or self.litellm_model,
            "critic": self.model_critic or self.model_reasoning or self.litellm_model,
            "extractor": self.model_extractor or self.model_balanced or self.litellm_model,
        }
        return mapping.get(profile, self.litellm_model)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for name in ("documents", "runs", "workspaces", "harness_steps", "ontology_cache"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
