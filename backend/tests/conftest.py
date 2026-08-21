from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    seed = Path(__file__).parents[1] / "app" / "seed" / "dblp_schema_2024-06-14.ttl"
    value = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'research.db'}",
        seed_ontology_path=seed,
        enable_langtrace=False,
        enable_agent_shell=False,
        enable_code_mode=False,
        enable_step_persistence=False,
    )
    value.ensure_dirs()
    return value
