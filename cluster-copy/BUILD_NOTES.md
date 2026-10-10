# Refactor notes

This repository was refactored from the supplied thesis scaffold rather than rebuilt blindly. Reused pieces include the FastAPI/Next.js shell, local FTS5 idea, artifact-inspection UX and PydanticAI orientation. Replaced pieces include predefined specialists, workspace-level persistent instance graph, fabricated/minimal DBLP ontology, evidence/provenance structures, EDC embedding shortlist, run persistence, telemetry, verification/finalization semantics and KG-oriented UI language.

The bundled DBLP file is the supplied official schema version dated 2024-06-14. Upstream EDC code is reference material only; its Extract/Define/Canonicalize ideas are adapted behind thesis-native typed models and evidence provenance.

## Validation status in the packaging environment

- `python -m compileall -q backend/app backend/tests`: **passed**.
- Repository secret scan for the supplied LiteLLM/Langtrace credentials: **passed**; real credentials are not bundled.
- Full pytest collection requires the declared runtime dependencies. The packaging sandbox did not have `pydantic_ai` or `pydantic_ai_harness` installed and outbound package installation was unavailable, so a live Harness/PydanticAI integration test was not claimed here.
- Run `make check` (or install `backend[dev]` and execute the documented test command) in the target environment after installing dependencies.
