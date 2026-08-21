# Adding Research Tools

Prefer typed Python tools over shell commands. Built-in specialist capabilities are assembled in `app/agents/toolsets.py` and granted only when the generated `AgentSpec.required_capabilities` asks for them.

For experiment-specific Python tools, set `AGENT_TOOL_EXTENSIONS` to comma-separated importable modules. Each module exposes `register(agent)` and should register typed PydanticAI tools. Treat arguments and results as observable research data: do not pass secrets, and persist large/full outputs as immutable artifacts while returning a bounded summary/reference.

Executable capability must respect the run workspace, path-traversal protections, environment filtering, tool/time/output budgets and idempotent side-effect semantics. Do not grant unrestricted arbitrary Python/system access to agent-authored code. The default shell allowlist is read/search-oriented (`rg`, `grep`, `find`, `cat`, `head`, `tail`, `wc`, `sed`, `jq`, etc.); bash is disabled unless explicitly enabled.
