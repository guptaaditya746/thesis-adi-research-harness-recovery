.PHONY: backend frontend test compile check benchmark-five
backend:
	@printf "Backend: http://127.0.0.1:8100\nHealth:  http://127.0.0.1:8100/health\nFrontend: http://141.54.132.237:3000\n\n"
	cd backend && uvicorn app.api.main:app --reload --port 8100 --no-access-log
frontend:
	cd frontend && npm run dev
test:
	cd backend && PYTHONPATH=. pytest -q
compile:
	python -m compileall -q backend/app backend/tests
check:
	./scripts/check.sh
benchmark-five:
	cd backend && PYTHONPATH=. uv run python scripts/run_five_cases.py $(ARGS)
