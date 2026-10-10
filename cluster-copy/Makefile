.PHONY: backend frontend test compile check
backend:
	cd backend && uvicorn app.api.main:app --reload --port 8000
frontend:
	cd frontend && npm run dev
test:
	cd backend && PYTHONPATH=. pytest -q
compile:
	python -m compileall -q backend/app backend/tests
check:
	./scripts/check.sh
