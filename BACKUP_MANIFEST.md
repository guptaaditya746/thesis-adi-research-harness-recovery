# Backup manifest

This repo preserves the existing 26-commit project history and includes the current local source changes and curated demo files. A high-confidence scan of tracked history found no known token or private-key patterns. Actual `.env` files remain excluded.

Excluded local or generated state:

- `.env` and `frontend/.env.local`, which contain local service credentials and database URLs.
- `frontend/node_modules/`, `.next/`, Python virtual environments and caches.
- `backend/data/*.db*`: local SQLite database plus journal/shared-memory files.
- logs, run output, `.orig` backups, and build artifacts.

The original source and its local environment files were left unchanged. Back up any database or run artifacts needed separately.
