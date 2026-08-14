#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$HOME/.venvs/crypto-factor-lab/bin/activate" ]]; then
  # Reuse the shared environment created for this project.
  source "$HOME/.venvs/crypto-factor-lab/bin/activate"
fi

python --version

python scripts/collect_real_data.py --mode rest --execute
python -m pytest -q
ruff check .
mypy config data factors evaluation utils scripts visualization
python scripts/build_data_health.py
python scripts/run_all_research.py

printf '\nReal-data collection and acceptance checks completed.\n'
printf 'Manifest: %s\n' "$PROJECT_DIR/reports/real_data_collection_manifest.json"
printf 'Catalog:  %s\n' "$PROJECT_DIR/reports/data_catalog.json"
