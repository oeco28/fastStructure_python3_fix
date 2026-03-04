#!/usr/bin/env bash
# Wrapper for structure.py — sets PYTHONPATH so bare imports like
# "import allelefreq" resolve to vars/allelefreq.so
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="${SCRIPT_DIR}/vars:${SCRIPT_DIR}:${PYTHONPATH}" \
    python3 "${SCRIPT_DIR}/structure.py" "$@"
