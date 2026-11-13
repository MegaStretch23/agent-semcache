#!/bin/sh
# Regenerates the saved demo run. The fake model is deterministic, so the output is byte-identical every time.
set -e
cd "$(dirname "$0")/.."
${PY:-python3} examples/demo.py > examples/demo_output.txt
