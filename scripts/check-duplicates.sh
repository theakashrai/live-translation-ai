#!/bin/bash
# Code quality check script

set -e

echo "🔍 Running code duplication checks..."

# Run jscpd for duplication detection
echo "📋 Checking for duplicate code..."
jscpd src/ --config .jscpd.json

# Run pylint for additional duplication checks
echo "🔍 Running pylint duplicate-code check..."
uv run pylint --enable=duplicate-code --disable=all src/live_translation/ || true

echo "✅ Code quality checks complete!"
