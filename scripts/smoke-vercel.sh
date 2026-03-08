#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${1:-${VERCEL_URL:-}}"

if [[ -z "${BASE_URL}" ]]; then
  echo "Usage: $0 <base-url>"
  echo "Example: $0 https://your-app.vercel.app"
  echo "You can also set VERCEL_URL and omit the argument."
  exit 1
fi

BASE_URL="${BASE_URL%/}"

echo "Running smoke tests against: ${BASE_URL}"

echo "1) Checking /api/health"
health_response="$(curl -sS -f "${BASE_URL}/api/health")"
echo "   OK"

if ! grep -q '"status"' <<<"${health_response}"; then
  echo "Health response missing status field"
  exit 1
fi

echo "2) Checking /api/sentiment/model-info"
model_response="$(curl -sS -f "${BASE_URL}/api/sentiment/model-info")"
echo "   OK"

if ! grep -q '"modelName"' <<<"${model_response}"; then
  echo "Model info response missing modelName"
  exit 1
fi

echo "3) Checking /api/sentiment/analyze"
analyze_response="$(curl -sS -f -X POST "${BASE_URL}/api/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text":"Battery life is excellent and performance is smooth."}')"
echo "   OK"

if ! grep -q '"sentiment"' <<<"${analyze_response}"; then
  echo "Analyze response missing sentiment"
  exit 1
fi

echo "Smoke test passed"
echo "Summary:"
echo "- health: ${health_response}"
echo "- model-info: ${model_response}"
echo "- analyze: ${analyze_response}"
