#!/usr/bin/env bash
# Run tests against multiple Kestra instances to validate cross-version compatibility.
#
# Usage: ./tests/run_all_versions.sh [pytest args]
# Example: ./tests/run_all_versions.sh -v -x  (verbose, stop on first failure)
#
# Reads credentials from .env in the project root (auto-loaded) or from your shell env.
#
# Required environment variables:
#   EE instances:   KESTRA_API_TOKEN_EE_DEVELOP, KESTRA_API_TOKEN_EE_LATEST
#   OSS instances:  KESTRA_USERNAME_OSS_DEVELOP, KESTRA_PASSWORD_OSS_DEVELOP,
#                   KESTRA_USERNAME_OSS_LATEST,  KESTRA_PASSWORD_OSS_LATEST
#
# Optional overrides (with defaults):
#   KESTRA_BASE_URL_EE_DEVELOP   (default: http://localhost:28080/api/v1)
#   KESTRA_BASE_URL_EE_LATEST    (default: http://localhost:18080/api/v1)
#   KESTRA_BASE_URL_OSS_DEVELOP  (default: http://localhost:48080/api/v1)
#   KESTRA_BASE_URL_OSS_LATEST   (default: http://localhost:38080/api/v1)
set -euo pipefail

PYTEST_ARGS="${@:---tb=short}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/.test-results"
mkdir -p "$RESULTS_DIR"

# ── Load .env if present ─────────────────────────────────────────────────────
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
SUMMARY=""

# ── Helper: uppercase a string (macOS ships Bash 3, which lacks ${VAR^^}) ──
to_upper() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

# ── Helper: resolve env var by name ──────────────────────────────────────────
env_val() { eval echo "\${$1:-}"; }

# ── Helper: run tests for one instance ───────────────────────────────────────
run_instance() {
  local NAME="$1"
  local TENANT="$2"
  local BASE_URL="$3"
  local AUTH_TYPE="$4"      # "token" or "basic"
  local DISABLED_TOOLS="$5" # "" or "ee"

  # Build env var names: ee-develop -> EE_DEVELOP, then KESTRA_API_TOKEN_EE_DEVELOP
  local SUFFIX
  SUFFIX="$(to_upper "$NAME" | tr '-' '_')"
  local TOKEN_VAR="KESTRA_API_TOKEN_${SUFFIX}"
  local USER_VAR="KESTRA_USERNAME_${SUFFIX}"
  local PASS_VAR="KESTRA_PASSWORD_${SUFFIX}"

  # Check required auth
  if [[ "$AUTH_TYPE" == "token" && -z "$(env_val "$TOKEN_VAR")" ]]; then
    echo "⏭  $NAME — $TOKEN_VAR not set, skipping"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    SUMMARY+="⏭  $NAME — skipped ($TOKEN_VAR not set)\n"
    return
  fi
  if [[ "$AUTH_TYPE" == "basic" ]]; then
    if [[ -z "$(env_val "$USER_VAR")" || -z "$(env_val "$PASS_VAR")" ]]; then
      echo "⏭  $NAME — $USER_VAR/$PASS_VAR not set, skipping"
      SKIP_COUNT=$((SKIP_COUNT + 1))
      SUMMARY+="⏭  $NAME — skipped (credentials not set)\n"
      return
    fi
  fi

  # Check if the instance is reachable
  if ! curl -s --connect-timeout 3 "$BASE_URL" > /dev/null 2>&1; then
    echo "⏭  $NAME ($BASE_URL) — not reachable, skipping"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    SUMMARY+="⏭  $NAME — skipped (not reachable)\n"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Running tests against: $NAME"
  echo "  URL: $BASE_URL  Tenant: $TENANT"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Set env vars for this run
  export KESTRA_TENANT_ID="$TENANT"
  export KESTRA_BASE_URL="$BASE_URL"
  export KESTRA_MCP_DISABLED_TOOLS="${DISABLED_TOOLS:-}"

  # Auth
  unset KESTRA_API_TOKEN KESTRA_USERNAME KESTRA_PASSWORD 2>/dev/null || true
  if [[ "$AUTH_TYPE" == "token" ]]; then
    export KESTRA_API_TOKEN="$(env_val "$TOKEN_VAR")"
  elif [[ "$AUTH_TYPE" == "basic" ]]; then
    export KESTRA_USERNAME="$(env_val "$USER_VAR")"
    export KESTRA_PASSWORD="$(env_val "$PASS_VAR")"
  fi

  REPORT="$RESULTS_DIR/$NAME.txt"
  if uv run pytest tests/ $PYTEST_ARGS 2>&1 | tee "$REPORT"; then
    PASS_COUNT=$((PASS_COUNT + 1))
    SUMMARY+="✅ $NAME — passed\n"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    SUMMARY+="❌ $NAME — FAILED (see $REPORT)\n"
  fi
}

# ── Define instances ─────────────────────────────────────────────────────────
# NAME              TENANT  BASE_URL                                    AUTH    DISABLED
run_instance "ee-develop"  "demo" "${KESTRA_BASE_URL_EE_DEVELOP:-http://localhost:28080/api/v1}"  "token" ""
run_instance "oss-develop" "main" "${KESTRA_BASE_URL_OSS_DEVELOP:-http://localhost:48080/api/v1}" "basic" "ee"
run_instance "ee-latest"   "demo" "${KESTRA_BASE_URL_EE_LATEST:-http://localhost:18080/api/v1}"   "token" ""
run_instance "oss-latest"  "main" "${KESTRA_BASE_URL_OSS_LATEST:-http://localhost:38080/api/v1}"  "basic" "ee"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf '%b' "$SUMMARY"

# Exit with failure if any instance failed
[[ $FAIL_COUNT -eq 0 ]]
