#!/bin/bash
# install.sh — first-run bootstrap for hfdl-recorder (Pattern A editable install)
#
# Usage: sudo ./scripts/install.sh [--pull] [--yes] [--no-build]
#
# What it does:
#   1. Creates service user hfdlrec:hfdlrec
#   2. Clones/links repo to /opt/git/sigmond/hfdl-recorder
#   3. Creates venv at /opt/hfdl-recorder/venv with editable install
#   4. Builds libacars + dumphfdl into /opt/hfdl-recorder/bin
#      (skip with --no-build if dumphfdl is already on disk)
#   5. Renders config template (non-destructive — never overwrites)
#   6. Installs systemd unit template
#   7. Disables the ka9q-radio hfdl.service if running (we replace it)
#   8. Enables hfdl-recorder@<radiod_id> instances from config
#
# Idempotent: safe to re-run.

set -euo pipefail

SERVICE_USER="hfdlrec"
SERVICE_GROUP="hfdlrec"
REPO_SOURCE="/opt/git/sigmond/hfdl-recorder"
PREFIX="/opt/hfdl-recorder"
VENV_DIR="${PREFIX}/venv"
CONFIG_DIR="/etc/hfdl-recorder"
CONFIG_FILE="${CONFIG_DIR}/hfdl-recorder-config.toml"
SPOOL_DIR="/var/lib/hfdl-recorder"
LOG_DIR="/var/log/hfdl-recorder"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ui_info()  { echo "[INFO]  $*"; }
ui_warn()  { echo "[WARN]  $*" >&2; }
ui_error() { echo "[ERROR] $*" >&2; }

# --- Phase 0: arg parsing ---
DO_PULL=false
AUTO_YES=false
DO_BUILD=true
for arg in "$@"; do
    case "$arg" in
        --pull)     DO_PULL=true ;;
        --yes)      AUTO_YES=true ;;
        --no-build) DO_BUILD=false ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    ui_error "Must run as root (sudo)"
    exit 1
fi

# --- Phase 0.5: ensure uv is on PATH (canonical sigmond-suite installer) ---
# Delegates to sigmond's shared helper if present; inline fallback for
# the bootstrap case.  Keep the fallback in sync with
# sigmond/scripts/install/ensure_uv.sh.
_ENSURE_UV_SH="/opt/git/sigmond/sigmond/scripts/install/ensure_uv.sh"
if [[ -r "$_ENSURE_UV_SH" ]]; then
    # shellcheck source=/dev/null
    source "$_ENSURE_UV_SH"
else
    _ensure_uv() {
        if command -v uv >/dev/null 2>&1; then
            printf '[INFO]  uv %s at %s\n' "$(uv --version 2>/dev/null | awk '{print $2}')" "$(command -v uv)"
            return 0
        fi
        printf '[INFO]  uv not found -- installing system-wide to /usr/local/bin\n'
        command -v curl >/dev/null || { printf '[ERROR] curl not found (apt install curl)\n' >&2; return 1; }
        if ! curl -LsSf https://astral.sh/uv/install.sh | env XDG_BIN_HOME=/usr/local/bin UV_NO_MODIFY_PATH=1 sh; then
            printf '[ERROR] uv installer failed\n' >&2
            return 1
        fi
        command -v uv >/dev/null || { printf '[ERROR] uv installer ran but uv is still not on PATH\n' >&2; return 1; }
        printf '[INFO]  uv %s installed\n' "$(uv --version 2>/dev/null | awk '{print $2}')"
    }
fi
_ensure_uv || { ui_error "_ensure_uv failed"; exit 1; }

# --- Phase 0.6: ensure ka9q-python sibling repo is on disk ---
# pyproject.toml's [tool.uv.sources] declares ka9q-python as a path-based
# editable dep at ../ka9q-python.  uv sync needs the directory to exist
# at /opt/git/sigmond/ka9q-python or it fails.
if [[ ! -f /opt/git/sigmond/ka9q-python/pyproject.toml ]]; then
    ui_info "ka9q-python sibling repo not at /opt/git/sigmond/ka9q-python -- cloning"
    mkdir -p /opt/git/sigmond
    git clone https://github.com/mijahauan/ka9q-python /opt/git/sigmond/ka9q-python \
        || { ui_error "Failed to clone ka9q-python"; exit 1; }
fi

# --- Phase 1: service user ---
if ! id -u "$SERVICE_USER" &>/dev/null; then
    ui_info "Creating service user $SERVICE_USER"
    useradd --system --shell /usr/sbin/nologin \
            --home-dir /nonexistent --no-create-home \
            "$SERVICE_USER"
fi

# --- Phase 2: repo + venv ---
if [[ ! -d "$REPO_SOURCE" ]]; then
    ui_info "Linking $REPO_ROOT -> $REPO_SOURCE"
    mkdir -p "$(dirname "$REPO_SOURCE")"
    ln -sfn "$REPO_ROOT" "$REPO_SOURCE"
fi

# Traversability check (Pattern A defense)
if ! sudo -u "$SERVICE_USER" test -r "$REPO_SOURCE/src/hfdl_recorder/__init__.py"; then
    ui_error "Service user $SERVICE_USER cannot read $REPO_SOURCE/src/hfdl_recorder/__init__.py"
    ui_error "Fix: ensure the repo is at /opt/git/sigmond/hfdl-recorder (not under a mode-700 home)"
    ui_error "  or: chmod g+rx the path and add $SERVICE_USER to the owner's group"
    exit 1
fi

if $DO_PULL; then
    ui_info "Pulling latest from origin"
    git -C "$REPO_SOURCE" pull --ff-only
fi

if [[ ! -d "$VENV_DIR" ]]; then
    ui_info "Creating venv at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    # --seed populates pip/setuptools/wheel for compatibility with tooling
    # that shells out to pip; harmless overhead otherwise.
    uv venv "$VENV_DIR" --python 3.11 --seed --quiet
fi

# uv sync reads pyproject.toml + uv.lock, resolves [tool.uv.sources]
# (ka9q-python editable from ../ka9q-python), installs hfdl-recorder
# itself editable, and pins exactly what's in uv.lock.  --no-dev skips
# dev extras (pytest etc.); --frozen requires uv.lock to be current
# (regenerate locally with `uv lock` if siblings or deps have shifted).
ui_info "Syncing hfdl-recorder + ka9q-python (editable) into $VENV_DIR"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
    uv sync --project "$REPO_SOURCE" --frozen --no-dev --quiet

# sigmond is the host-wide orchestrator; hfdl-recorder lazy-imports
# sigmond.hamsci_sink.Writer in core/ch_tailer.py for the hfdl.spots
# SQLite sink (with a no-op fallback when absent).  Not declared in
# hfdl-recorder's pyproject so uv sync doesn't install it; explicit
# uv pip install when the sibling exists.  uv pip install needs
# --python (UV_PROJECT_ENVIRONMENT only applies to project commands
# like uv sync).
if [[ -d /opt/git/sigmond/sigmond ]]; then
    ui_info "Installing sigmond (editable) into venv"
    uv pip install --quiet --python "$VENV_DIR/bin/python3" -e /opt/git/sigmond/sigmond
else
    ui_info "sigmond repo not found at /opt/git/sigmond/sigmond -- SQLite spot sink"
    ui_info "  will resolve to a no-op fallback."
fi

# Post-install verify
if ! sudo -u "$SERVICE_USER" "$VENV_DIR/bin/python3" -c 'import hfdl_recorder' 2>/dev/null; then
    ui_error "Post-install verify failed: $SERVICE_USER cannot import hfdl_recorder"
    exit 1
fi
ui_info "Post-install verify OK"

# --- Phase 3: dumphfdl C build ---
if $DO_BUILD; then
    if [[ -x "$PREFIX/bin/dumphfdl" ]]; then
        ui_info "dumphfdl already at $PREFIX/bin/dumphfdl (use --no-build to skip, or scripts/build-dumphfdl.sh --force to rebuild)"
    else
        ui_info "Building dumphfdl (libacars + dumphfdl)"
        "$REPO_SOURCE/scripts/build-dumphfdl.sh"
    fi
else
    ui_info "Skipping dumphfdl build (--no-build)"
fi

# --- Phase 4: config ---
mkdir -p "$CONFIG_DIR"
if [[ ! -f "$CONFIG_FILE" ]]; then
    ui_info "Rendering config template -> $CONFIG_FILE"
    cp "$REPO_SOURCE/config/hfdl-recorder-config.toml.template" "$CONFIG_FILE"
    ui_warn "Edit $CONFIG_FILE with your station_id and radiod settings"
else
    ui_info "Config exists at $CONFIG_FILE — not overwriting"
fi

# --- Phase 5: directories ---
for dir in "$SPOOL_DIR" "$LOG_DIR"; do
    mkdir -p "$dir"
    chown "$SERVICE_USER:$SERVICE_GROUP" "$dir"
done

# --- Phase 6: systemd ---
ui_info "Installing systemd unit template"
install -o root -g root -m 644 \
    "$REPO_SOURCE/systemd/hfdl-recorder@.service" \
    /etc/systemd/system/hfdl-recorder@.service
systemctl daemon-reload

# --- Phase 7: disable ka9q-radio's hfdl.service if active ---
# ka9q-radio ships a `hfdl.service` that uses pcmrecord + start-hfdl.sh
# to spawn dumphfdl per band. We replace that whole pipeline.
if systemctl is-active --quiet hfdl.service 2>/dev/null; then
    ui_warn "Disabling ka9q-radio hfdl.service (hfdl-recorder replaces it)"
    systemctl disable --now hfdl.service
fi

# --- Phase 8: enable instances ---
ui_info "Parsing radiod IDs from $CONFIG_FILE"
RADIOD_IDS=$("$VENV_DIR/bin/python3" -c "
import tomllib
with open('$CONFIG_FILE', 'rb') as f:
    cfg = tomllib.load(f)
blocks = cfg.get('radiod', [])
if isinstance(blocks, dict):
    blocks = [blocks]
for b in blocks:
    print(b.get('id', 'default'))
" 2>/dev/null)

if [[ -z "$RADIOD_IDS" ]]; then
    ui_warn "No radiod IDs found in config — no instances enabled"
else
    for rid in $RADIOD_IDS; do
        ui_info "Enabling hfdl-recorder@${rid}.service"
        systemctl enable "hfdl-recorder@${rid}.service"
        ui_info "  (not starting — review $CONFIG_FILE first)"
    done
fi

ui_info "Install complete. Edit $CONFIG_FILE then start instances with:"
ui_info "  sudo systemctl start hfdl-recorder@<radiod-id>"
