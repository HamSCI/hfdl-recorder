# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**hfdl-recorder** is a Python client that subscribes to per-band IQ
multicast streams from one or more ka9q-radio `radiod` instances via
`ka9q-python`, supervises one `dumphfdl` C subprocess per enabled band
(feeding it CF32 IQ via stdin), and writes the decoded JSON to a local
file per band — optionally pushing to `feed.airframes.io:5556` over TCP.

It is part of the HamSCI sigmond suite — see
`/opt/git/sigmond/sigmond/CLAUDE.md` (orchestrator) and
`/opt/git/sigmond/CLAUDE.md` (umbrella) for cross-repo context. Follows
the same Pattern A install layout and contract surface as `psk-recorder`,
`wspr-recorder`, and `hf-timestd`.

Documentation layout:

- `README.md` — top-level summary + install + airframes.io setup.
- `docs/HFDL.md` — what HFDL is, how the worldwide network is
  constructed, what its transmissions carry, and why it is
  valuable as an opportunistic propagation source for HamSCI.
- this file — developer / operator briefing.

## Authors

- Michael Hauan (AC0G, GitHub: mijahauan)
- Repo: https://github.com/HamSCI/hfdl-recorder

## Quick Reference

```bash
# Development
uv sync --extra dev
uv run pytest tests/ -v
uv run hfdl-recorder inventory --json --config config/hfdl-recorder-config.toml.template
uv run hfdl-recorder validate --json --config tests/fixtures/test-config.toml

# pip fallback / run-from-source:
PYTHONPATH=src python3 -m hfdl_recorder inventory --json --config config/hfdl-recorder-config.toml.template

# Production install (uses sigmond's shared _ensure_uv helper)
sudo ./scripts/install.sh           # first-run: user, venv (via uv), dumphfdl build, config, systemd
sudo ./scripts/deploy.sh            # ongoing: refresh editable install + restart instances
sudo ./scripts/deploy.sh --pull     # git pull then deploy

# CLI (current — verify against `hfdl-recorder --help`)
hfdl-recorder inventory --json      # per-instance resource view
hfdl-recorder validate --json       # config validation
hfdl-recorder version --json        # version + git sha
hfdl-recorder status                # health check
hfdl-recorder config init|edit      # whiptail wizard via sigmond.wizard_dispatch
hfdl-recorder daemon --config /etc/hfdl-recorder/hfdl-recorder-config.toml --radiod-id my-rx888
hfdl-recorder squitter-analyze --radiod-id my-rx888     # time saved squitter windows
hfdl-recorder squitter-analyze --dir W --no-sink --jsonl rows.jsonl   # off-station, no sigmond
hfdl-recorder label-check --radiod-id my-rx888 --ref-freq-hz 10000000 --band HFDL13
```

The test suite is moderate (~309 tests). When iterating, target the
affected file with `uv run pytest tests/test_<area>.py -v` rather than
the whole suite.

## Architecture

```
radiod (ka9q-radio)
  │  per-band IQ multicast (one group per HFDL band; usually all on hfdl.local)
  │  preset=iq, samprate=band-specific, encoding=F32LE
  ▼
hfdl-recorder daemon (one per radiod, = one systemd instance)
  │
  ├─ BandPipeline(HFDL21)
  │    ├─ ka9q.MultiStream subscription (float32 IQ samples)
  │    ├─ writer task: float32 → CF32 (numpy view+cast) → dumphfdl stdin
  │    └─ dumphfdl subprocess: --iq-file - --sample-format cf32 --centerfreq 21964 --sample-rate 80000 ...
  │         └─ JSON sinks: local file (always) + feed.airframes.io (opt-in)
  ├─ BandPipeline(HFDL13)
  ├─ BandPipeline(HFDL11)
  └─ ... one per enabled band
  │
  └─ ChTailer(HFDL21..) — one daemon thread per band; tails the per-band
       JSON spool and hands SPDU frames to the squitter keepers; it does
       not stage decoded messages in the sink (Michael 2026-11-13)
```

The daemon requests `F32LE` IQ from radiod (`encoding=4`) even though the
radiod HFDL fragment declares the bands as `s16be`: ka9q-python's IQ
payload parser hard-codes float32 LE, so a distinct `(freq, sample_rate,
encoding)` tuple is provisioned on demand. See `core/radiod.py` for the
rationale.

## Project Structure

```
src/hfdl_recorder/
  cli.py              # CLI entry point, argparse, stdout-cleanliness guard
  config.py           # TOML loader, radiod block resolution, defaults
  configurator.py     # config init|edit subcommands
  contract.py         # inventory/validate JSON builders (contract v0.8)
  bands.py            # static HFDL_BANDS table (12 entries)
  version.py          # GIT_INFO dict for provenance
  core/
    daemon.py         # HfdlRecorder: orchestrates per-band pipelines
    band_pipeline.py  # BandPipeline: ka9q subscription + dumphfdl Popen
    ch_tailer.py      # ChTailer: tails per-band JSON spool → squitter keepers (no hfdl.spots)
    feed.py           # build dumphfdl --output argv from [sinks]
    radiod.py         # ensure_channel() wrapper
  timing/             # squitter timing (off by default: [timing_squitters])
    frame_grid.py     # HFDL frame/slot grid on UTC
    waveform.py       # HFDL waveform constants, synthetic squitters
    analyzer.py       # time one saved squitter window (ANALYZER_VERSION)
    batch.py          # squitter-analyze: windows -> squitter_timing rows
    label_check.py    # label-check CLI (probe channel vs hf-timestd; diagnostic only)
    window_keeper.py  # live: saves labelled 6 kS/s windows around squitters
    channel_status.py # band channel's counter map + least-late anchor
tools/
  squitter_eval.py    # devbox only: offline evaluation of squitter_timing rows
tests/
  test_contract.py
  test_config.py
  test_bands.py
  test_band_pipeline.py
  fixtures/
    test-config.toml
config/
  hfdl-recorder-config.toml.template
systemd/
  hfdl-recorder@.service              # Template unit; %i = reporter key (legacy: radiod_id)
  hfdl-recorder-squitters@.service    # oneshot squitter-analyze (not enabled by default)
  hfdl-recorder-squitters@.timer      # every 15 min, once enabled
scripts/
  install.sh          # First-run bootstrap (Pattern A) + dumphfdl build
  deploy.sh           # Editable-install refresh
  build-dumphfdl.sh   # Vendored libacars + dumphfdl C build
deploy.toml           # Sigmond deploy manifest
```

## Key Design Decisions

- **One systemd instance per receiver** (`hfdl-recorder@<reporter key>`,
  below), or per radiod on a station still on the shared config
  (`hfdl-recorder@<radiod_id>`), matching `psk-recorder` / `wspr-recorder`. The Python daemon supervises
  one `dumphfdl` subprocess per enabled band.
- **Python is in the IQ data path**, but only as a thin float32→CF32
  forwarder via numpy. Matches `wspr-recorder` symmetry (no `pcmrecord`
  dependency). Per-band data rate ≤ 1.1 MB/s on the widest band (HFDL5
  @ 277.2 kS/s); GIL releases during socket recv and subprocess write.
- **dumphfdl is the decoder.** No reimplementation; libacars + dumphfdl
  encode >15k LOC of mature C with years of FER tuning. Built from source
  by `scripts/build-dumphfdl.sh` into `/opt/git/sigmond/hfdl-recorder/bin/dumphfdl`.
- **ka9q-python owns multicast destination** — we never pass
  `destination=` to `ensure_channel()`. Inventory reports the resolved
  address read back from `ChannelInfo`.
- **Per-band restart isolation** — one bad band restarts only its own
  subprocess pair (exponential backoff, cap 60 s); only repeated failures
  across most bands trigger sd_notify failure → unit restart.
- **Aggregators are opt-in** — `sinks.local_json` defaults true,
  `sinks.airframes_io` defaults false; extra TCP/UDP sinks via the
  `sinks.extra` array.  The feed's station name lives in
  `sinks.airframes_station_id`, so turning it on never renames the
  station in the local JSON spool.
- **Decoded messages are not stored in the sink** — since 2026-11-13
  (Michael) the daemon passes `stage_spots=False`, so `hfdl.spots` rows
  are no longer written (on AC0G/B4 they held 266,854 rows / 474 MB of
  a 1 GB sink, and nothing shipped them). `core/ch_tailer.py` still runs
  one daemon thread per band that tails the per-band JSON spool and
  passes SPDU frames to `on_squitter`; with no squitter keepers, the
  daemon builds no tailer. dumphfdl still writes its per-band JSON spool
  and feeds airframes.io only when the operator opts in
  (`airframes_io = true`). Squitter timing rows
  (`hfdl.squitter_timing`) still go to sigmondb.

## Client contract (v0.8)

hfdl-recorder implements the HamSCI client contract at version 0.8
(authoritative source: `/opt/git/sigmond/sigmond/docs/CLIENT-CONTRACT.md`).
`src/hfdl_recorder/contract.py` carries `CONTRACT_VERSION = "0.8"`.

Sections implemented:

- **§1 / §2 / §3 / §4 / §5** — native TOML config, radiod-id binding,
  self-describe CLI (`inventory`/`validate`/`version` `--json`),
  templated systemd unit, `deploy.toml` manifest.
- **§6 / §7** — ka9q-python `MultiStream` per band; data destination
  read from `ChannelInfo`, never client-specified.
- **§8** — `RADIOD_<id>_CHAIN_DELAY_NS` read from `coordination.env`.
- **§10 / §11** — `log_paths` in inventory output (per-band dumphfdl
  stderr files + JSON spool); the daemon process log goes to the
  systemd journal, so it is not listed in `log_paths`.
  `HFDL_RECORDER_LOG_LEVEL` / `CLIENT_LOG_LEVEL` honored on startup
  and SIGHUP.
- **§12.2** — duplicate-band check (the HFDL analogue of psk-recorder's
  SSRC-collision check, since HFDL bands aren't keyed by SSRC).
- **§14** — `config init`/`edit` via `configurator.py`
  (whiptail wizard + `sigmond.wizard_dispatch`).
- **§17** — `ChTailer` reads dumphfdl JSON frames for the squitter
  keepers; decoded messages are not stored in the sink (2026-11-13).
- **§18 (timing authority)** — capability boolean declared in the
  inventory; `timing_authority_applied` always `null` (RTP-default
  mode; no §18 subscriber wired yet).

## External Dependencies (not pip-installable)

- **dumphfdl** (https://github.com/szpajder/dumphfdl) — HFDL waveform
  decoder. Built by `scripts/build-dumphfdl.sh` into
  `/opt/git/sigmond/hfdl-recorder/bin/dumphfdl`.
- **libacars** (https://github.com/szpajder/libacars) — ACARS upper-layer
  parser library. Build dependency of dumphfdl; same script handles both.
- **ka9q-radio radiod** with the HFDL channel fragment loaded
  (`config/fragments/hfdl.conf` or `radiod@*.conf.d/51-hfdl.conf`).

## Production Paths

- Config: `/etc/hfdl-recorder/<reporter key>.toml` per receiver; legacy shared
  `/etc/hfdl-recorder/hfdl-recorder-config.toml`
- JSON spool: `/var/lib/hfdl-recorder/<radiod_id>/<BAND>.json`
- systable: `/var/lib/hfdl-recorder/systable.conf` (auto-updated)
- Per-band logs: `/var/log/hfdl-recorder/<radiod_id>-<BAND>.log` (dumphfdl stderr)
- Squitter windows (only with `[timing_squitters] enabled`):
  `/var/lib/hfdl-recorder/squitters/<radiod_id>/<YYYYMMDD>/<gs>-<kHz>-<frame>.npz`,
  each with a `.done` marker once analysed (it holds the analyser version)
- Squitter timing rows: `hfdl.squitter_timing` in `/var/lib/sigmond/sink.db`
  (local rows, never uploaded; no automatic trim)
- Squitter labels read, never written: hf-timestd's T3 shadow minute rows
  (`/var/lib/timestd/t3_shadow`, the T6 anchor) and its authority history
  (`/var/lib/timestd/authority_history.db`, `mode=ro`).  Each row names the
  ruler that labelled it in `label_origin` (`t6_anchor`, `t1`..`t6`, `host`,
  `unknown`) and, on its sigmondb arrival, in `origin` with the tier's
  published sigma in `u_epoch_ns` (Michael 2026-11-14: report squitters with
  the timing authority the station has at the time).  Only a station whose
  authority held T6 waits for its anchor, and only `T6_ANCHOR_WAIT_S`.
  Details: the `timing/batch.py` docstring.
- label-check result: `/var/lib/hfdl-recorder/label_check_results.toml` (nothing reads it)
- Process log: systemd journal (`StandardOutput=journal`) —
  `journalctl -u hfdl-recorder@<radiod_id>` or `smd log hfdl-recorder`
- Venv: `/opt/git/sigmond/hfdl-recorder/venv`
- Source: `/opt/git/sigmond/hfdl-recorder` (editable install)
- dumphfdl binary: `/opt/git/sigmond/hfdl-recorder/bin/dumphfdl`
- Service user: `hfdlrec:hfdlrec`

## Running Tests

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run pytest tests/test_band_pipeline.py -v          # one file
uv run pytest tests/test_band_pipeline.py::TestX::testY  # one test
uv run pytest -k contract -v                          # by keyword
```

## One instance per receiver, keyed by reporter (2026-11-18)

A receiver (antenna + RX888 + radiod) names every row it produces; a
transporter that runs hfdl-recorder for receivers on another machine
(AC0G-B2 for B1A/B1B, KFS-MAIN) only sends the upload
(`~/hamsci/tasks/transporter-reporters/2026-11-18-design.md`).  So such a
station runs one instance per receiver:

- Units `hfdl-recorder@AC0G-B1A` and, with squitter timing on,
  `hfdl-recorder-squitters@AC0G-B1A.timer` (the reporter key: `/` and `=`
  become `-`).  Both pass `--instance %i --radiod-id %i`.
- Config `/etc/hfdl-recorder/AC0G-B1A.toml`: `[instance] reporter_id =
  "AC0G/B1A"` plus the full config and that receiver's one `[[radiod]]`
  block (`status` = its radiod).  The daemon takes the file's block and
  never synthesizes one from `/etc/radio`; `--radiod-id AC0G-B1A` matches
  nothing and is ignored.  `squitter-analyze` replaces it with the block's
  status, the name the window keeper files windows under.
- Env file `/etc/hfdl-recorder/env/AC0G-B1A.env` (optional).
- Squitter timing rows (`squitter_timing`, sigmondb arrival/capture) carry
  `AC0G/B1A`.  Spool, logs and windows stay keyed by radiod.
- `hfdl-recorder config init --non-interactive --instance AC0G-B1A
  --reporter-id AC0G/B1A --radiod-status <radiod>` writes that file (env
  fallbacks: `SIGMOND_INSTANCE`, `STATION_REPORTER_ID` only when its key is
  the instance's, `SIGMOND_RADIOD_STATUS`).  It refuses rather than guess
  the receiver's reporter or radiod.
- `inventory`/`validate` take `--instance`; with no shared config they
  report every per-receiver file (`config_paths`).  An instance's
  `instance` field is its unit key; it also names its `reporter_id`.
- `install.sh` renders no shared config once per-receiver files exist, and
  enables only their reporter-keyed instances.

Legacy, until sigmond migrates a station: `hfdl-recorder@<radiod status>`
reads the shared `hfdl-recorder-config.toml` (a `DeprecationWarning`); its
squitter rows keep an empty reporter, which the server files under the
uploading site.

See `/opt/git/sigmond/sigmond/docs/MULTI-INSTANCE-ARCHITECTURE.md`
for the architecture and phase plan.
