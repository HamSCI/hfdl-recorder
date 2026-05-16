# CLAUDE.md — hfdl-recorder Development Briefing

## What this project is

**hfdl-recorder** is a Python client that subscribes to per-band IQ
multicast streams from one or more ka9q-radio `radiod` instances via
`ka9q-python`, supervises one `dumphfdl` C subprocess per enabled band
(feeding it CF32 IQ via stdin), and writes the decoded JSON to a local
file per band — optionally pushing to `feed.airframes.io:5556` over TCP.

It is the fourth client in the HamSCI sigmond contract v0.6 family
(after `psk-recorder`, `wspr-recorder`, `hf-timestd`) and follows the
same Pattern A install layout, deploy ergonomics, and contract surface.

## Authors

- Michael Hauan (AC0G, GitHub: mijahauan)
- Repo: https://github.com/mijahauan/hfdl-recorder

## Quick Reference

```bash
# Development
uv sync --extra dev
uv run pytest tests/ -v
uv run hfdl-recorder inventory --json --config config/hfdl-recorder-config.toml.template
uv run hfdl-recorder validate --json --config tests/fixtures/test-config.toml

# pip fallback / run-from-source:
PYTHONPATH=src python3 -m hfdl_recorder inventory --json --config config/hfdl-recorder-config.toml.template

# Production install (Pattern A editable install)
sudo ./scripts/install.sh           # first-run: user, venv, dumphfdl build, config, systemd
sudo ./scripts/deploy.sh            # ongoing: pip install -e, restart instances
sudo ./scripts/deploy.sh --pull     # git pull then deploy

# CLI
hfdl-recorder inventory --json
hfdl-recorder validate --json
hfdl-recorder version --json
hfdl-recorder daemon --config /etc/hfdl-recorder/hfdl-recorder-config.toml --radiod-id my-rx888
```

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
       JSON spool and inserts parsed frames into hfdl.spots via
       sigmond.hamsci_ch.Writer.from_env() (local SQLite sink by default)
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
  contract.py         # inventory/validate JSON builders (contract v0.6)
  bands.py            # static HFDL_BANDS table (12 entries)
  version.py          # GIT_INFO dict for provenance
  core/
    daemon.py         # HfdlRecorder: orchestrates per-band pipelines
    band_pipeline.py  # BandPipeline: ka9q subscription + dumphfdl Popen
    ch_tailer.py      # ChTailer: tails per-band JSON spool → hfdl.spots
    feed.py           # build dumphfdl --output argv from [sinks]
    radiod.py         # ensure_channel() wrapper
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
  hfdl-recorder@.service   # Template unit; %i = radiod_id
scripts/
  install.sh          # First-run bootstrap (Pattern A) + dumphfdl build
  deploy.sh           # Editable-install refresh
  build-dumphfdl.sh   # Vendored libacars + dumphfdl C build
deploy.toml           # Sigmond deploy manifest (contract v0.6)
```

## Key Design Decisions

- **One systemd instance per radiod** (`hfdl-recorder@<radiod_id>.service`),
  matching `psk-recorder` / `wspr-recorder`. The Python daemon supervises
  one `dumphfdl` subprocess per enabled band.
- **Python is in the IQ data path**, but only as a thin float32→CF32
  forwarder via numpy. Matches `wspr-recorder` symmetry (no `pcmrecord`
  dependency). Per-band data rate ≤ 1.1 MB/s on the widest band (HFDL5
  @ 277.2 kS/s); GIL releases during socket recv and subprocess write.
- **dumphfdl is the decoder.** No reimplementation; libacars + dumphfdl
  encode >15k LOC of mature C with years of FER tuning. Built from source
  by `scripts/build-dumphfdl.sh` into `/opt/hfdl-recorder/bin/dumphfdl`.
- **ka9q-python owns multicast destination** — we never pass
  `destination=` to `ensure_channel()`. Inventory reports the resolved
  address read back from `ChannelInfo`.
- **Per-band restart isolation** — one bad band restarts only its own
  subprocess pair (exponential backoff, cap 60 s); only repeated failures
  across most bands trigger sd_notify failure → unit restart.
- **Aggregators are opt-in** — `sinks.local_json` defaults true,
  `sinks.airframes_io` defaults false; extra TCP/UDP sinks via the
  `sinks.extra` array.
- **Spot tailer feeds `hfdl.spots`** — `core/ch_tailer.py` runs one
  daemon thread per band that tails the per-band JSON spool and inserts
  parsed frames via `sigmond.hamsci_ch.Writer.from_env()`, which stages
  rows into sigmond's local SQLite sink (`/var/lib/sigmond/sink.db`) by
  default. It resolves to a no-op only when the sink path is unwritable.
  Independent of dumphfdl's own airframes.io TCP feed (which is
  dumphfdl-internal).

## Client Contract (v0.6)

hfdl-recorder implements the HamSCI client contract v0.6 as defined in
`sigmond/docs/CLIENT-CONTRACT.md`. Key surfaces:

- `hfdl-recorder inventory --json` — per-instance resource view
- `hfdl-recorder validate --json` — config validation
- `deploy.toml` — build/install manifest
- `EnvironmentFile=-/etc/sigmond/coordination.env` in the systemd unit
- §7: data destination read from ka9q-python, not client-specified
- §8: `RADIOD_<id>_CHAIN_DELAY_NS` read from env on startup
- §10: `log_paths` in inventory output (per-band logs, JSON sinks); the
  daemon process log goes to the systemd journal, not a file, so it is
  not listed in `log_paths`
- §11: `HFDL_RECORDER_LOG_LEVEL` / `CLIENT_LOG_LEVEL` honored on startup
  and SIGHUP
- §12.2: duplicate-band check (analogue of psk-recorder's SSRC collision check)
- §17: `ChTailer` parses dumphfdl frames into `hfdl.spots`

## External Dependencies (not pip-installable)

- **dumphfdl** (https://github.com/szpajder/dumphfdl) — HFDL waveform
  decoder. Built by `scripts/build-dumphfdl.sh` into
  `/opt/hfdl-recorder/bin/dumphfdl`.
- **libacars** (https://github.com/szpajder/libacars) — ACARS upper-layer
  parser library. Build dependency of dumphfdl; same script handles both.
- **ka9q-radio radiod** with the HFDL channel fragment loaded
  (`config/fragments/hfdl.conf` or `radiod@*.conf.d/51-hfdl.conf`).

## Production Paths

- Config: `/etc/hfdl-recorder/hfdl-recorder-config.toml`
- JSON spool: `/var/lib/hfdl-recorder/<radiod_id>/<BAND>.json`
- systable: `/var/lib/hfdl-recorder/systable.conf` (auto-updated)
- Per-band logs: `/var/log/hfdl-recorder/<radiod_id>-<BAND>.log` (dumphfdl stderr)
- Process log: systemd journal (`StandardOutput=journal`) —
  `journalctl -u hfdl-recorder@<radiod_id>` or `smd log hfdl-recorder`
- Venv: `/opt/hfdl-recorder/venv`
- Source: `/opt/git/sigmond/hfdl-recorder` (editable install)
- dumphfdl binary: `/opt/hfdl-recorder/bin/dumphfdl`
- Service user: `hfdlrec:hfdlrec`

## Running Tests

```bash
uv sync --extra dev
uv run pytest tests/ -v
```


1. Don’t assume. Don’t hide confusion. Surface tradeoffs.

2. Minimum code that solves the problem. Nothing speculative.

3. Touch only what you must. Clean up only your own mess.

4. Define success criteria. Loop until verified.
