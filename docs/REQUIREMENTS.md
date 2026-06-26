# hfdl-recorder — Requirements Specification

**Status:** v0.1 baseline (retroactive). **Owner:** Michael Hauan (AC0G).
**Last reconciled against code:** hfdl-recorder `0.1.0` / deploy `0.1.0` /
contract `0.8` (2026-06-25, `fdb28eb`).
**Prefix:** `HFD`.

> Application of [sigmond/docs/REQUIREMENTS-TEMPLATE.md](https://github.com/HamSCI/sigmond/blob/main/docs/REQUIREMENTS-TEMPLATE.md),
> for an **Active** recorder client. The sigmond↔component **interface**
> requirements are specified once in the
> [client contract](https://github.com/HamSCI/sigmond/blob/main/docs/CLIENT-CONTRACT.md)
> (v0.8) and referenced — not restated — here (§8.3). Provenance tags:
> `[DOC]` documented · `[CODE]` implicit-in-code · `[NEW]` surfaced by this review.
> Status: ✅ implemented · 🟡 partial/unverified · ⬜ planned.

## 1. Context & problem statement

HFDL (High Frequency Data Link) is the aeronautical industry's HF packet-data
service, operated under ARINC Specification 635 from ~a dozen GPS-disciplined
ground stations worldwide on 12 sub-bands spanning roughly 2.8–22 MHz. Every
transmission carries the ground station's identity and a UTC timestamp, and
many downlink frames carry aircraft position reports — so each decoded frame is
an **opportunistic propagation observation with both endpoints known** (ground
station location is fixed and published; aircraft position is in the payload).
That density of well-anchored skywave paths is the scientific by-product HamSCI
cares about; the live community feed to airframes.io is the operational payoff
that recruits stations.

`hfdl-recorder` is the sigmond-suite client that turns a `radiod` receiver into
an HFDL station. It subscribes to one per-band wide-IQ multicast stream per
enabled HFDL band from one or more `radiod` instances via `ka9q-python`,
supervises **one `dumphfdl` subprocess per band** (feeding it CF32 IQ over
stdin), and routes the decoded JSON traffic three ways: a canonical per-band
local JSON file (always), an opt-in direct TCP push to `feed.airframes.io:5556`
(emitted by dumphfdl's own `--output` sink), and an always-on parse into the
shared HamSCI SQLite sink table `hfdl.spots` (via an in-process tailer thread).

Its defining design principle: **`dumphfdl` is the decoder** — the recorder does
not reimplement HFDL demodulation/FEC/ACARS parsing (>15k LOC of mature, FER-tuned
C in libacars + dumphfdl). The Python daemon is a thin, restart-isolating
supervisor and IQ forwarder. It follows the same Pattern-A install/contract
surface as psk-recorder, wspr-recorder, and hf-timestd.

## 2. Goals & objectives

- Decode HFDL traffic from the high-yield bands across day/night propagation
  using `dumphfdl`, one subprocess per band, with **per-band fault isolation**.
- Feed **airframes.io** (the de-facto community aggregator) directly from
  dumphfdl when the operator opts in and has registered a station name.
- Produce a durable per-band **local JSON** record as the canonical artifact,
  independent of any aggregator.
- Land each decoded frame as a science-grade row in the shared **`hfdl.spots`**
  sink (ground station, ICAO/flight, position, signal level), no-op when the
  sink is unwritable.
- Run as a well-behaved suite client (multi-instance per radiod, off radiod
  cores, contract-conformant self-description) **and** usefully standalone
  (radiod + dumphfdl + local JSON, no sigmond required).

## 3. Non-goals / out of scope

- **Reimplementing the HFDL decoder.** Demod, FEC, deinterleaving, and ACARS
  upper-layer parsing are owned by `dumphfdl`/`libacars` (external binary).
- **Being a receiver / tuning hardware.** It consumes pre-provisioned IQ from
  `radiod`; channel destinations are owned by ka9q-python. (Owner: ka9q-radio.)
- **Producing a timing authority.** It runs RTP-default (HFDL decode is
  ms-tolerant); the timing authority is hf-timestd's job (§18 producer).
- **Uploading the `hfdl.spots` sink upstream.** The recorder is the *producer*
  side; draining `hfdl.spots` to any HamSCI/PSWS endpoint is `hs-uploader` scope.
- **Aircraft tracking / situational-display products.** Downstream consumers of
  `hfdl.spots`, not this recorder.

## 4. Stakeholders & actors

Station operator · `radiod` (ka9q-radio; per-band wide-IQ multicast source,
required) · `ka9q-python` (`MultiStream`/`RadiodControl`, required) · **`dumphfdl`**
(external decoder binary, required) + `libacars` (its build dependency) ·
`airframes.io` (`feed.airframes.io:5556`, opt-in aggregator; requires a
registered station name) · the shared SQLite sink + `hs-uploader` (downstream
drainer) + downstream HamSCI science consumers · `hf-timestd` (§18 timing-authority
producer, not yet consumed) · sigmond (multi-instance lifecycle, CPU affinity,
status/inventory, config wizard) · the ka9q-radio HFDL fragment
(`config/fragments/hfdl.conf`, band/samprate source of truth).

## 5. Assumptions & constraints

- `HFD-C-001` `[DOC]` ✅ `radiod` SHALL be present and multicasting per-band HFDL
  IQ channels (the HFDL fragment loaded); there is no unicast fallback.
- `HFD-C-002` `[CODE]` ✅ The **`dumphfdl` binary SHALL exist** at the configured
  path (default `/opt/git/sigmond/hfdl-recorder/bin/dumphfdl`, built from source
  by `scripts/build-dumphfdl.sh`); its absence is a hard `validate` failure. This
  external, non-pip-installable C dependency is the load-bearing constraint of
  the whole component.
- `HFD-C-003` `[CODE]` ✅ The daemon SHALL request `F32LE` IQ (`encoding=4`) from
  radiod even though the HFDL fragment declares `s16be`, because ka9q-python's
  IQ payload parser hard-codes float32 LE; a distinct `(freq, sample_rate,
  encoding)` tuple is provisioned on demand (`core/radiod.py`).
- `HFD-C-004` `[CODE]` ✅ Python is in the IQ data path but only as a thin
  float32→CF32 forwarder (numpy view+cast); per-band rate ≤ ~1.1 MB/s on the
  widest band (HFDL5 @ 277.2 kS/s). The GIL releases on socket recv and
  subprocess write.
- `HFD-C-005` `[CODE]` ✅ One systemd **instance per radiod** (`%i` = radiod_id =
  the mDNS status name); the band set is per-instance config, never an instance
  key.
- `HFD-C-006` `[CODE]` ✅ Python ≥3.10; runtime deps `ka9q-python>=3.14.0`,
  `numpy>=1.24`, `tomli` (<3.11). `ka9q-python` is an editable sibling install.
- `HFD-C-007` `[CODE]` ✅ Band centers/sample rates SHALL track the ka9q-radio
  HFDL fragment; the static `HFDL_BANDS` table (12 bands) is vendored from it
  and `aux/start-hfdl.sh`, reconciled on upstream change.

## 6. Functional requirements

### 6.1 Acquisition (per band)
- `HFD-F-001` `[DOC]` ✅ SHALL register one ka9q-python `MultiStream` channel per
  enabled band (`preset=iq`, band-specific `samprate`, `F32LE`) against the
  resolved radiod, grouped onto a single `MultiStream` per radiod.
- `HFD-F-002` `[CODE]` ✅ SHALL forward delivered float32 IQ to that band's
  `dumphfdl` stdin as interleaved CF32, zeroing NaN/Inf samples
  (`np.nan_to_num`) so the decoder never sees a garbage sample from a
  packet-drop gap.
- `HFD-F-003` `[CODE]` ✅ SHALL provision channels **once** at startup and treat a
  ka9q stream drop/restore as a no-op (SSRC is stable, derived from freq/preset;
  the slot resumes on restore).
- `HFD-F-004` `[CODE]` ✅ When `radiod_lifetime_frames > 0`, SHALL run a
  LIFETIME keep-alive thread refreshing each channel's lifetime at ~4× safety
  margin; SHALL no-op when 0.

### 6.2 Decoding (the dumphfdl subprocess model)
- `HFD-F-010` `[DOC]` ✅ SHALL supervise exactly one `dumphfdl --iq-file -
  --sample-format cf32` subprocess per enabled band, with `--centerfreq`/
  `--sample-rate` from the band table and the band's in-band channel list as
  positional args.
- `HFD-F-011` `[CODE]` ✅ SHALL pass `--station-id` (when set) and
  `--system-table`/`--system-table-save` (default
  `/var/lib/hfdl-recorder/systable.conf`) so dumphfdl learns and persists
  ground-station IDs from squitters.
- `HFD-F-012` `[CODE]` ✅ SHALL write each band's `dumphfdl` stderr to a per-band
  log file `<log_dir>/<radiod_id>-<BAND>.log`.
- `HFD-F-013` `[CODE]` ✅ SHALL restart a dead `dumphfdl` with exponential backoff
  (1 s → cap 60 s); a run ≥30 s resets the backoff. Per-band restart isolation —
  one bad band restarts only its own subprocess.

### 6.3 Output / forwarding (the three sinks)
- `HFD-F-020` `[DOC]` ✅ SHALL always write decoded JSON (one object per line) to
  the per-band local file `<spool_dir>/<radiod_id>/<BAND>.json` when
  `sinks.local_json` (default true) — the canonical artifact.
- `HFD-F-021` `[DOC]` ✅ When `sinks.airframes_io` (default false), SHALL emit a
  `dumphfdl` `--output decoded:json:tcp:address=feed.airframes.io,port=5556`
  sink — a **direct dumphfdl-internal TCP feed**, not routed through the shared
  sink. SHALL require `station.station_id` (validate `fail` otherwise).
- `HFD-F-022` `[CODE]` ✅ SHALL support arbitrary additional sinks via
  `sinks.extra` (`tcp`/`udp`/`file`, `json`/`text`), formatted into dumphfdl
  `--output` specs with stable ordering.
- `HFD-F-023` `[DOC]` ✅ SHALL run one in-process `ChTailer` thread per band that
  tails the per-band JSON spool and inserts parsed frames into the shared SQLite
  sink table **`hfdl.spots`** (`mode=hfdl`, schema `hfdl:1`) via
  `sigmond.hamsci_sink.Writer.from_env()` — independent of dumphfdl's
  airframes.io TCP feed.
- `HFD-F-024` `[CODE]` ✅ The tailer SHALL extract high-signal fields (timestamp,
  ground station, ICAO addr, flight/reg, ACARS label/message, lat/lon/alt,
  sig/noise level, freq skew, direction) best-effort and preserve the full
  `raw_json` for future re-parse; an unparseable frame is skipped silently.
- `HFD-F-025` `[CODE]` ✅ Each `hfdl.spots` row SHALL carry provenance:
  `host_call`, `host_grid`, `radiod_id`, `reporter_id`, `processing_version`.

### 6.4 Service profiles & control
- `HFD-F-030` `[CODE]` ✅ SHALL be `Type=notify`; SHALL `sd_notify(READY=1)` after
  all band pipelines start and ping `WATCHDOG` (`WatchdogSec=120`).
- `HFD-F-031` `[CODE]` 🟡 Only repeated dumphfdl failures across most bands SHALL
  escalate to an sd_notify failure → unit restart; a single bad band SHALL NOT
  take the unit down. *(escalation policy present; threshold unverified.)*

### 6.5 Self-description & config (contract surface)
- `HFD-F-040` `[CODE]` ✅ SHALL implement `inventory --json` / `validate --json` /
  `version --json` / `status` per contract v0.8 (see §8.3) with pure-JSON stdout.
- `HFD-F-041` `[CODE]` ✅ SHALL implement `config init|edit` via `configurator.py`
  (whiptail wizard + `sigmond.wizard_dispatch`), honoring the §14.3 env bag
  (`STATION_*`, `SIGMOND_INSTANCE`, `SIGMOND_RADIOD_STATUS`).
- `HFD-F-042` `[CODE]` ✅ `validate` SHALL `fail` on: missing `dumphfdl` binary;
  no `[[radiod]]` block; a radiod with no `status`; `airframes_io` with no
  `station_id`; an unknown band name; a band listed twice. SHALL `warn` on empty
  `station_id` or no bands enabled; SHALL emit `info` for the dumphfdl version
  and the HFDL15-only squitter case.
- `HFD-F-043` `[CODE]` ✅ SHALL prefer `/etc/hfdl-recorder/<instance>.toml` when
  present (post `smd instance migrate`), falling back to the legacy shared
  `hfdl-recorder-config.toml` with a one-line `DeprecationWarning`.

## 7. Quality / non-functional requirements

- `HFD-Q-001` `[CODE]` ✅ The recorder SHALL run off radiod's CPU cores (sigmond
  `AFFINITY_UNITS`) — decode threads polluting radiod's L3 cause RX888 USB drops.
- `HFD-Q-002` `[CODE]` ✅ A single failing band SHALL be isolated: its subprocess
  pair restarts under backoff without disturbing the other bands.
- `HFD-Q-003` `[CODE]` ✅ Sink writes SHALL degrade to a clean **no-op** when the
  shared SQLite sink is unwritable; local JSON and the airframes feed SHALL be
  unaffected.
- `HFD-Q-004` `[CODE]` ✅ The IQ forwarder SHALL never crash the band on a dying
  subprocess (`BrokenPipeError`/`OSError` are swallowed; the supervisor restarts).
- `HFD-Q-005` `[CODE]` ✅ Memory SHALL be bounded by `MemoryMax=2G`,
  `MemorySwapMax=0`, `Nice=5`; the unit runs under `ProtectSystem=strict` with
  `ReadWritePaths` limited to the spool + log dirs and `CAP_NET_RAW`/
  `CAP_NET_BIND_SERVICE` only.
- `HFD-Q-006` `[CODE]` ✅ The tailer SHALL batch (default 200 rows) and flush at a
  bounded cadence (30 s), and SHALL handle file truncation/rotation (reset on
  shrink) and partial trailing lines.
- `HFD-Q-007` `[CODE]` ✅ stdout SHALL be JSON-clean for the self-describe verbs
  (CLI stdout-cleanliness guard); the process log goes to the journal, not files.
- `HFD-Q-008` `[NEW]` 🟡 dumphfdl JSON volume estimate in inventory
  (`mb_per_day`) is a coarse `max(1, n_bands)` placeholder; SHALL be measured
  against a live station for the disk-budget summary to be meaningful. *(gap.)*

## 8. External interfaces

### 8.1 Inputs
- radiod per-band wide-IQ via ka9q-python (`F32LE`, band-specific samprate),
  default 7 enabled bands → 7 channels: HFDL5/6/8/10/11/13/21 (centers
  5587000, 6622000, 8902500, 10061500, 11287000, 13310000, 21964000 Hz).
- `/etc/hfdl-recorder/hfdl-recorder-config.toml` (or per-instance
  `<instance>.toml`). Operator MUST set: `[station].station_id` (must match the
  airframes.io registered name if feeding); `[[radiod]].status` (mDNS name);
  `[radiod.bands].enabled`. Optional: `[station].grid_square`, `[paths]`,
  `[sinks].airframes_io`/`extra`, `[timing].chain_delay_ns`.
- Coordination/identity from `/etc/sigmond/coordination.env` +
  `/etc/hfdl-recorder/env/<instance>.env`; `RADIOD_<id>_CHAIN_DELAY_NS`,
  `HFDL_RECORDER_LOG_LEVEL`/`CLIENT_LOG_LEVEL`.
- The external **`dumphfdl`** binary (+ its persisted `systable.conf`).

### 8.2 Outputs
- **Local JSON** (canonical): `/var/lib/hfdl-recorder/<radiod_id>/<BAND>.json`,
  one dumphfdl JSON object per line.
- **airframes.io** (opt-in): direct TCP push `feed.airframes.io:5556` from
  dumphfdl's `--output decoded:json:tcp:…`.
- **Shared sink:** `hfdl.spots` rows (target_db `hfdl`, table `spots`, schema
  `hfdl:1`) staged into `/var/lib/sigmond/sink.db` via `hamsci_sink.Writer`;
  fields per `core/ch_tailer.py` (time, band_name, station_id, frequency,
  bit_rate, sig/noise_level, freq_skew, slot, direction, ground_station,
  icao_addr, flight, aircraft_reg, acars_label/message, position lat/lon/alt,
  raw_json, + host_call/host_grid/radiod_id/reporter_id/processing_version).
- **Per-band dumphfdl stderr logs:** `/var/log/hfdl-recorder/<radiod_id>-<BAND>.log`.
- **Process log:** systemd journal (`smd log hfdl-recorder`).
- **Self-description:** `inventory`/`validate`/`version --json` (derived in §8.1/§8.3).

### 8.3 Contracts / APIs (reference, not restated)
- `HFD-I-001` `[CODE]` ✅ Conforms to **client contract v0.8** (multi-instance);
  `deploy.toml` declares `contract_version="0.8"`,
  `templated_units=["hfdl-recorder@.service"]`, `[contract.config]` init/edit,
  a `build` step set (venv + `build-dumphfdl.sh`), `deps.binary=[dumphfdl]`,
  `deps.git=[ka9q-radio]`, `deps.pypi=[ka9q-python]`, and `client_features`
  (watch verb `hfdl`, receiver_channels parser). `inventory` declares one
  instance per `[[radiod]]`, `data_sinks=[file (json spool), file (logs)]`,
  `ka9q_channels`/`frequencies_hz`/`bands`/`modes=[hfdl]`,
  `provides_timing_calibration=false`. Full field semantics: contract
  §3/§6/§7/§16/§17.
  *(Doc-drift note: CLAUDE.md prose still says "v0.7"; the code constant
  `CONTRACT_VERSION` and `deploy.toml` are both `0.8` — see HFD-F-091.)*
- `HFD-I-002` `[CODE]` 🟡 **Timing-authority consumer (declared, not wired):**
  `inventory` reports `uses_timing_calibration=false`,
  `timing_authority_applied=null` (explicit-null, distinguishing
  contract-aware-in-default-mode from a pre-v0.7 client). HFDL decode is
  ms-tolerant; no §18 (`AuthorityReader`) consumption is wired (see HFD-F-090).
  `chain_delay_ns_applied` is read from `coordination.env` when present.
- `HFD-I-003` `[DOC]` ✅ The **airframes.io feed seam** is governed externally:
  the live feed requires a station name **registered at app.airframes.io** and
  software `HFDL-DUMPHFDL`; a mismatched `station_id` is silently dropped
  server-side. This is not a sigmond contract surface — dumphfdl talks to
  airframes directly.

## 9. Data requirements

`hfdl.spots` (schema `hfdl:1`, additive, channel/band-keyed): one row per decoded
frame with the fields in §8.2. The per-band local JSON is the canonical
artifact; `data_sinks.retention_days=0` for the spool (operator-managed) and
`365` for logs. Volume is bursty and modest (order ~kB–MB/band/day; inventory
estimate is a placeholder — HFD-Q-008). `systable.conf` is per-instance writable
state that dumphfdl reads and updates as it learns ground-station IDs. Timing
provenance is RTP-default (no applied authority); every spot carries
`reporter_id`/`processing_version` for downstream attribution.

## 10. Dependencies & development sequence

**Runtime deps:** `radiod` (required), **`dumphfdl` + `libacars`** (external C,
built from source — the load-bearing dep), `ka9q-python>=3.14.0` (editable
sibling), `numpy>=1.24`, `tomli` (<3.11). Optional: `sigmond.hamsci_sink`
(lazy-imported for the sink; no-op if absent), `sigmond.wizard_dispatch` (config
wizard). The ka9q-radio HFDL fragment must be loaded in radiod.

**Development sequence (intended, recovered as requirement):**
- **v0.1 (current, Active):** per-band dumphfdl supervision + IQ forwarding;
  three output paths (local JSON, airframes.io opt-in, `hfdl.spots` tailer);
  contract v0.8 self-description; multi-instance per-radiod cutover (Phase 5 —
  `--instance %i`, per-instance config, `reporter_id`).
- **Next:** §18 timing-authority consumption if/when sub-frame timing is wanted
  (HFD-F-090); move the airframes/upstream upload path onto `hs-uploader`
  (HFD-F-093); measured disk-budget estimate (HFD-Q-008).

## 11. Acceptance criteria & verification

- Contract conformance → `hfdl-recorder validate --json` (exit 0, no `fail`) +
  surfaced via `smd status`. The dumphfdl-missing `fail` is the canonical
  install gate (HFD-C-002).
- Decode liveness → per-band `tail -f <spool>/<BAND>.json` shows frames; the
  `smd watch hfdl` verb shows per-band frame counts + GS/aircraft.
- airframes.io feed → `ss -tnp | grep feed.airframes.io` ESTAB (one per band) +
  the station dashboard at app.airframes.io populates (HFD-F-021/I-003).
- Sink integrity → `hfdl.spots` rows accrue when the sink is writable; clean
  no-op (local JSON unaffected) when not (HFD-Q-003).
- Fault isolation → killing one band's dumphfdl restarts only that band under
  backoff; unit stays up (HFD-Q-002).
- Standalone operability → `scripts/install.sh` on a radiod-only host reaches a
  decoding `status` without sigmond present.

## 12. Risks & open questions

- `HFD-F-090` `[NEW]` ⬜ **§18 timing authority read-but-not-consumed:**
  `timing_authority_applied=null`, no `AuthorityReader` wired. Acceptable while
  HFDL is ms-tolerant; SHALL be closed if any sub-frame timing/Doppler product
  is ever claimed. *(candidate #18 Clients issue.)*
- `HFD-F-091` `[NEW]` 🟡 **Contract-version doc drift:** CLAUDE.md prose states
  "client contract (v0.7)" while `contract.py` (`CONTRACT_VERSION="0.8"`) and
  `deploy.toml` (`contract_version="0.8"`) say 0.8, and the docstrings still
  reference "v0.6". The prose SHALL be reconciled to 0.8.
- `HFD-F-092` `[NEW]` 🟡 **No `dumphfdl` version pin / reconciliation:** the band
  table and `--output`/`systable` flags are tied to a dumphfdl built from
  `main`; an upstream flag/JSON-schema change could silently break parsing
  (`ch_tailer` is tolerant but field coverage could regress). SHALL pin or
  reconcile dumphfdl on update (analogue of the SuperDARN vendored-data rule).
- `HFD-F-093` `[NEW]` ⬜ **Producer-only sink:** `hfdl.spots` is staged locally;
  draining it upstream (HamSCI/PSWS) via `hs-uploader` is not yet wired — the
  airframes.io feed is currently the only egress for the science endpoint data.
- `HFD-Q-008` `[NEW]` 🟡 **Disk-budget estimate is a placeholder** (HFD-Q-008
  above) — measure on a live station.
- Restart-escalation threshold (HFD-F-031) — "most bands failing" is
  qualitative; confirm the concrete sd_notify-failure condition.

## 13. Traceability

| Requirement | #18 issue | Verification | PSWS #6 |
|---|---|---|---|
| HFD-I-001 (contract v0.8) | Clients: hfdl-recorder | `validate --json` exit 0 / `smd status` | #6:31 (sensor integ.) |
| HFD-F-021 (airframes.io feed) | — | ESTAB socket + dashboard populates | — |
| HFD-F-023 (`hfdl.spots` sink) | Clients: hfdl-recorder | sink rows accrue; no-op when unwritable | #6:31 |
| HFD-C-002 (dumphfdl required) | — | `validate` fail when binary absent | — |
| HFD-F-090 (§18 consumption) | *(new — file)* | timing-authority test | #6:50 |
| HFD-F-091 (contract-version drift) | *(new — file)* | doc review | — |
| HFD-F-092 (dumphfdl reconciliation) | *(new — file)* | upstream-drift check | — |
| HFD-F-093 (hs-uploader egress) | *(new — file)* | upload path test | #6:40 |

*New rows (HFD-F-090/091/092/093, HFD-Q-008) are this review's surfaced gaps;
promote to the #18 hfdl-recorder Clients epic.*
