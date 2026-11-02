# HFDL Squitter Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Time HFDL ground-station squitters on the sample counter of UTC-labelled IQ windows, and evaluate whether they improve UTC reconstruction overnight.

**Architecture:** The station side copies a labelled window of samples around each expected squitter from two or three chosen ground stations, inside hfdl-recorder's existing sample path, doing no signal processing beyond a narrow-and-decimate.  A batch analyser, run at idle priority, correlates each window against the known HFDL pre-key and preamble and writes one local sink row per squitter.  An offline solver on the devbox scores the result.  A label check gates everything: no squitter gets timed until each HFDL channel's UTC label proves good to 0.1 ms.

**One change from the spec:** the spec asked for a model of radiod's filter delay per band rate, checked on the rig.  Task 1's label check measures that delay directly instead, together with any label error, against hf-timestd's own channel; Task 4 subtracts the measured constant.  Running `label-check` on the rig covers the spec's rig check.

**Tech Stack:** Python 3.11, numpy, ka9q-python (`MultiStream`, `SlotClock`, `rtp_to_utc`), hamsci-dsp (`acquire_anchor_utc`, `AuthorityReader`), sigmond `hamsci_sink.Writer` (`local=True`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-squitter-timing-design.md` (approved 2026-11-02).

## Global Constraints

- Squitters only (uplink, slot 0 of each ground station's frame); no aircraft bursts.
- No signal processing in `_on_samples` beyond a copy, a mix and a decimation; correlation never runs in the sample path.  Never block `_on_samples`: a full queue drops the window and counts the drop.
- dumphfdl decoding and the airframes.io feed must behave exactly as before, with the feature on or off.
- Off by default: `[timing_squitters] enabled = false`.
- Two labels per window (spec, "Changed after the label gate", Michael 2026-11-02): the counter label (radiod input-sample index of the first saved sample, via the channel's own `hamsci_dsp.counter_map.CounterMap`) scores results; the status-pair label (`hamsci_dsp.timing.acquire_anchor_utc` with the hf-timestd authority offset) rides beside it as a record.  The host clock only disambiguates the RTP wrap.
- Nothing leaves the station: sink rows written with `Writer.from_env(..., local=True)`; windows stay on disk.
- Windows: from 30 ms before the nominal slot start to the slot's end (32/13 s); narrowed to one 3 kHz channel at 6000 S/s complex64.
- Retention on the station: 5 days of windows.
- `utc_sync` false on a squitter: record the row, mark it, exclude it from results.
- Protocol constants (pre-key length, A, M1 with its 8 shifts, M2, T) come from ARINC 635 as dumphfdl 1.7 encodes them (`src/hfdl.c`).  dumphfdl carries GPLv3 and hfdl-recorder MIT: take the sequences as protocol facts, cite the spec, and use dumphfdl only as a test oracle.  Write our own pulse shaping; copy no dumphfdl code, including its matched-filter taps.
- Measure any CPU cost on the Proxmox host, never inside the VM.
- Station work: read the claude-bus first and announce there before deploying.  Deploy by git (Michael, 2026-11-02), ahead of a blessed release.

## Review Focus

1. A ground station that changes frequency mid-night: the keeper must follow the squitters' announced frequencies, not keep windowing a dead one.  Test in Task 5.
2. An RTP wrap or a radiod restart mid-run: windows must re-anchor, never carry a stale anchor.  Test in Task 5.
3. A window holding two bursts, or none: the analyser must report "no burst" or flag the ambiguity, never time noise.  Test in Task 4.
4. A disk filling on the station: the keeper stops saving, logs once, and decoding carries on.  Test in Task 5.
5. Frame index near UTC midnight (the index restarts daily): nominal slot start must land on the right day.  Test in Task 3.

---

### Task 1: Label check, the gate

Proves whether an HFDL channel's UTC label agrees with hf-timestd's labels on the same radiod.  Method: subscribe to one of hf-timestd's own channels (same frequency, preset, rate and encoding returns hf-timestd's SSRC), create a probe channel on the same frequency at an HFDL band's rate, label both with `acquire_anchor_utc`, resample the probe to the reference rate, and cross-correlate the two sample streams.  The same RF passes through both, so the lag between labelled streams equals the label difference plus the filter-delay difference, with no ionosphere in it.  Repeat after deleting and recreating the probe channel, to expose a per-channel origin.

**Files:**
- Create: `src/hfdl_recorder/timing/__init__.py`, `src/hfdl_recorder/timing/label_check.py`
- Modify: `src/hfdl_recorder/cli.py` (new subcommand `label-check`), `pyproject.toml` (add `"hamsci-dsp>=0.8.0"`; `[tool.uv.sources] hamsci-dsp = { path = "../hamsci-dsp", editable = true }`, as psk-recorder does)
- Test: `tests/test_label_check.py`

**Interfaces:**
- Produces: `labelled_lag_seconds(ref: np.ndarray, ref_fs: int, ref_t0_utc: float, probe: np.ndarray, probe_fs: int, probe_t0_utc: float) -> tuple[float, float]` returning (lag in seconds, peak correlation 0..1).  Positive lag: the probe's label runs late.
- Produces: CLI `hfdl-recorder label-check --config ... --radiod-id ... --ref-freq-hz F --band HFDL13 --seconds 30 --repeats 5 --json`, printing one JSON object per repeat: `{"band", "probe_fs", "ref_fs", "lag_us", "corr", "anchor_source", "t_level_active"}`.
- Produces: `label_corrections.toml` written next to the spool (`[<band>] lag_us = ...`), read by Task 4.

- [ ] **Step 1: Write the failing tests** for `labelled_lag_seconds`, with the same noise signal sampled at 12000 and 100000 S/s:
  - `test_zero_lag_when_labels_agree`: |lag| < 2e-6, corr > 0.9.
  - `test_reports_known_label_offset`: probe_t0_utc shifted by +3.7e-3 gives lag 3.7e-3 ± 2e-6.
  - `test_fractional_sample_offset`: a 0.3-sample delay at 12000 S/s comes back within 2e-6.
  - `test_uncorrelated_streams_report_low_corr`: corr < 0.2.
- [ ] **Step 2: Run** `uv run pytest tests/test_label_check.py -v`.  Expected: FAIL (module missing).
- [ ] **Step 3: Implement `labelled_lag_seconds`.**  Resample the probe to `ref_fs` (`scipy` is not a dependency: use FFT-domain resampling in numpy), align by labels, cross-correlate by FFT, refine the peak with a parabola on the magnitude.
- [ ] **Step 4: Run the tests.**  Expected: PASS.  Mutation: flip the sign convention in the implementation; `test_reports_known_label_offset` must fail.
- [ ] **Step 5: Implement the `label-check` subcommand.**  Find the reference channel through `ka9q.discover_channels(status)`, choosing an hf-timestd channel on `--ref-freq-hz`.  Subscribe through `MultiStream.add_channel` with that channel's exact preset, rate and encoding.  Delete the probe channel between repeats (`RadiodControl` channel removal, or a LIFETIME of 1 frame and wait for it to expire), so each repeat gets a fresh channel and a fresh RTP origin.
- [ ] **Step 6: Commit.**  `git commit -m "timing: label-check measures HFDL channel labels against hf-timestd's"`
- [ ] **Step 7: Run at ND, UACNJ and WB6CXC-7** (bus announcement first).  For each band to be used, 5 repeats of 30 s, against hf-timestd's 10 MHz (or 5 or 15) channel.  Record the results in `docs/superpowers/plans/2026-11-02-label-check-results.md`.
- [ ] **Step 8: Gate.**  Pass when, per station and band, the repeats' lags spread less than 0.1 ms (a constant lag is fine; Task 4 subtracts it).  Fail: stop the plan, report the numbers to Michael, and draft the upstream note about the RTP origin regression for him (he chose to wait on it until now).

### Task 2: Waveform constants and synthetic squitters

**Files:**
- Create: `src/hfdl_recorder/timing/waveform.py`
- Test: `tests/test_waveform.py`

**Interfaces:**
- Produces: `SYMBOL_RATE = 1800`, `SUBCARRIER_HZ = 1440.0`, `PREKEY_LEN = 448`, `A_LEN = 127`, `M1_LEN = 127`, `T_LEN = 15`, `M_SHIFTS = (72, 82, 113, 123, 61, 103, 93, 9)`.
- Produces: `a_symbols() -> np.ndarray` (±1, length 127); `m1_symbols(shift_index: int) -> np.ndarray` (±1, 127); `timing_template() -> np.ndarray` = 448 pre-key symbols of +1, then A, then A again, as ±1 symbols (length 702).
- Produces: `modulate(symbols: np.ndarray, fs: float, rolloff: float = 0.35) -> np.ndarray` complex baseband at the subcarrier, our own raised-cosine pulse; `synth_squitter(fs: float, delay_s: float, *, snr_db: float, doppler_hz: float = 0.0, paths: list[tuple[float, float]] = (), m1_shift: int = 0, rng) -> np.ndarray` for tests (a window of length `(0.03 + 32/13) * fs`, burst starting at `0.03 + delay_s`).

- [ ] **Step 1: Write the failing tests.**
  - `test_a_sequence_matches_dumphfdl_octets`: the bits of `a_symbols()` equal the 127 bits from dumphfdl's `A_octets` (`0b01011011, 0b10111100, 0b01110100, 0b01010111, 0b00000011, 0b11011001, 0b10001001, 0b00111001, 0b11110010, 0b00001000, 0b11010101, 0b00110110, 0b10010100, 0b00101100, 0b00110010, 0b11111110`, first 127 bits, MSB first), as an oracle in the test only.
  - `test_m1_shifts_are_rotations`: `m1_symbols(k)` equals the base sequence rotated by `M_SHIFTS[k]`; all 8 differ.
  - `test_a_autocorrelation_is_sharp`: periodic autocorrelation sidelobes below 0.15 of the peak.
  - `test_synth_burst_lands_where_asked`: energy onset of `synth_squitter(6000, 0.0123, snr_db=40)` within one sample of 0.0423 s.
- [ ] **Step 2: Run** `uv run pytest tests/test_waveform.py -v`.  Expected: FAIL.
- [ ] **Step 3: Implement** `waveform.py`.  Put the base M1 bit list (127 values) and A bits in the module as data, with a comment citing ARINC 635 and noting that dumphfdl 1.7 `src/hfdl.c` served as cross-check.  The pre-key carries constant phase.
- [ ] **Step 4: Run tests.**  Expected: PASS.
- [ ] **Step 5: Commit.**  `git commit -m "timing: HFDL waveform constants and synthetic squitters"`

### Task 3: Frame grid

**Files:**
- Create: `src/hfdl_recorder/timing/frame_grid.py`
- Test: `tests/test_frame_grid.py`

**Interfaces:**
- Produces: `FRAME_S = 32.0`, `SLOT_S = 32.0 / 13`.
- Produces: `slot_start_utc(frame_index: int, frame_offset: int, near_utc: float) -> float`.  The frame index counts 32 s frames from 00:00 UTC (2700 a day); `near_utc` picks the day (the one whose slot start lies nearest).
- Produces: `next_slot_starts(frame_offset: int, after_utc: float, until_utc: float) -> list[float]`: every nominal start of slot `frame_offset` in that span.

- [ ] **Step 1: Write the failing tests.**
  - `test_riverhead_on_nd`: frame 2280, offset 5, near 2026-11-01T20:16:12Z gives 20:16:00 + 5 × 32/13 s exactly.
  - `test_midnight_picks_the_right_day`: frame 2699 offset 12, near 00:00:20Z on 11-02, gives 11-01 23:59:28 + 12 × 32/13 s (Review Focus 5).
  - `test_next_slot_starts_spacing`: consecutive starts differ by exactly 32.0 s.
- [ ] **Step 2: Run.**  Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.**  Expected: PASS.
- [ ] **Step 5: Commit.**  `git commit -m "timing: HFDL frame grid on UTC"`

### Task 4: Squitter timing core

**Files:**
- Create: `src/hfdl_recorder/timing/analyzer.py`
- Test: `tests/test_analyzer.py`

**Interfaces:**
- Consumes: `timing_template`, `modulate`, `m1_symbols`, `synth_squitter` (Task 2); `slot_start_utc` (Task 3).
- Produces: dataclass `Arrival(found: bool, arrival_utc: float, delay_s: float, snr_db: float, peak_width_s: float, second_peak_ratio: float, second_peak_delay_s: float, doppler_hz: float, m1_shift: int | None, reason: str)`.
- Produces: `time_squitter(x: np.ndarray, fs: float, t0_utc: float, nominal_utc: float, label_lag_s: float = 0.0) -> Arrival`.  `delay_s = arrival_utc - nominal_utc`, after subtracting `label_lag_s` from Task 1's table.

- [ ] **Step 1: Write the failing tests** (all on `synth_squitter`, fs = 6000):
  - `test_whole_and_fractional_delays`: delays 0.0050, 0.01234, 0.0201 s at 20 dB SNR come back within 20 µs.
  - `test_shift_moves_answer_exactly` (mutation): prepend 37 zeros (and 0.4 sample by FFT shift); the delay moves by exactly that ± 2 µs.
  - `test_doppler_tolerated_and_measured`: doppler ±3 Hz; delay within 30 µs; `doppler_hz` within 0.3 Hz.
  - `test_two_path`: paths at +0 (1.0) and +2.5 ms (0.6); reports the stronger at delay ± 30 µs, and `second_peak_delay_s` ≈ +2.5 ms.
  - `test_no_burst_reports_not_found`: pure noise gives `found=False`, `reason="no_peak"` (Review Focus 3).
  - `test_two_bursts_flagged`: two equal bursts 0.4 s apart give `found=False`, `reason="ambiguous"` (Review Focus 3).
  - `test_broken_template_fails` (mutation): monkeypatch `timing_template` to flip one A symbol in ten; `test_whole_and_fractional_delays` tolerance fails.
- [ ] **Step 2: Run.**  Expected: FAIL.
- [ ] **Step 3: Implement `time_squitter`.**  Coarse Doppler from the pre-key: FFT peak over the first 0.25 s after the energy onset, interpolated.  Remove it, then correlate the window against `modulate(timing_template())` by FFT; refine the peak by parabola on the magnitude; SNR from peak over the median correlation.  Try the 8 M1 shifts after the A pair only to report `m1_shift`; timing uses pre-key + A + A.  Thresholds: found when SNR ≥ 10 dB; ambiguous when a second peak more than 50 ms away reaches 0.8 of the first.
- [ ] **Step 4: Run.**  Expected: PASS.
- [ ] **Step 5: Commit.**  `git commit -m "timing: time a squitter against the HFDL preamble"`

### Task 5: Window keeper in the sample path

**Files:**
- Create: `src/hfdl_recorder/timing/window_keeper.py`
- Modify: `src/hfdl_recorder/core/band_pipeline.py` (`_on_samples` hands `(samples, quality)` to an optional keeper after the dumphfdl write), `src/hfdl_recorder/core/ch_tailer.py` (optional `on_squitter: Callable[[dict], None]` called with each parsed SPDU frame), `src/hfdl_recorder/core/daemon.py` (builds one keeper per band when enabled), `src/hfdl_recorder/config.py` and `config/hfdl-recorder-config.toml.template` (new section).
- Test: `tests/test_window_keeper.py`, additions to `tests/test_band_pipeline.py`, `tests/test_config.py`.

**Interfaces:**
- Consumes: `next_slot_starts`, `SLOT_S` (Task 3).
- Config: `[timing_squitters] enabled = false`, `ground_stations = ["Riverhead, New York", "San Francisco, California", "Molokai, Hawaii"]`, `window_dir = "/var/lib/hfdl-recorder/squitters"`, `retention_days = 5`, `min_free_gb = 5`.
- Produces: `WindowKeeper(band: HfdlBand, fs: int, out_dir: Path, ground_stations: list[str], anchor_fn: Callable[[int], AnchorUTC], counter_map_fn: Callable[[], CounterMap | None], max_queue: int = 8)` with `feed(samples: np.ndarray, rtp_start: int) -> None` (non-blocking), `note_squitter(spdu_frame: dict) -> None`, `stats() -> dict` (`saved`, `dropped_queue`, `dropped_disk`, `reanchors`).
- Produces: window files `<out_dir>/<YYYYMMDD>/<gs_id>-<freq_khz>-<frame_index>.npz` holding `x` (complex64, 6000 S/s), and JSON metadata `t0_utc`, `nominal_utc`, `fs`, `freq_khz`, `gs_id`, `gs_name`, `frame_index`, `frame_offset`, `band`, `anchor_source`, `offset_ns`, `t_level_active`, `utc_sync`; and the counter label: `input_index0` (float, radiod input-sample index of the first saved sample), `input_samprate`, `counter_map` (`CounterMap.to_dict()`), `input_epoch_gps_s` (names the radiod run).  A window whose counter map is missing still saves, with `input_index0 = null`.

- [ ] **Step 1: Write the failing tests** (synthetic stream fed in 20 ms blocks with RTP timestamps; `anchor_fn` a fake):
  - `test_window_starts_30ms_before_nominal`: the saved window's `t0_utc` = nominal − 0.030 within one output sample; length = round((0.030 + 32/13) × 6000).
  - `test_only_live_chosen_stations`: squitters noted from a non-chosen station make no windows.
  - `test_follows_frequency_change` (Review Focus 1): after `note_squitter` announces a new frequency and the old one goes silent for 3 frames, windows switch.
  - `test_reanchors_on_rtp_jump` (Review Focus 2): an RTP discontinuity (restart, or wrap) re-anchors; `stats()["reanchors"] == 1`; no window spans the jump.
  - `test_feed_never_blocks`: with a writer stubbed to sleep 1 s, 100 `feed` calls finish in under 0.1 s and `dropped_queue` counts the rest.
  - `test_disk_low_stops_saving` (Review Focus 4): with free space below `min_free_gb`, nothing saves, one warning logs, `dropped_disk` counts.
  - In `test_band_pipeline.py`, `test_dumphfdl_bytes_unchanged_with_keeper`: the CF32 bytes sent to dumphfdl stay identical with the keeper on and off.
  - In `test_config.py`, `test_timing_squitters_off_by_default`.
- [ ] **Step 2: Run.**  Expected: FAIL.
- [ ] **Step 3: Implement.**  Ring buffer per band holding about 3 s of samples in RTP space; `SlotClock(cadence_sec=32.0, sample_rate=fs)` anchored through `anchor_fn` (the daemon builds it from `acquire_anchor_utc` with an `AuthorityReader`).  The counter label comes from `counter_map_fn`: the daemon keeps one `CounterMap` per band channel from that channel's own status (`RadiodControl.poll_status(ssrc)` → `CounterPoll`, as hf-timestd's `counter_map_probe.py` does: 5 polls at start, then one a minute, majority 3 of 5, roll forward, drop when stale or on a new radiod run), off the sample path.  Poll only the band's own live SSRC: on radiod a command to an unknown SSRC mints a channel.  The slot start at offset k sits at k × 32/13 s into the frame, generally between samples: keep the integer sample start and record the exact `t0_utc` of the first saved sample.  Mix to the channel (freq_khz × 1000 + 1440 − band centre), decimate to 6000 S/s with a short FIR, hand the result to a writer thread through a bounded queue.  Prune dated folders older than `retention_days` once an hour from the writer thread.
- [ ] **Step 4: Run** the three test files.  Expected: PASS.  Run the whole suite: `uv run pytest tests/ -q --override-ini addopts=` (all pass).
- [ ] **Step 5: Commit.**  `git commit -m "timing: keep labelled windows around expected squitters"`

### Task 6: Batch analysis and the sink table

**Files:**
- Create: `src/hfdl_recorder/timing/batch.py`, `systemd/hfdl-recorder-squitters@.service`, `systemd/hfdl-recorder-squitters@.timer`
- Modify: `src/hfdl_recorder/cli.py` (`squitter-analyze`), `deploy.toml` (the new units, installed disabled)
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: `time_squitter`, `Arrival` (Task 4); window files (Task 5); hf-timestd's T3 shadow minute rows (`/var/lib/timestd/t3_shadow/minutes-<YYYYMMDD>.jsonl`, fields `t6_anchor_input_index`, `t6_anchor_utc_ns`, `t3.counter_epoch_id`), read-only.
- Produces: CLI `hfdl-recorder squitter-analyze --dir DIR [--since UTC] [--csv OUT]`, which skips windows already analysed (a `.done` marker beside each).
- Produces: sink rows, table `squitter_timing`, mode `hfdl`, `local=True`, one per window: the window metadata plus every `Arrival` field, plus `arrival_input_index` (the arrival in input-sample index) and `delay_counter_s` (arrival through hf-timestd's index registration nearest in time, minus nominal; null when no registration of the same radiod run lies within 120 s), and `label_diff_s` (status-pair label minus counter label at the window start: the status-path noise).

- [ ] **Step 1: Write the failing tests.**
  - `test_rows_from_synthetic_windows`: three synthetic window files with known delays give three rows whose `delay_s` match ± 20 µs, through a fake writer factory.
  - `test_counter_delay_from_registration`: a synthetic minute row (index, UTC) and a window with known `input_index0` give `delay_counter_s` within 1 µs of the known delay; a registration from another radiod run, or older than 120 s, gives null.
  - `test_label_diff_recorded`: a window whose status-pair label runs 2.3 ms late against its counter label gives `label_diff_s` = +0.0023.
  - `test_utc_sync_false_marked`: the row carries `excluded = "utc_sync_false"`.
  - `test_second_run_skips_done`: a rerun inserts nothing.
  - `test_writer_is_local`: the factory gets `local=True`.
- [ ] **Step 2: Run.**  Expected: FAIL.
- [ ] **Step 3: Implement.**  The service runs `nice -n 19` with `IOSchedulingClass=idle` and `CPUSchedulingPolicy=idle`; the timer fires every 15 minutes.
- [ ] **Step 4: Run.**  Expected: PASS.
- [ ] **Step 5: Commit.**  `git commit -m "timing: batch squitter analysis into a local sink table"`

### Task 7: Confirm the WWV inputs for the comparison

Before the test week: find out whether hf-timestd keeps, at each station, a week of per-minute WWV, WWVH and WWVB arrivals and the reference tier's readings, apart from the fused answer.

**Files:**
- Create: `docs/superpowers/plans/2026-11-02-wwv-inputs.md` (findings)

- [ ] **Step 1:** Query the graph: `graphify query "where does hf-timestd store per-minute tick arrivals and the authority history" --graph repos/graphify-out/graph.json --budget 6000` (run from `/home/mjh/hamsci`).
- [ ] **Step 2:** On ND only, read-only: list tables and row date ranges in `/var/lib/timestd/authority_history.db` and the T3 measurement store the graph names.  No heavy queries on station VMs.
- [ ] **Step 3:** Write the findings.  If a week of arrivals and reference readings already persists, record the tables and the pull method.  If not, write the smallest hf-timestd export task, raise it with Michael, and hold the test week until it runs.
- [ ] **Step 4: Commit** the findings document.

### Task 8: Deploy to the three test stations

**Files:**
- Create: `/home/mjh/hamsci/tasks/hfdl-timing/pull.sh` (devbox, not in the repo): rsync each station's `squitters/`, `squitter_timing` rows and `/var/lib/timestd/t3_shadow/` into `/home/mjh/hamsci/tasks/hfdl-timing/<station>/`; a user timer runs it daily at 12:30Z.

- [ ] **Step 1:** Read the claude-bus; announce: hfdl-recorder install at UACNJ and WB6CXC-7, the timing feature at ND, UACNJ and WB6CXC-7, deployed by git.
- [ ] **Step 2:** Choose bands per station from ND's decodes: the bands that carried Riverhead, San Francisco and Molokai squitters between 00:00 and 12:00Z on 11-02.  Two or three bands only.
- [ ] **Step 3:** Install hfdl-recorder at UACNJ and WB6CXC-7 through smd, with those bands, airframes.io off.  Confirm dumphfdl decodes squitters within an hour.
- [ ] **Step 4:** Move all three stations to the feature commit by git; set `[timing_squitters] enabled = true`; restart only `hfdl-recorder@`.  Enable the squitters timer.
- [ ] **Step 5:** After one hour: windows appear; `stats()` in the journal shows no drops; dumphfdl decode counts match the hour before.  Measure CPU of hfdl-recorder on the Proxmox host before and after; record both.
- [ ] **Step 6:** Install the pull script and its timer; check that the first pull lands.

### Task 9: Offline evaluation

**Files:**
- Create: `tools/squitter_eval.py` (repo; runs on the devbox only), `tests/test_squitter_eval.py`

**Interfaces:**
- Consumes: pulled `squitter_timing` rows and window metadata; Task 7's WWV inputs and reference readings.  Scores on `delay_counter_s`; reports `delay_s` (status-pair label) beside it, and the distribution of `label_diff_s` per station.
- Produces: `squitter_eval.py report --data DIR --train-nights N --out report.md` with four sections: checks 1–4 as the spec defines them, plus the same-frame multi-frequency consistency (one ground station's squitter on two bands in the same frame must agree within the path dispersion, about 0.3 ms; a larger gap points at the labels).

- [ ] **Step 1: Write the failing tests** on synthetic rows: a station offset of 1.234 ms with 0.2 ms scatter comes back within 0.05 ms (check 1); training and scoring nights never overlap (check 4); the reference column never enters the reconstruction (assert by poisoning the reference with NaN and getting the same reconstruction).
- [ ] **Step 2: Run.**  Expected: FAIL.
- [ ] **Step 3: Implement.**  Reconstruction per minute: weighted least squares for the clock error plus one nuisance per (ground station, receiving station, band) learned on training nights; path delay from great-circle geometry with a one-hop F-layer at 300 km as the prior, its residue absorbed by the nuisance.  Weighted by each source's demonstrated scatter.
- [ ] **Step 4: Run.**  Expected: PASS.
- [ ] **Step 5: Commit.**  `git commit -m "tools: offline squitter timing evaluation"`
- [ ] **Step 6:** After seven nights of data, run the report; send it to Michael with the verdict against the spec's success criterion.
