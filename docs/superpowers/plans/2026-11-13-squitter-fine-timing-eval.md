# Squitter Fine Timing, Offline Evaluation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the multi-path squitter fit, prove it on synthetic bursts, then score it against analyser version 2 on the windows already pulled to the devbox, and write the report that section 4 of the spec asks for.

**Architecture:** Two new modules in the hfdl-recorder package hold the fit: `timing/known_symbols.py` describes every known symbol of a single-slot burst and renders it, and `timing/multipath.py` fits K paths to the complex samples. The fit takes its seed from today's analyser, which gains one keyword and no change in output. Four devbox tools in `tools/` drive the evaluation: a synthetic grid, a pulse check, a streaming replay that writes a v2 row and a fit row per window, and a report that reruns the existing keying scripts and `squitter_eval.py` on each arm. Nothing runs on a station.

**Tech Stack:** Python 3.11, numpy only (no scipy: the station venv has none, and hfdl-analyzer-3 will import this code), pytest via `uv run`, multiprocessing for the replay.

**Spec:** `docs/superpowers/specs/2026-11-13-squitter-fine-timing-design.md` (accepted by Michael 2026-11-13 with all nine recommendations).

**Where the fit lives, and why:** in hfdl-recorder (`src/hfdl_recorder/timing/`), not hamsci-dsp. The fit renders HFDL's own sequences from `timing/waveform.py`, takes its seed from `timing/analyzer.py`'s `Arrival`, and the live analyser (`timing/batch.py`) already imports its timing from inside the package. hamsci-dsp holds only waveform-free helpers (`dsp.py`: correlation, noise floors, Doppler from phase). No shared multi-path fitter exists to extend: CODAR's CLEAN never left `tasks/codar-mf/clean.py`, and it works on incoherent power, not on complex samples. Putting the fit in hamsci-dsp would also tie hfdl-analyzer-3 to a hamsci-dsp release on every station.

## Global Constraints

- Devbox only. No task touches a station, with one exception: Task 8 reads one config value from KFS-MAIN, read-only, after reading the claude-bus (`/home/mjh/hamsci/CLAUDE.md`).
- Never write inside `/home/mjh/hamsci/tasks/hfdl-timing/<station>/squitters/`: no `.done` markers, no temporary files. Never delete anything. Pruning follows this plan, done by Michael, from the list Task 12 writes.
- Stream the windows. Each worker loads one `.npz`, keeps it in memory for both analysers, and drops it. Never copy a window file. All outputs go under `/home/mjh/hamsci/tasks/hfdl-timing/fit-eval/`.
- Disk: the devbox root had 3.7 GB free on 2026-11-13. Every long-running tool refuses to start below 1.5 GB free and stops cleanly, resumable, below 0.75 GB.
- numpy only in `src/`. No new dependency in `pyproject.toml`.
- dumphfdl carries GPLv3; hfdl-recorder is MIT. Take the probe sequence as a protocol fact, cite where it comes from, write our own code, and copy no dumphfdl code (as the 11-02 plan required for A, M1, M2, T).
- Analyser version 2's outputs stay exactly as they are with default arguments. `ANALYZER_VERSION` stays 2. hfdl-analyzer-3 belongs to a later plan.
- Spec decisions, as accepted: the row's delay names the earliest significant path, with the strongest in extra columns (1); one row per squitter (2); the replay uses the fitted roll-off per (receiving station, ground station) from Task 7, since the rig measurement comes later (3, fallback b); known symbols only: pre-key (through the seed's Doppler), preamble, probes (4); the keying prediction plays no part in this plan (8 permits it only to choose the reference path; deferred to the live plan); San Francisco gets reported as diagnosis only, never scored (9).
- Scoring stations: B4, AC0G-ND, K3LR, plus KFS-MAIN and W2NAF when their rows qualify (at least 1,000 found rows with `t_level_active == "T6"`, and T6 on at least half the found rows). WB6CXC-7 and UACNJ run as extra targets and appear in the report without a pass mark (spec section 4, item 3).
- Honesty of `delay_sigma`: the pass mark uses the synthetic truth and the split-burst check on real windows (Task 11), because the ionosphere's 0.5 to 0.8 ms swamps a covariance sigma of microseconds when scored against T6. The literal "against T6" figure appears as a diagnostic. Michael confirmed this reading 2026-11-13.
- Pass marks: the section 4 table. "No worse" means the fit's value exceeds version 2's by no more than 5 % of version 2's value. Pass marks compare the two arms on the paired window set (windows both arms timed), never against the spec's "Today" column; that column came from 11-03 to 11-08 windows, most of which no longer exist on the devbox.
- Tests: `cd /home/mjh/hamsci/repos/hfdl-recorder && uv run pytest <file> -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`.
- Long runs: `nice -n 19`, at most 4 worker processes (the devbox has 6 cores and 7 GB), and not across 04:45 UTC if avoidable (the nightly graphify rebuild).
- Git: work on main, never push. Every commit message ends with exactly:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01BUMH4psju42aywNgiJv98Z
  ```

- `tasks/hfdl-timing/` lives outside any git repo. Changes there get no commit; the task's verification step stands in for it.
- Report prose: short sentences, active voice with a named actor, the verb "to be" used sparingly, no project shorthand left unexplained.

## Review Focus

1. **A burst that runs past the window's end.** A delay over about 120 ms cuts the last probe blocks off. The fit must clip its mask to the window and still time the path from what remains. Test in Task 4.
2. **Zero-filled gaps.** A window with `n_gap_samples > 0` holds runs of exact zeros. The fit must leave them out of the mask, not fit a burst to them. Test in Task 4.
3. **A wrong pulse invents a close echo.** At 40 dB a pulse mismatch leaves structured residue beside the main path. The fit must not report it as a second path when the pulse matches, and the wrong-pulse mutation must break a test. Test in Task 5.
4. **A replay killed mid-run and restarted.** Rows written before the kill must not appear twice, and no window may vanish from the manifest. Test in Task 9.
5. **Window names repeat daily.** The frame index restarts each UTC day, so `1-13276-0.npz` exists in every day folder. Every key over windows needs the day. `squitter_eval.py` gets this wrong today (Task 1); the replay's keys must not (Task 9).

---

### Task 1: Key windows by day in `squitter_eval.py`

`_latest_analysis` keys rows by `(radiod_id, window_file)`. The name repeats every UTC day, so the evaluator keeps one day's row per name and counts the rest as `superseded_analysis`: 20,490 of B4's 28,611 rows in `report-train4.md`. Check 3, one of the section 4 measures, ran on that remnant.

**Files:**
- Modify: `tools/squitter_eval.py` (`_latest_analysis`, about lines 489-505)
- Test: `tests/test_squitter_eval.py`

**Interfaces:**
- Produces: `_latest_analysis(rows, counts) -> list` keyed by `(radiod_id, window_file, _utc_day(nominal_utc))`; a row without `nominal_utc` keys with day `None`.

- [ ] **Step 1: Write the failing test** `test_same_window_name_on_two_days_both_kept`: two rows, same `radiod_id` and `window_file`, `analyzer_version` 2, `nominal_utc` 86,400 s apart. Assert `len(_latest_analysis(rows, counts)) == 2` and `counts["superseded_analysis"] == 0`. Add `test_same_window_same_day_newest_wins`: same name, same day, versions 1 and 2; one row survives, version 2, and `superseded_analysis == 1`.
- [ ] **Step 2: Run** `uv run pytest tests/test_squitter_eval.py -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: the first new test FAILS (1 row kept).
- [ ] **Step 3: Add the day to the key** in `_latest_analysis`, using the existing `_utc_day`.
- [ ] **Step 4: Run** the whole file. Expected: all PASS.
- [ ] **Step 5: Commit.** `git commit -m "squitter_eval: key windows by UTC day; names repeat daily"`

### Task 2: Seams in the analyser and the batch

Two behaviour-preserving changes the replay needs. The analyser learns to time an ambiguous window instead of refusing it, so the fit can try to resolve the near echo. The batch splits loading a window from building its row, so the replay reads each file once and maps both arms through the same counter-label code.

**Files:**
- Modify: `src/hfdl_recorder/timing/analyzer.py` (`time_squitter`; new `v2_view`)
- Modify: `src/hfdl_recorder/timing/batch.py` (`_row_for`, about lines 195-248)
- Test: `tests/test_analyzer.py`, `tests/test_batch.py`

**Interfaces:**
- Produces: `time_squitter(x, fs, t0_utc, nominal_utc, label_lag_s=0.0, *, refuse_ambiguous: bool = True) -> Arrival`. With `refuse_ambiguous=False`, a window that would come back `ambiguous` instead comes back found and timed through step 7, with `ambiguous_far` in `flags` when the far test fired (`second_far >= AMBIGUOUS_RATIO * peak`) and `ambiguous_near` when the near one did (`ratio2 >= AMBIGUOUS_RATIO`); both may appear.
- Produces: `v2_view(a: Arrival) -> Arrival`: `_not_found("ambiguous", a.snr_db)` when `flags` holds either ambiguous flag, else `a` unchanged.
- Produces: `load_window(path: Path) -> tuple[np.ndarray, dict]` (samples as complex128, parsed meta) and `window_row(path: Path, meta: dict, arr: Arrival, regs: Registrations) -> tuple[dict, bool]` in `batch.py`. `_row_for(path, regs)` becomes `load_window`, then `time_squitter`, then `window_row`.

- [ ] **Step 1: Write the failing tests.**
  - `test_v2_view_matches_default` (parametrised over the synthetic windows of `test_equal_echo_within_50ms_is_ambiguous`, `test_two_bursts_flagged`, `test_two_path` and one clean burst): `v2_view(time_squitter(x, ..., refuse_ambiguous=False))` equals `time_squitter(x, ...)` field by field, NaN equal to NaN.
  - `test_near_ambiguity_flagged_not_refused`: the 2.5 ms, 0.9 echo window comes back `found` with `"ambiguous_near" in flags` and no `ambiguous_far`.
  - `test_far_ambiguity_flagged_not_refused`: the two-bursts window carries `ambiguous_far`.
  - `test_window_row_matches_row_for` in `tests/test_batch.py`: for a `_window(...)` with a `_registration(...)`, `window_row(p, meta, time_squitter(x, ...), regs)` equals `_row_for(p, regs)`.
- [ ] **Step 2: Run** both test files. Expected: the new tests FAIL (unknown keyword, missing names).
- [ ] **Step 3: Implement** the keyword, the two flags and `v2_view` in `analyzer.py`; split `_row_for` into `load_window` and `window_row` in `batch.py`.
- [ ] **Step 4: Run** `uv run pytest tests/test_analyzer.py tests/test_batch.py -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: all PASS, including every test that existed before.
- [ ] **Step 5: Commit.** `git commit -m "analyzer: refuse_ambiguous keyword and v2_view; batch: load_window/window_row seam"`

### Task 3: The known symbols of a single-slot burst

**Files:**
- Create: `src/hfdl_recorder/timing/known_symbols.py`
- Test: `tests/test_known_symbols.py`

**Interfaces:**
- Consumes: `waveform.a_symbols`, `waveform.m1_symbols`, `waveform._T_BITS`, `waveform.SYMBOL_RATE`, `waveform.SUBCARRIER_HZ`, `waveform.PREKEY_LEN`, `waveform.REAL_PREKEY_LEN`, `waveform.SLOT_S`, `waveform._LEAD_S`.
- Produces: `PREAMBLE_LEN = 531`, `DATA_LEN = 30`, `PROBE_LEN = 15`, `N_BLOCKS = 72` (single slot).
- Produces: `@dataclass(frozen=True) class Pulse: family: str = "rc"; rolloff: float = 0.35` (`family` in `{"rc", "rrc"}`) and `pulse_shape(x: np.ndarray, pulse: Pulse) -> np.ndarray` (value at x symbol periods from the centre).
- Produces: `known_stream(m1_shift: int | None, segments: str = "all") -> tuple[np.ndarray, np.ndarray]`: complex symbols indexed from A1 (index 0 = first A symbol) through the last probe, and a boolean "known" array of the same length. Unknown symbols hold 0. `segments` takes `"all"`, `"preamble"` (A1 through the ninth T block) or `"probes"` (the 72 probe blocks only). `m1_shift=None` marks M1 and M2 unknown.
- Produces: `known_runs(known: np.ndarray) -> list[tuple[int, int]]`: half-open symbol runs of known symbols.
- Produces: `render(symbols: np.ndarray, fs: float, n: int, a1_s: float, pulse: Pulse) -> np.ndarray`: n samples at fs, symbol k's pulse centred at `a1_s + (k + 0.5) / 1800`, on the 1440 Hz subcarrier attached to the delayed signal (the convention of `waveform._render`), complex symbols allowed.
- Produces: `synth_burst(fs: float, delay_s: float, paths: Sequence[tuple[float, complex, float]], *, snr_db: float, rng, pulse: Pulse = Pulse(), m1_shift: int = 0, prekey_len: int = REAL_PREKEY_LEN, prekey_phase: float | None = None) -> np.ndarray`. Window length `round((0.03 + SLOT_S) * fs)`; path k = `(extra_delay_s, amplitude, doppler_hz)`, its A1 at `0.03 + delay_s + extra_delay_s + 448/1800`. Known symbols from `known_stream(m1_shift)`, data symbols random from `rng` in the constellation Step 1 records, pre-key tone of `prekey_len` symbols before each path's A1 at `prekey_phase` (uniform from `rng` when None). Noise as `waveform.synth_squitter` sets it: complex white, at `snr_db` against the mean power of one unit-amplitude path.

- [ ] **Step 1: Pin the probe symbols from the protocol source.** Clone `https://github.com/szpajder/dumphfdl` into `~/.cache/dumphfdl-src` and check out tag `v1.7.0`. Read `src/hfdl.c` (and any scrambler file it includes) only to learn: the probe block's symbol values or the generator that makes them, whether probes or data change with the M1 shift, the data constellation per M1 shift, and the T blocks' values in the preamble. Record each fact with file and line numbers in the module docstring. Copy no code.
- [ ] **Step 2: Write the failing tests.**
  - `test_layout`: `known_stream(0)[0]` has length `531 + 72 * 45`; `known_runs` gives one run `(0, 531)` then 72 runs `(531 + 45 b + 30, 531 + 45 b + 45)`; `segments="preamble"` gives only the first; `segments="probes"` only the 72; `m1_shift=None` removes `(254, 396)` from the first run.
  - `test_render_matches_waveform`: for real ±1 symbols and `Pulse("rc", 0.35)`, `render` equals `waveform.modulate` shifted to the same start within 1e-12.
  - `test_rc_pulse_is_nyquist`: `pulse_shape(k, Pulse("rc", r))` gives 1 at 0 and below 1e-12 at every other integer k in ±8, for r in 0.2, 0.35, 0.5.
  - `test_rrc_squared_is_rc`: the discrete convolution of the RRC pulse with itself, sampled at 32 per symbol, matches the RC pulse of the same roll-off within 1e-3 of its peak.
  - `test_synth_path_lands_where_asked`: a noiseless one-path `synth_burst` correlated against `render(known)` peaks at the asked A1 within 0.01 sample.
  - `test_real_windows_carry_the_probes` (fixtures `4-8912-2558.npz`, `4-8912-2566.npz`): time each with `time_squitter`, rotate out its Doppler, and for each of the 72 probe blocks take `|<x, s_b>| / sqrt(<x, x><s_b, s_b>)` over the block's samples with `s_b` rendered from `known_stream` at the analyser's A1. The mean over blocks with the true probes must reach at least 3 times the median of the same statistic over 20 random wrong probe sequences (rng seed 0). The same test runs for the nine preamble T blocks.
- [ ] **Step 3: Run** `uv run pytest tests/test_known_symbols.py -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: FAIL, module missing.
- [ ] **Step 4: Implement** the module.
- [ ] **Step 5: Run** the tests. Expected: PASS. If `test_real_windows_carry_the_probes` fails while the rest pass, stop. Mark the test `xfail` with the measured ratio, make `"preamble"` the default for `segments` everywhere downstream, and report it to Michael: the probes cannot then help the fit until someone explains the gap.
- [ ] **Step 6: Mutation.** Flip one bit of the probe source in the implementation; `test_real_windows_carry_the_probes` must fail. Restore it.
- [ ] **Step 7: Commit.** `git commit -m "timing: known symbols of a single-slot HFDL burst, with probes"`

### Task 4: One-path fit on the complex samples

**Files:**
- Create: `src/hfdl_recorder/timing/multipath.py`
- Test: `tests/test_multipath.py`

**Interfaces:**
- Consumes: Task 3's `known_stream`, `known_runs`, `render`, `Pulse`, `synth_burst`; Task 2's `time_squitter(..., refuse_ambiguous=False)`; `analyzer._noise_sigma`; `waveform.PREKEY_LEN`, `SYMBOL_RATE`.
- Produces:

  ```python
  @dataclass
  class PathFit:
      a1_s: float            # A1, seconds after x[0]
      delay_s: float         # arrival (A1 - 448 symbols) minus nominal, Arrival's convention
      arrival_offset_s: float  # arrival minus t0_utc, formed in-window
      doppler_hz: float
      amp: complex
      snr_db: float          # |amp|^2 * mean burst power / noise_sigma^2, in dB (synth's input-SNR convention)
      sigma_s: float         # 1-sigma delay from the fit covariance

  @dataclass
  class FitResult:
      ok: bool
      reason: str            # ok, no_seed, bad_input, too_few_samples, diverged
      paths: list[PathFit]   # sorted by delay, earliest first
      ref: int               # reference path index: 0, the earliest significant (decision 1)
      strongest: int
      noise_sigma: float     # per-sample complex RMS, analyzer._noise_sigma
      resid_ratio: float     # J / (n_mask * noise_sigma**2)
      n_mask: int
      pulse: Pulse
      segments: str
      cpu_s: float
  ```

- Produces: `fit_paths(x: np.ndarray, fs: float, t0_utc: float, nominal_utc: float, seed: Arrival, *, pulse: Pulse = Pulse(), segments: str = "all", max_paths: int = 4, joint_refit: bool = True) -> FitResult`. In this task the fit refines the seed path only; Task 5 adds the search for more and names the default `MAX_PATHS`.
- Produces constants: `GUARD_SYM = 3`, `MIN_MASK_SAMPLES = 600`, `LM_MAX_IT = 50`, `LM_TOL = 1e-9`.

The method:

- Seed: A1 at `seed.arrival_offset_s + PREKEY_LEN / SYMBOL_RATE` seconds into x, Doppler `seed.doppler_hz`, M1 `seed.m1_shift`. A seed with `found` false gives `reason="no_seed"`. A seed flagged `ambiguous_far` also gives `no_seed`.
- Template: render `known_stream(m1, segments)` once at A1 = P0 on an FFT grid of length N, the next power of two at least `len(x) + 2 * (pad + search span)`; cache it by `(fs, N, m1, segments, pulse)`. Path k's signal comes from the template's spectrum times a phase ramp for its A1 (exact for this band-limited signal), cropped to the window, times `exp(j 2 pi f_k (n - n_c) / fs)` with `n_c` the window centre (which decorrelates Doppler from phase).
- Mask: sample n counts when, for every path, its symbol position `(n / fs - a1_k) * 1800` lies inside a known run at least `GUARD_SYM` symbols from either end; when it lies inside the window; and when it lies outside every run of 4 or more exact zeros in x. The pre-key never enters the mask. The mask stays fixed during one refinement and gets rebuilt between refinements. Fewer than `MIN_MASK_SAMPLES` gives `too_few_samples`.
- Refinement: Levenberg-Marquardt over every path's `(a1, f, Re a, Im a)`, analytic Jacobian (the delay derivative from the template spectrum times `-j 2 pi nu`, the Doppler one from `j 2 pi (n - n_c) / fs`), amplitudes started by linear least squares, damping started at 1e-3 and moved by factors of 10, stopping at `LM_TOL` relative change in J or `LM_MAX_IT` iterations. A path whose A1 moves more than 2 symbols from its start, or a non-finite J, gives `diverged`.
- Sigma: `sigma_eff^2 = J / (n_mask - 2K)`; covariance `sigma_eff^2 * inv(Jr^T Jr)` with `Jr` the stacked real Jacobian; `sigma_s` per path from its A1 entry.
- Rows: `delay_s` and `arrival_offset_s` follow `time_squitter`'s formulas exactly (small differences first, never an epoch subtraction).

- [ ] **Step 1: Write the failing tests** (FS = 6000, NOMINAL and T0 as in `test_analyzer.py`, seed from `time_squitter(x, FS, T0, NOMINAL, refuse_ambiguous=False)`):
  - `test_single_path_delay`: delays 0.0050, 0.01234, 0.0201 s at 20 dB; `abs(r.paths[r.ref].delay_s - d) < 5e-6`.
  - `test_shift_moves_answer_exactly`: prepend 37 zeros and delay by 0.4 sample (`_frac_delay` from `test_analyzer.py`); the reference delay moves by 37.4 / FS within 1e-6.
  - `test_precision_no_worse_than_v2`: 10 seeds at 10 dB; rms of the fit's error at most 1.2 times the rms of `time_squitter`'s on the same windows.
  - `test_sigma_honest_single_path`: 40 seeds at 10 dB; `0.7 <= rms(err / sigma_s) <= 1.5`.
  - `test_doppler_measured`: Doppler 3.3 Hz at 20 dB; `abs(doppler_hz - 3.3) < 0.05`.
  - `test_snr_reported`: 20 dB input; path `snr_db` within 1 dB of 20.
  - `test_late_burst_clips_mask` (Review Focus 1): delay 0.150 s at 20 dB; `ok`, error under 10e-6, and `n_mask` smaller than for delay 0.010.
  - `test_zero_gap_left_out` (Review Focus 2): zero 600 samples in the middle of the probes; `ok`, error under 10e-6, and no masked sample falls inside the gap.
  - `test_no_seed`: a seed with `found=False` gives `ok=False, reason="no_seed"`; non-finite x gives `bad_input`; never an exception.
- [ ] **Step 2: Run** `uv run pytest tests/test_multipath.py -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: FAIL, module missing.
- [ ] **Step 3: Implement** `fit_paths` for one path.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "timing: one-path fit of the known symbols on complex samples"`

### Task 5: More paths, the joint refit and the stop rule

**Files:**
- Modify: `src/hfdl_recorder/timing/multipath.py`
- Test: `tests/test_multipath.py`

**Interfaces:**
- Produces constants: `MAX_PATHS = 4`, `SEARCH_EARLY_S = 0.020`, `SEARCH_LATE_S = 0.020`, `DOPPLER_SEARCH_HZ = 1.5`, `DOPPLER_STEP_HZ = 0.25`, `MIN_PATH_SEP_S = 0.25e-3`, `PATH_DJ_MIN = 12.0`.
- Produces: `refit_paths(x, fs, t0_utc, nominal_utc, start: FitResult, *, segments: str) -> FitResult`: the same path count and starting values as `start`, refined on another segment set, no search. Task 9's split-burst check uses it.

The search, after each refinement with K paths:

1. Residual `e = x - model`. Correlate `e` with the template under Doppler hypotheses `f_strongest +/- DOPPLER_SEARCH_HZ` in `DOPPLER_STEP_HZ` steps, over A1 from the strongest path's minus `SEARCH_EARLY_S` to plus `SEARCH_LATE_S`, clipped to the window; upsample by 8 and refine by parabola.
2. Skip any candidate within `MIN_PATH_SEP_S` of an existing path; take the strongest remaining.
3. Refine K+1 paths jointly (with `joint_refit=False`: only the new path's four parameters move).
4. Compare `J_K` and `J_{K+1}` on the mask of the K+1 paths. Keep the new path when `(J_K - J_{K+1}) / sigma_eff^2 >= PATH_DJ_MIN`, with `sigma_eff^2 = J_{K+1} / (n_mask - 2(K+1))`; otherwise keep the K-path result and stop. The threshold: under noise alone, the best of about 540 search cells (25 ms times 2.4 kHz in delay, 13 Doppler cells) reduces J by about ln(540 / 0.01) = 10.9 noise variances at 1 % false alarm; 12 leaves a margin.
5. Stop at `max_paths`. Sort paths by delay; `ref = 0`; `strongest` = index of the largest `|amp|`.

- [ ] **Step 1: Write the failing tests** (seed as in Task 4):
  - `test_two_paths_resolved`: extra 0.5 ms and 2.5 ms, amplitude 0.5, Doppler difference 0.5 Hz, 20 dB, 5 seeds each; `n_paths == 2` and each path's delay error under 20e-6.
  - `test_earliest_path_is_reference`: a 0.4-amplitude path 3 ms earlier than a unit path, 20 dB; `ref == 0` holds the earlier path, `strongest == 1`, both delays within 20e-6.
  - `test_far_late_path_found`: a 0.3-amplitude path 14 ms late, 20 dB; found, error under 20e-6.
  - `test_v2_ambiguous_near_window_fitted`: the window of `test_equal_echo_within_50ms_is_ambiguous`; `ok`, `n_paths == 2`, both delays within 20e-6.
  - `test_no_false_path`: one path, 20 seeds at each of 0, 20, 40 dB; at most 1 of 20 per SNR comes back with more than one path (Review Focus 3).
  - `test_refit_paths_same_count`: `refit_paths(..., segments="preamble")` on a two-path result returns two paths within 50e-6 of the full fit's.
  - Mutation tests, each asserting that the broken fit fails the bound the real fit meets:
    - `test_mutation_no_joint_refit_fails_overlap`: 0.5 ms apart, amplitude 0.7, 30 dB, `joint_refit=False`; at least one path's error exceeds 20e-6 in at least 3 of 5 seeds.
    - `test_mutation_wrong_pulse_fails`: burst made with `Pulse("rrc", 0.35)`, fitted with `Pulse("rc", 0.35)`, one path at 40 dB; either a second path appears or the error exceeds 5e-6, in at least 3 of 5 seeds. The same burst fitted with the right pulse keeps one path and an error under 5e-6.
- [ ] **Step 2: Run** the file. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the search, the stop rule, `refit_paths` and the `joint_refit` switch.
- [ ] **Step 4: Run** the file. Expected: PASS. Then the whole suite: `uv run pytest tests/ -q --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: no failures.
- [ ] **Step 5: Commit.** `git commit -m "timing: multi-path search, joint refit and noise-based stop rule"`

### Task 6: Synthetic grid against version 2

**Files:**
- Create: `tools/fit_synth.py`
- Test: `tests/test_fit_synth.py` (load the tool by path, as `tests/test_squitter_eval.py` loads `squitter_eval.py`)

**Interfaces:**
- Consumes: `synth_burst`, `fit_paths`, `time_squitter`, `v2_view`.
- Produces: CLI `fit_synth.py run --out DIR [--seeds 10] [--workers 4] [--quick]` writing `DIR/cases.csv` (one row per trial: case keys, seed, truth delays, fit paths, v2 delay, cpu) and `DIR/synth.md`.
- Produces: `summarise(trials: list[dict]) -> list[dict]`, one row per case: `count_correct` (fraction with `n_paths` equal to the truth), per-path `rms_err_s` and `bias_s` (fit paths matched to truth by nearest delay), `v2_err_earliest_s` and `v2_err_strongest_s` (rms), `honesty` (rms of err / sigma over matched paths), `cpu_s` median.
- Produces: `marks(summary) -> list[tuple[str, bool, str]]`, the four synthetic pass marks below.

The grid (spec section 4, item 1, plus a 0.4 ms spacing for the section 1 resolution target): one path at SNR -10, 0, 10, 20, 30, 40 dB; two paths at 0.3, 0.4, 0.5, 0.8, 1.3, 2.5 ms; second-path amplitude 0.1, 0.3, 0.5, 0.7, 0.9; Doppler difference 0, 0.5, 1.0 Hz; order `weaker_later` and `weaker_earlier`; same six SNRs; 10 seeds per case; base delay 0.0123 s; carrier phase of the second path uniform from the seed. `--quick` runs 2 cases with 2 seeds.

Synthetic pass marks (from spec section 1):
- S1: SNR 20 dB, spacing at least 0.5 ms, amplitude at least 0.3: `count_correct >= 0.9` and each path's rms error at most 20 µs.
- S2: SNR 30 dB, spacing 0.4 ms, amplitude at least 0.5: `count_correct >= 0.8`.
- S3: one path, every SNR: false-extra-path rate at most 5 %; at SNR 0 dB and above, rms error at most 1.2 times version 2's.
- S4: honesty between 0.7 and 1.5 for one path at SNR 0 dB and above, and for S1's cases.

- [ ] **Step 1: Write the failing tests.** `test_summarise_counts_and_errors` on hand-made trials (two cases, known errors) checks `count_correct`, `rms_err_s` and `honesty` to 1e-12. `test_marks_pass_and_fail` feeds one summary that meets S1 and one that misses it by one trial. `test_quick_run_writes_outputs(tmp_path)` runs `main(["run", "--out", tmp, "--quick", "--workers", "1"])` and finds both files.
- [ ] **Step 2: Run** `uv run pytest tests/test_fit_synth.py -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: FAIL.
- [ ] **Step 3: Implement** the tool.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Run the grid.** `cd /home/mjh/hamsci/repos/hfdl-recorder && nice -n 19 uv run python tools/fit_synth.py run --out /home/mjh/hamsci/tasks/hfdl-timing/fit-eval/synth --workers 4`. Expected: `synth.md` lists S1-S4. A failed mark does not stop the plan; it goes into the report and back to Michael before Task 10's real run.
- [ ] **Step 6: Commit.** `git commit -m "tools: synthetic grid for the multi-path fit against analyser v2"`

### Task 7: Pulse check on high-SNR real windows

**Files:**
- Create: `tools/pulse_check.py`
- Test: `tests/test_pulse_check.py`

**Interfaces:**
- Consumes: `load_window`, `time_squitter`, `fit_paths(..., pulse=..., max_paths=1)`.
- Produces: `best_pulse(x, fs, t0_utc, nominal_utc, seed) -> tuple[Pulse, float]`: grid over family `rc` and `rrc` and roll-off 0.10 to 0.70 in 0.05 steps, one-path fit each, minimum of `resid_ratio` refined by a parabola in roll-off; returns the pulse and its `resid_ratio`.
- Produces: `verdict(table: dict) -> str` over `{station: {gs: (median, ci_lo, ci_hi, n)}}` returning `"receive chain"` when, at every station, every ground station's 95 % interval overlaps every other's; `"transmitters"` when at some station two ground stations' intervals do not overlap and, for each ground station, the intervals across stations overlap; otherwise `"inconclusive"`.
- Produces: CLI `pulse_check.py run --data /home/mjh/hamsci/tasks/hfdl-timing --stations b4,ac0g-nd,k3lr[,kfs-main,w2naf] --out DIR` writing `DIR/pulse.toml` (`[<station>.<gs_id>] family, rolloff, n, ci_lo, ci_hi, snr_floor`) and `DIR/pulse.md`.

Window choice per (station, ground station): version 2 found, `snr_db >= 30`, first two UTC days of that station's windows on disk, at most 200 windows taken evenly through them. With fewer than 30 at that floor, take the 100 highest `snr_db` and record the floor used. Family per (station, gs): the one with lower `resid_ratio` on more than half the windows. Intervals: bootstrap of the median roll-off, 1,000 resamples, rng seed 0.

- [ ] **Step 1: Write the failing tests.** `test_recovers_rolloff`: `synth_burst` with `Pulse("rrc", 0.25)` at 30 dB, 5 seeds; `best_pulse` returns family `rrc` and roll-off within 0.03 of 0.25 in every seed. `test_verdict_rules`: three hand-made tables give each verdict. `test_reads_without_writing(tmp_path)`: run on a tmp window tree; the tree's file listing stays identical.
- [ ] **Step 2: Run** `uv run pytest tests/test_pulse_check.py -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: FAIL.
- [ ] **Step 3: Implement** the tool.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Run it** on B4, AC0G-ND and K3LR (and KFS-MAIN, W2NAF if their windows have arrived): `nice -n 19 uv run python tools/pulse_check.py run --data /home/mjh/hamsci/tasks/hfdl-timing --stations b4,ac0g-nd,k3lr --out /home/mjh/hamsci/tasks/hfdl-timing/fit-eval/pulse`. Expected: `pulse.toml` with an entry per (station, gs) seen, and a verdict line in `pulse.md`.
- [ ] **Step 6: Commit.** `git commit -m "tools: pulse check, fitted roll-off per station and ground station"`

### Task 8: Prepare the keying scripts and the stations file

The keying scripts in `tasks/hfdl-timing/keying/` read `<ROOT>/<station>/squitter-part1.csv` with ROOT fixed to `tasks/hfdl-timing`. Each arm needs them pointed at its own directory, with their logic unchanged.

**Files (outside git):**
- Modify: `/home/mjh/hamsci/tasks/hfdl-timing/keying/kdata.py` (ROOT, CACHE, STATIONS)
- Modify: `/home/mjh/hamsci/tasks/hfdl-timing/keying/correct.py` (OUT; its station list)
- Modify: `/home/mjh/hamsci/tasks/hfdl-timing/keying/check4b.py` (its station list)
- Modify: `/home/mjh/hamsci/tasks/hfdl-timing/stations.toml` (add `kfs-main`, `w2naf`)

**Interfaces:**
- Produces: environment variables, each defaulting to today's behaviour: `HFDL_TIMING_ROOT` (default: the parent of `keying/`); `KDATA_CACHE` (default `keying/.kdata.pkl`); `HFDL_TIMING_STATIONS` (comma list; default `ac0g-nd,b4,k3lr,wb6cxc` for kdata and check4b, and `ac0g-nd,b4,k3lr,uacnj,wb6cxc` for correct.py); `KEYING_OUT` (default `keying/`, the parent of `corrected-L*`).
- Produces: `[stations.w2naf]` with `grid = "FN21ei"` (ops/fleet.toml), `reference = ["T6"]`, `hide_reference = true`, `night_hours = [3, 11]`; `[stations.kfs-main]` with the grid from the station's hf-timestd config, `reference = ["T6"]`, `hide_reference = true`, `night_hours = [6, 14]`.

- [ ] **Step 1: Save the baselines.** From `keying/`: `python3 jumprule.py > /home/mjh/hamsci/tasks/hfdl-timing/fit-eval/baseline/jumprule.txt` and `python3 check4b.py corrected-L1 > .../baseline/check4b.txt`, with the repo venv's python (`/home/mjh/hamsci/repos/hfdl-recorder/.venv/bin/python`).
- [ ] **Step 2: Add the environment overrides.**
- [ ] **Step 3: Verify defaults unchanged.** Rerun both commands with no variables set; `diff` against the baselines. Expected: identical.
- [ ] **Step 4: Verify the override.** Make `/home/mjh/hamsci/tasks/hfdl-timing/fit-eval/override-check/` holding a symlink per station to the existing `<station>/squitter-part1.csv` and `t3_shadow/` files (as `rawdir.py` does), set `HFDL_TIMING_ROOT` to it, `KDATA_CACHE` and `KEYING_OUT` inside it, run `correct.py 1`, then `check4b.py` on the new `corrected-L1`. Expected: identical to the baseline `check4b.txt` block for corrected-L1.
- [ ] **Step 5: Read the KFS-MAIN grid.** Read the claude-bus first (`/home/mjh/hamsci/CLAUDE.md`). Then, read-only: `/home/mjh/hamsci/ops/bin/fleet-ssh kfs-main 'grep -h -i grid /etc/hf-timestd/*.toml'`. Write both entries into `stations.toml`. Without an answer, leave `kfs-main` out and say so in the report.
- [ ] **Step 6: Check the file parses:** `cd /home/mjh/hamsci/repos/hfdl-recorder && uv run python -c "import sys; sys.path.insert(0, 'tools'); import squitter_eval as se; print(sorted(se.load_stations('/home/mjh/hamsci/tasks/hfdl-timing/stations.toml')))"`. Expected: seven stations listed.

### Task 9: The streaming replay

**Files:**
- Create: `tools/fit_replay.py`
- Test: `tests/test_fit_replay.py` (reuse `_window` and `_registration` from `tests/test_batch.py`)

**Interfaces:**
- Consumes: `load_window`, `window_row`, `Registrations` (batch); `time_squitter(..., refuse_ambiguous=False)`, `v2_view`; `fit_paths`, `refit_paths`; `pulse.toml` from Task 7.
- Produces: `window_key(path: Path) -> str` = `"<radiod_id>/<YYYYMMDD>/<name>"`, the day taken from the folder.
- Produces: `replay_one(path: Path, regs: Registrations, pulse: Pulse, split: bool) -> tuple[dict, dict]`: the v2 row (`window_row(path, meta, v2_view(a), regs)`) and the fit row. The fit row starts from `window_row(path, meta, a_ref, regs)`, where `a_ref = dataclasses.replace(a, delay_s=..., arrival_utc=..., a1_utc=..., arrival_offset_s=..., doppler_hz=...)` from the fit's reference path, so the counter-label arithmetic stays the batch's own. A fit with `ok=False` gives `found=False` and `reason="fit_<reason>"`: it never falls back to version 2's numbers. A window flagged `ambiguous_far` stays not found in both arms. Both rows carry `window_key`.
- Produces fit-row columns: `fit_reason, n_paths, ref_path, strongest_path, delay_sigma_s, fit_resid_ratio, fit_n_mask, pulse_family, rolloff, pulse_source` (`"table"` or `"default"`), `paths` (JSON list, earliest first, each `{"rel_delay_s", "rel_level_db", "doppler_hz", "sigma_s", "snr_db"}` relative to the reference), `fit_cpu_s`, `v2_cpu_s`, and on split windows `split_delta_s` (preamble-only minus probes-only reference delay, from `refit_paths`) and `split_sigma_s` (the two sigmas combined in quadrature).
- Produces: CLI
  - `fit_replay.py run --data /home/mjh/hamsci/tasks/hfdl-timing --station b4 --out /home/mjh/hamsci/tasks/hfdl-timing/fit-eval/replay --pulse .../pulse/pulse.toml [--workers 4] [--split-every 10] [--limit N]`: walks `<data>/<station>/squitters/*/*/*.npz` in sorted order (skipping `.tmp`), skips keys already in the manifest, sends windows to a `multiprocessing.Pool` (`imap_unordered`, chunksize 8), appends both rows to `rows/<station>.v2.jsonl` and `rows/<station>.fit.jsonl`, flushes and fsyncs both every 100 windows, and only then appends those windows' absolute paths to `consumed-<station>.txt` and fsyncs it. Registrations come from `<data>/<station>/t3_shadow/`. Split-burst refits run on every tenth window in sorted order.
  - `fit_replay.py build --out .../replay --data /home/mjh/hamsci/tasks/hfdl-timing`: for each station with rows, writes `arms/{v2,fit,fit-all}/<station>/squitter-part1.csv` and symlinks `<data>/<station>/t3_shadow/*.jsonl` beside it. Rows deduplicate by `window_key`, the last one written winning. In `v2` and `fit`, a window not found by both arms gets `unpaired` appended to `excluded`; `fit-all` keeps every fit row untouched. Columns: the union, as `batch._write_csv` builds it.
- Produces: free-space guard `check_space(path, start: bool) -> None`, raising `SystemExit` below 1.5 GB free at start and stopping the run cleanly (after the current flush) below 0.75 GB.

- [ ] **Step 1: Write the failing tests.**
  - `test_both_arms_same_windows(tmp_path)`: three windows (one clean, one 2.5 ms 0.9 echo, one noise); both JSONL files hold the same three `window_key`s; the echo window comes back not found in v2 (`ambiguous`) and found with `n_paths == 2` in fit.
  - `test_v2_row_equals_batch(tmp_path)`: for the clean window, the v2 row equals `batch._row_for(path, regs)` plus `window_key`.
  - `test_fit_row_uses_counter_label(tmp_path)`: with a registration, the fit row's `delay_counter_s` differs from the v2 row's by exactly the fit's reference `delay_s` minus v2's `delay_s`, within 1e-9.
  - `test_window_dir_untouched(tmp_path)`: the listing (names and mtimes) of the window tree stays identical after a run.
  - `test_resume_no_duplicates(tmp_path)` (Review Focus 4): run with `--limit 2`, then again with no limit; the manifest lists each window once; after `build`, each arm CSV holds one row per window. A second case appends a row to the JSONL without its manifest line (a simulated kill between flush and manifest); after resume and `build`, still one row per window.
  - `test_same_name_two_days(tmp_path)` (Review Focus 5): the same file name under two day folders gives two keys, two rows per arm, and two manifest lines.
  - `test_space_guard(tmp_path, monkeypatch)`: with `shutil.disk_usage` patched to 1.0 GB free, `run` exits before writing anything.
  - `test_unpaired_tag(tmp_path)`: after `build`, the noise window (not found in either arm) carries no tag; a window found only by the fit carries `unpaired` in `v2` and `fit` but not in `fit-all`.
- [ ] **Step 2: Run** `uv run pytest tests/test_fit_replay.py -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: FAIL.
- [ ] **Step 3: Implement** the tool.
- [ ] **Step 4: Run** the tests, then the whole suite. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "tools: streaming replay, v2 and fit rows per window, resumable manifest"`

### Task 10: Run the replay

Version 2 costs about 0.25 s a window on the devbox (measured 2026-11-13, B4). Expect the fit to cost a few times that. About 110,000 windows sit on the devbox (AC0G-ND 23,255; B4 45,499; K3LR 17,684; UACNJ 12,635; WB6CXC-7 11,176), so plan on several hours with 4 workers. Give Michael a UTC timestamp at each station's start and end.

- [ ] **Step 1: Pilot.** `nice -n 19 uv run python tools/fit_replay.py run --data /home/mjh/hamsci/tasks/hfdl-timing --station b4 --out /home/mjh/hamsci/tasks/hfdl-timing/fit-eval/replay --pulse /home/mjh/hamsci/tasks/hfdl-timing/fit-eval/pulse/pulse.toml --limit 400`. Expected: 400 manifest lines; record the median `fit_cpu_s` and `v2_cpu_s` and the share of `fit_*` failures. More than 1 % failures: stop and debug (superpowers:systematic-debugging) before the full run.
- [ ] **Step 2: Parity with the earlier evaluation.** Join the pilot's v2 rows to `/home/mjh/hamsci/tasks/hfdl-timing/b4/squitter-part1.csv` on (`window_file`, UTC day of `nominal_utc`). Expected: every matched row's `delay_counter_s` agrees within 1e-9 s, and `found` and `reason` agree. Any difference: stop and find out why.
- [ ] **Step 3: Full runs**, one station at a time, in this order: b4, ac0g-nd, k3lr, wb6cxc, uacnj. Same command without `--limit`. The manifest makes each one resumable.
- [ ] **Step 4: KFS-MAIN and W2NAF.** If `/home/mjh/hamsci/tasks/hfdl-timing/kfs-main/squitters/` or `w2naf/squitters/` holds windows, run them too, with their pulse entries (rerun Task 7, Step 5 for them first). If not, record that in `fit-eval/replay/README.txt` with the reason (the daily pull refuses below 10 GB free).
- [ ] **Step 5: Build the arms.** `uv run python tools/fit_replay.py build --out /home/mjh/hamsci/tasks/hfdl-timing/fit-eval/replay --data /home/mjh/hamsci/tasks/hfdl-timing`. Expected: three arm folders, one `squitter-part1.csv` per station in each, row counts equal to the manifest's line counts.

### Task 11: The report

**Files:**
- Create: `tools/fit_report.py`
- Test: `tests/test_fit_report.py`

**Interfaces:**
- Consumes: the arm folders from Task 10; `synth/cases.csv` from Task 6; `pulse/pulse.md` from Task 7; `squitter_eval.load_stations`, `load_station_dir`, `evaluate` (Task 1's fixed version); the keying scripts via Task 8's environment variables.
- Produces: `run_keying(arm_root: Path, stations: list[str], keying_dir: Path) -> dict[str, str]`: runs, with `HFDL_TIMING_ROOT=arm_root`, `KDATA_CACHE=arm_root/keying-run/.kdata.pkl`, `KEYING_OUT=arm_root/keying-run`, `HFDL_TIMING_STATIONS` set, and cwd `arm_root/keying-run`: `jumprule.py`; `correct.py 1`; `check4b.py arm_root/keying-run/corrected-L1`; `crossview.py Riverhead 8.4086 65.9926`; `crossview.py Molokai -0.4082 51.569`. Saves each stdout as `arm_root/keying-run/<script>.txt` and returns them by name.
- Produces parsers: `parse_jumprule(text) -> dict[str, dict]` giving per ground station `false_jumps` (the sum of rule outcomes whose name starts with `rejected`, plus `forced`) and `rsd_l1` (the "V2 jump rule" row, L = 0.5 min); `parse_check4b(text) -> dict[str, dict]` giving per station `p95` and `rsd` from the `uncal` line; `parse_crossview(text) -> dict`.
- Produces measures: `check3_sigmas(arm_root, stations_toml) -> dict[tuple, tuple[float, int]]` from `evaluate(...).check3` with `train_nights=2`, pairs of scoring stations only, San Francisco left out of the pass mark; `honesty(rows) -> dict` (rms of `split_delta_s / split_sigma_s` per station, plus the synthetic value from `cases.csv`, plus a diagnostic named `t6_literal`: rms of (reference delay minus the median of the same ground station and frequency within plus or minus 5 minutes) over `delay_sigma_s`); `spacing_stability(rows) -> list[dict]` (per station, ground station, frequency and night, for rows with two or more paths: median and robust sd of the first two paths' spacing; and per pair of stations sharing a ground station whose path midpoints lie within 250 km of each other, the difference of nightly medians); `san_francisco(v2_rows, fit_rows) -> dict` (each version 2 step over 1 ms between consecutive frames of one station and frequency, labelled `mode hop` when the fit's reference delay moved less than 0.3 ms across the same step and `keying or path` otherwise; counts per station).
- Produces: `render(measures) -> str` and CLI `fit_report.py --replay .../replay --synth .../synth --pulse .../pulse --stations /home/mjh/hamsci/tasks/hfdl-timing/stations.toml --keying /home/mjh/hamsci/tasks/hfdl-timing/keying --out /home/mjh/hamsci/tasks/hfdl-timing/fit-eval/report.md`.

The report holds, in order: what ran (windows per station and arm, dates, CPU per window by path count); the section 4 success table with columns Measure, Today (spec), v2 replay, fit replay, pass mark, verdict, scored on the `v2` and `fit` arms, one line per scoring station or ground station; the same table for `fit-all` against `v2`, without verdicts; the synthetic marks S1-S4; the pulse check and its verdict; the common-view check; path spacing; San Francisco; the extra targets (WB6CXC-7, UACNJ); and what remains open. Verdicts: false jumps per ground station `fit <= 0.5 * v2`; minute p95 `fit < v2` at every scoring station; minute robust sd, check 3 sigma per pair with n of at least 50, and keying residual: no worse (within 5 %); honesty between 0.7 and 1.5.

- [ ] **Step 1: Write the failing tests.** Parsers against the saved baselines (`fit-eval/baseline/jumprule.txt` gives Riverhead `false_jumps == 141` (70 + 70 + 1) and `rsd_l1 == 0.15`; Molokai 57 and 0.23; `check4b.txt` gives ac0g-nd `p95 == 4.90`, `rsd == 0.65` for corrected-L1). `test_honesty_on_known_rows`, `test_spacing_on_known_rows`, `test_san_francisco_labels` on hand-made rows. `test_verdicts`: a measures dict just inside and just outside each mark.
- [ ] **Step 2: Run** `uv run pytest tests/test_fit_report.py -v --basetemp=$HOME/.cache/pytest-tmp/hfdl-fit`. Expected: FAIL.
- [ ] **Step 3: Implement** the tool.
- [ ] **Step 4: Run** the tests, then the whole suite. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "tools: fine-timing evaluation report from the replay arms"`
- [ ] **Step 6: Write the report.** Run the CLI. Read the result through once against the spec's section 4, and write a five-line summary at its top in plain words: what passed, what failed, and what Michael needs to decide.

### Task 12: The list of consumed windows

**Files:**
- Modify: `tools/fit_replay.py` (new subcommand `consumed`)
- Test: `tests/test_fit_replay.py`

**Interfaces:**
- Produces: CLI `fit_replay.py consumed --out .../replay --report .../report.md`, writing `/home/mjh/hamsci/tasks/hfdl-timing/fit-eval/consumed/<station>.txt` (absolute `.npz` paths, sorted, each one in the manifest, present on disk, and holding a row in both arms' CSVs) and `consumed/SUMMARY.txt` (per station: windows, bytes, first and last UTC day, and any manifest entry left out, with the reason). It refuses to run when `report.md` does not exist or predates the newest arm CSV. It deletes nothing.

- [ ] **Step 1: Write the failing tests.** `test_consumed_lists_only_complete_windows(tmp_path)`: one window in the manifest with rows in both arms, one in the manifest with its file removed, one with no fit row; only the first appears, and SUMMARY names the other two with reasons. `test_consumed_needs_a_fresh_report(tmp_path)`: a report older than the arm CSVs makes it exit non-zero. `test_consumed_deletes_nothing(tmp_path)`: the window tree's listing stays identical.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** the subcommand.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "tools: list the windows the evaluation consumed, for pruning"`
- [ ] **Step 6: Run it** and give Michael the paths of `consumed/SUMMARY.txt` and the per-station lists. Delete nothing.

---

## Follow-on plans (not tasked here)

1. **The live analyser, hfdl-analyzer-3.** Promote the fit into `batch.py` as a stage after detection (`ANALYZER_VERSION` 3), fill `delay_sigma_ns`, and add the engineering keys of spec section 5 (`n_paths`, `ref_path`, `strongest_path`, per-path relative delay, level and Doppler, all as text), keeping `second_peak_ratio`, `second_peak_delay_s`, `peak_width_s` and `lobe_distortion` for one release. First measure the fit's CPU on the Proxmox host, never inside the VM (decision 5). Announce on the claude-bus before any B4 deploy. The same plan decides whether the keying prediction chooses the reference path (decision 8) and how to treat `ambiguous_far` windows.
2. **The rig bench session (decision 3c).** Prerequisites: this evaluation's pulse verdict, which says how much the bench can settle; an RF source able to play a CF32 file at an HFDL frequency, locked to the rig's 10 MHz reference; a combiner and attenuators to inject at the RX888 input at a level the rig's AGC leaves alone; the rig's radiod carrying the HFDL fragment, with hfdl-recorder's window keeper on; a CF32 export from `known_symbols.synth_burst` at the source's rate, with a known pulse; `pulse_check.py` able to read the bench windows; and Michael present to cable it and schedule it. The session then fits the receive chain's part, and the station tables keep only the transmitter's part.
3. **The T3 augmentation study's score (decision 7c).** Before any HFDL source joins fusion, rerun `tasks/t3-augment-study` on the fit's reference rows.
4. **KFS-MAIN and W2NAF, second pass.** If their windows have not reached the devbox by Task 10, rerun Tasks 7 (their entries), 10 (their stations), 11 and 12 once they have, with at least three days each.
