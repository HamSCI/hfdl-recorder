# Squitter fine timing: a multi-path fit for HFDL arrivals

Design, 13 November 2026.  For Michael Hauan (AC0G) and Rob to approve.  Drafted with Claude.
Status: accepted by Michael 2026-11-13 with all nine recommendations (section "Open questions").  No code exists yet; next comes the evaluation plan.

## Why

Michael, 13 November: "We absolutely should time the squitters more finely."

The purpose comes from the T3 admission design (hf-timestd
`docs/superpowers/specs/2026-11-12-t3-automatic-admission-design.md`, "Offsets from the T6
stations").  T6 stations measure each HFDL ground station's offset and publish it through
sigmondb.  T3 stations remove that offset, predict the path, and use each squitter the way they
use WWV, above all at night.  Every millisecond the HFDL arrival carries into fusion, from
whatever cause, lands on a T3 station's estimate of UTC.

Today's analyser (hfdl-recorder `timing/analyzer.py`, version 2) finds one peak per squitter.
It cannot see an echo closer than about 1.3 ms, and when it sees two paths it reports the
stronger.  This design replaces that single peak with a fit of every path the burst arrived by.

## 1. What "finer" means, in numbers

### Where the error comes from today

One squitter's delay, as a T3 station would use it, carries five terms.  The table gives what
the evidence so far shows for each.

| Term | Size today | Source |
|---|---|---|
| Label and anchor | about 0.5 µs | Counter label against hf-timestd's channel on the same radiod (squitter-timing spec, "Changed after the label gate") |
| Detector, thermal noise | 0.7 µs rms at 40 dB input, 2.0 at 10 dB, 5.7 at 0 dB, 17 at −10 dB | Synthetic bursts, SDD task-4 report |
| Echo closer than 1.3 ms, merged into the main lobe | pulls the answer by up to about 0.13 ms (0.6 echo at 0.3 ms: −127 µs; at 0.6 ms: +91 µs); the largest unflagged pull measured −21 µs | Synthetic bursts, SDD task-4 report |
| Peak choice: a later or earlier path taken as "the" arrival | jumps of about +2.3 ms and +14.0 ms | Unconfirmed phase shifts, `tasks/hfdl-timing/keying/README-2.md` |
| Ground-station keying, after the phase model | 0.15 ms (phase 32 s old) to 0.88 ms (30 min, with the 125-frame schedule) at Riverhead; 0.23 to 0.79 ms at Molokai | `keying/README-2.md` |
| Path (ionosphere) | drifts 0.7 to 0.9 ms per 30 min (robust sd); a floor of roughly 0.5 to 0.8 ms under any HFDL timing | `keying/README-2.md`, part 3 |

Three results tie these together.  The same squitter heard at two T6 sites agrees to 0.1 to
0.8 ms (sigma; evaluation check 3, `eval/report-train4.md`).  With the keying sawtooth removed,
Riverhead's and Molokai's within-night scatter falls from 19.3 and 14.5 ms to 0.55 ms.  And the
corrected squitters read a station's clock per minute to 0.6 to 0.9 ms robust sd, with a p95 of
2.3 to 5.3 ms (`keying/check4b.txt`, phase 32 s old).

The 2 to 4 ms spread per HFDL source in the T3 augmentation study
(`tasks/t3-augment-study/result-*.md`) came from the keying sawtooth, not the detector.  The
phase model already answers it.

### What a finer fit can and cannot buy

The detector's thermal noise sits two to three orders below everything else.  Timing the main
peak more precisely would buy nothing.  The ionosphere sets a floor of about half a millisecond,
and no analyser removes it from a single path.

The fit attacks the two terms between those extremes:

- **Peak choice.**  The +2.3 ms and +14.0 ms jumps land in the tails.  The p95 of 2.3 to 5.3 ms
  against a robust sd near 0.7 ms shows how much the tails matter.  `README-2.md` traced part of
  those tails to the sawtooth's wrap; peak choice accounts for some of the rest (a guess, to be
  measured).
- **Unresolved echoes.**  A close echo moves the answer by tens to about a hundred
  microseconds, and it moves it differently at each site and each hour.  The evaluator counted
  `unresolved_multipath` on 3,871 rows beside 7,172 used at ND, 4,889 beside 7,029 at B4, and
  4,782 beside 6,613 at K3LR.  Whether those flags mark real echoes or a pulse-shape mismatch
  remains unknown (task-4 report, "Concerns").  The fit answers that as a side effect.

A third gain matters as much for T3.  Each path the fit separates corresponds to a propagation
mode.  T3 predicts a mode's delay from geometry, IRI and ray tracing.  It can match a predicted
1F arrival only if the row says which arrival came by 1F.  A single peak that sometimes names one
mode and sometimes another defeats that.

### The target

- Every path above the noise reported with its own delay, amplitude, Doppler and sigma.
- Paths resolved down to about 0.4 ms apart at good SNR, against 1.3 ms today.  Basis
  (estimate): the burst occupies about 2.4 kHz (1800 baud, raised-cosine roll-off 0.35), and a
  model fit at high SNR can separate paths somewhat closer than one over the bandwidth.
- Each resolved path timed to 20 µs or better at 20 dB input when the paths lie 0.5 ms or more
  apart.  Estimate, from today's synthetic precision (2 µs at 10 dB, single path) and the
  extra cost of fitting several overlapping paths.
- No row whose delay jumps from one mode to another without saying so.
- Overall: tails that shrink, not a lower floor.  The success measure in section 4 states this
  in numbers.

## 2. The approach

### What the fit models

One squitter leaves the ground station once.  It reaches the receiver by K paths, K from 1 to
about 4.  Each path k delays the burst by τ_k, scales and rotates it by a complex amplitude a_k,
and shifts it by a Doppler f_k:

    y(t) = Σ_k a_k · s(t − τ_k) · exp(j 2π f_k t) + noise

Here s(t) stands for the known part of the burst, modulated with the transmit pulse.  The fit
finds K, and every τ_k, a_k and f_k, by least squares on the complex samples.

Complex, not magnitude.  Two paths from one burst add coherently, with a relative phase that
changes from squitter to squitter.  At 0.3 ms apart they can cancel or reinforce each other.
A fit on correlation magnitude cannot tell those cases apart; a fit on the samples can.

### What it takes from CODAR's CLEAN

`tasks/codar-mf/clean.py` (3 November) solved the same problem for CODAR's gated chirps.  It
synthesises the waveform's noiseless response at a sub-sample delay, scales it to the strongest
remaining peak, subtracts it, and repeats.  After each new component it refits every amplitude
together.  Greedy subtraction alone left a weak arrival on a strong one's rung 1 dB low.  It
stops at a floor in dB above the noise or at a maximum count.  With it, the UACNJ "tracks"
turned out to be ladder rungs, and the real spacings between three radars held to microseconds
all night.

The squitter fit borrows four things:

1. Synthesise the exact response of the known waveform, at any delay, from the waveform code
   already in `timing/waveform.py`.  No measured template.
2. Find paths one at a time, strongest first, as a starting point.
3. Refit all paths jointly after each new one.  For HFDL that refit covers delays and Dopplers
   as well as amplitudes, since a single burst gives no sweeps to average.
4. Stop when a new path fails to lower the residual by more than the noise explains.  The
   current analyser already estimates the noise from the empty half of the spectrum.

It also borrows CODAR's testing habit: synthetic cases with known answers, and mutations that
must break the tests.

One difference matters.  CODAR's CLEAN works on power averaged over many sweeps, where paths add
incoherently.  A squitter offers one burst, so the fit must work on complex samples (above).

### What the known waveform offers

The current fine template covers A, A, M1 and M2: 396 symbols, about 0.22 s.  The saved window
holds the whole slot.  After the preamble come 72 blocks of 30 data symbols, each followed by
15 known probe symbols (squitter-timing spec, "The signal").  So the known symbols run on
through the whole 2.4 s burst:

- the pre-key tone (448 symbols nominal; about 415 at Riverhead), which fixes Doppler;
- the preamble (A, A, M1, M2 and the T blocks), which fixes delay;
- the 1,080 probe symbols, spread over 2.2 s, which follow each path's phase through the burst.

Two paths a fraction of a millisecond apart may still differ in Doppler.  Over 2.2 s the probes
resolve Doppler to roughly 0.45 Hz (estimate: one over the span).  Where delay alone cannot
separate two paths, delay and Doppler together often can.

The fit marks the data symbols as unknown and leaves them out of the model.  dumphfdl decodes
them, so a later version could re-encode the decode and fill the gaps.  That would need
dumphfdl's scrambler, interleaver and coder in Python.  The first build does without it (open
question 4).

### How it combines with the keying-phase model

The keying lag belongs to the transmitter.  It delays every path of one squitter by the same
amount.  Two consequences follow.

- **The spacing between paths of one squitter carries no keying and no clock.**  τ_2 − τ_1 needs
  no T6, no phase model and no label.  Like CODAR's clock-free spacings, it measures the
  ionosphere directly, and any station can use it.
- **The keying tracker should follow one named path.**  The tracker (`keying/ktrack.py` and the
  jump rule) takes each squitter's delay, less the predicted lag, as its evidence.  When that
  delay hops between paths, the tracker sees a false jump.  The +2.3 ms and +14.0 ms clusters
  look like exactly that.  Fed the same path every time, the tracker should see fewer false jumps
  and need its glitch rule less.

The keying phase may help the fit find the paths.  It predicts where the earliest path should
fall, to 0.15 to 0.9 ms.  The fit may use that prediction to name which path counts as the
reference.  It never moves a delay toward the prediction.  A path that lands far from the
prediction goes out as measured, with a flag.  Timing faults get exposed, not corrected.

### Which path the row names

Earlier work at ND found WWVB's arrivals right-skewed by multipath and chose the leading edge.
For HFDL the same reasoning applies.  Later paths come from more hops or higher reflection, and
vary more.  The design names the **earliest path above the significance floor** as the reference
path.  The strongest path, and every other, travels with it.  Open question 1 asks Michael to
confirm that choice.

## 3. What it needs from the data already saved

### What the stations save

hfdl-recorder's window keeper saves each expected squitter as a window: complex64 at 6 kS/s,
14,949 samples (2.49 s), from 30 ms before the nominal slot start to the slot's end.  Each
window carries both labels (counter and status-pair), the ground station, frequency, frame
index and slot.  That covers the pre-key, the preamble and every probe block.

The 2 November spec planned about 2 GB a day and five days on the station.  The measured figure
came out smaller: 0.5 to 0.8 GB per station per day.  Since 9 November the stations keep one day
(B4 five days), with the analyser running on the station.

### What the devbox holds

The daily pull (`tasks/hfdl-timing/pull.sh`) copies every window to the devbox:

| Station | Nights on the devbox | Size |
|---|---|---|
| AC0G-ND | 11-02 to 11-13 | 3.2 GB |
| B4 | 11-03 to 11-13 | 6.0 GB |
| K3LR | 11-03 to 11-13 | 2.7 GB |
| UACNJ | 11-03 to 11-12 | 1.8 GB |
| WB6CXC-7 | 11-02 to 11-13 | 1.6 GB |

The pull brings 4,000 to 6,300 new windows a day from each of ND, B4 and K3LR (`pull.log`,
11-12 and 11-13).

KFS-MAIN and W2NAF run hfdl-recorder and send rows to sigmondb, but `pull.sh` does not cover
them.  With a one-day ring, their windows vanish daily.  Adding them to the pull changes nothing
on the stations: it reads, at low priority, as it does for the other five (open question 6).

### New capture

The fit needs no new capture.  The windows already hold every known symbol.

One small measurement would help: the receive chain's own pulse response.  Task 4 could not tell
whether the multipath flag on strong windows marks real echoes or a gap between our assumed
pulse (raised cosine, roll-off 0.35) and the real one.  A fit with the wrong pulse invents close
echoes.  A synthetic squitter played through the rig's radiod would settle the receive side
(open question 3).

## 4. How to prove it

### Before any live change

All of this runs on the devbox, on pulled windows.  Nothing touches a station.

1. **Synthetic first.**  Bursts built from the real sequences with known paths: one path; two
   paths at 0.3, 0.5, 0.8, 1.3 and 2.5 ms apart; amplitude ratios 0.1 to 0.9; Doppler
   differences of 0 to 1 Hz; input SNR from −10 to +40 dB.  Report, per case, the delay error
   of each path and whether the fit found the right number of paths.  Mutations: shift the input
   by a known amount and every delay must move by exactly that; swap in a wrong pulse and the
   tests must fail; drop the joint refit and the overlap cases must fail.
2. **Pulse check.**  On high-SNR real windows, fit the pulse roll-off as a free parameter.  If
   one value suits every ground station at one site, the receive chain sets it.  If it differs by
   ground station, the transmitters do.
3. **Replay.**  Run analyser version 2 and the fit side by side on the same windows from B4, ND,
   K3LR and KFS-MAIN (KFS-MAIN once its windows reach the devbox).  WB6CXC-7 and UACNJ come
   along as extra targets, not as scoring stations.  Every station scores against its own T6.
4. **Feed the keying work.**  Rerun `keying/jumprule.py`, `keying/check4b.py` and the
   common-view check on the fit's reference path, with nothing else changed.

### The measures

| Measure | Today (version 2) | Success |
|---|---|---|
| False jumps the tracker sees (rejected or forced shifts) | Riverhead 70 rejected + 70 forced over 12 series; Molokai 40 + 17 over 5 | Halved or better |
| Minute clock error, p95 (check4b, phase 32 s old) | ND 4.9, B4 5.3, K3LR 4.8, WB6CXC 2.3 ms | Lower at every T6 station |
| Minute clock error, robust sd | 0.62 to 0.91 ms | No worse |
| Same squitter at two T6 sites, reference path, robust sd | 0.1 to 0.8 ms | No worse; lower counts as a gain |
| Keying prediction residual, phase 32 s old | Riverhead 0.15 ms, Molokai 0.23 ms | No worse |
| Honesty of delay_sigma: rms of (error / sigma) against T6 | none (sigma not reported) | Between 0.7 and 1.5 |

The success thresholds above stand as Claude's proposal (open question 7).

Two further results carry no pass mark but go in the report.  First, whether the spacing between
the first two paths holds steady through a night and agrees between sites that share a
midpoint.  Second, whether San Francisco's 4 to 5 ms wander comes from mode hopping, which the
fit would show, or from its keying.

## 5. What changes, and what stays

### In the live path

The window keeper stays exactly as it runs.  It copies and labels samples and does no signal
processing; the anti-goal of the 2 November spec still binds.

The batch analyser, which already runs on the station at low priority, gains the fit as a stage
after detection.  The current A-pair detection, the Doppler from the pre-key and the M1 search
stay as they run, and seed the fit.  The analyser version goes from 2 to 3, so rows from the two
never mix.

The fit costs more CPU than one correlation.  Estimate: tens to a hundred milliseconds per
window for a joint fit of up to four paths over about 15,000 samples, so 2 to 10 minutes of one
core per day at 6,000 windows.  That guess needs a measurement on the Proxmox host before any
station runs it.  Until then, the fit runs only on the devbox.

Any change on B4 goes on the claude-bus first, as always.

### In the sigmondb rows

Today each squitter gives one arrival row: `detector_version` hfdl-analyzer-2, `clean_tag`
"squitter", `delay_ns` from the single peak, `delay_sigma_ns` empty, and engineering keys for
frame, flags, second peak, peak width and lobe distortion.

The design proposes, as a first step:

- `detector_version` hfdl-analyzer-3.
- One arrival row per squitter still, as today.  Its `delay_ns`, `doppler_hz` and `level_db` now
  describe the reference path (the earliest significant one).
- `delay_sigma_ns` filled, from the fit's covariance.
- New engineering keys: `n_paths`; `ref_path` (its rank by delay, always 0 under the proposed
  rule); `strongest_path`; and, per other path, its delay relative to the reference, its level
  relative to the reference in dB, and its Doppler.  Engineering holds strings, so these go in
  as text, as the current keys do.
- `second_peak_ratio`, `second_peak_delay_s`, `peak_width_s` and `lobe_distortion` stay, filled
  as now, for one release, so the evaluator and the back-fill parity test keep working.

A second step could write one row per path, as codar-sounder already does for CODAR peaks.  It
waits until every consumer filters on `clean_tag`, since a consumer that takes every row for a
transmitter would count one squitter twice (open question 2).

### What stays

- The windows, their two labels and the counter-label route to UTC.
- The capture rows.
- The keying model and the per-cycle publication of each (ground station, frequency) phase.
- The rule that T3 stations never re-level a phase on their own squitters.
- The airframes.io feed and dumphfdl's decoding, untouched.

## 6. Questions for Michael and Rob

1. **Which path should the row's delay name?**  Options: (a) the earliest significant path;
   (b) the strongest path, as today; (c) a path labelled by predicted mode (1F, 2F, E), once T3's
   path model can name modes.  Recommendation: (a) now, with the strongest path in the
   engineering keys; move to (c) when the T3 path model can use it.

2. **One row per squitter, or one row per path?**  Options: (a) one row, other paths in
   engineering keys; (b) one row per path, tagged by `clean_tag`, as CODAR does; (c) both, under
   two detector versions.  Recommendation: (a) for the first release, since today's consumers
   (the keying publisher, the T3 augmentation study, the back-fill parity test) expect one row.
   Plan (b) once those consumers filter on the tag.

3. **How should the fit learn the pulse shape?**  Options: (a) assume raised cosine 0.35, as
   today; (b) fit the roll-off per (receiving station, ground station) from high-SNR windows;
   (c) measure the receive chain on the rig with a synthetic squitter, then fit only the
   transmitter's part per ground station.  Recommendation: (c).  It costs one bench session, and
   it separates our chain from theirs.  (b) serves as the fallback.

4. **Should the fit use the data symbols?**  Options: (a) known symbols only (pre-key,
   preamble, probes); (b) also re-encode dumphfdl's decode to make the whole burst known.
   Recommendation: (a).  The probes already give 1,080 known symbols over 2.2 s.  Take (b) only
   if the evaluation shows the probes fall short.

5. **Where should the fit run, once proven?**  Options: (a) in the station's batch analyser, at
   low priority; (b) only off the station, on the devbox or the sigmondb server, from pulled
   windows; (c) both, the station first.  Recommendation: (a), after a CPU measurement on the
   Proxmox host shows it costs little.  (b) cannot scale: it would pull every window from every
   station.

6. **Add KFS-MAIN and W2NAF to the daily window pull now?**  Their windows last one day on the
   station.  Options: (a) add both to `pull.sh` now (read-only, low priority); (b) raise their
   ring to five days as well; (c) neither, and evaluate on B4, ND and K3LR alone.
   Recommendation: (a).  KFS-MAIN sits near the San Francisco ground station and W2NAF near
   Riverhead, so their short paths test the fit where echoes should come fewest.

7. **Are the proposed success thresholds right?**  Options: (a) accept the table in section 4
   (false jumps halved, p95 lower at every T6 station, nothing else worse, sigma honest to
   0.7-1.5); (b) demand a fixed p95, say under 2 ms; (c) judge only by the T3 augmentation
   study's own score at B4.  Recommendation: (a) for the analyser, then (c) before any HFDL
   source joins fusion, as ruling 9 of 10-30 already requires.

8. **May the keying prediction guide the fit?**  Options: (a) never; the fit sees only the
   window; (b) only to choose which path counts as the reference, never to move a delay; (c) as
   a prior that pulls delays.  Recommendation: (b).  (c) would hide a keying change or a
   timing fault inside the fit.

9. **Should San Francisco count toward success?**  Its keying wanders 4 to 5 ms with no model.
   Options: (a) include it in the measures; (b) report it only as a diagnosis (mode hopping or
   keying).  Recommendation: (b), until something explains its wander.

## Sources

- Squitter-timing spec and plan: `docs/superpowers/specs/2026-11-02-squitter-timing-design.md`,
  `docs/superpowers/plans/2026-11-02-squitter-timing.md`.
- Analyser and its measured limits: `src/hfdl_recorder/timing/analyzer.py`;
  `.superpowers/sdd/2026-11-02-squitter-timing/task-4-report.md`.
- Rows: `src/hfdl_recorder/timing/sigmondb_rows.py`.
- Evaluation: `tasks/hfdl-timing/eval/report-train3.md`, `report-train4.md`.
- Keying: `tasks/hfdl-timing/keying/README.md`, `README-2.md`, `check4b.txt`, `jumprule.txt`.
- Registry: `tasks/hfdl-timing/2026-11-09-hfdl-registry-design.md`.
- CODAR CLEAN: `tasks/codar-mf/clean.py`; codar-sounder `src/codar_sounder/core/sigmondb.py`.
- T3 purpose: hf-timestd `docs/superpowers/specs/2026-11-12-t3-automatic-admission-design.md`;
  `tasks/t3-augment-study/PLAN-v2.md` and `result-*.md`.
