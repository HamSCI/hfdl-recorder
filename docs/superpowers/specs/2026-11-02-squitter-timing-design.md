# Squitter timing: HFDL ground stations as a night-time UTC source

Design, 2 November 2026.  Michael Hauan (AC0G); drafted with Claude.

## Why

sigmond stations reconstruct UTC from HF signals when they lack a local GPS and PPS.  CHU went
silent this spring, and WWV, WWVH and WWVB serve poorly overnight.  HFDL ground stations
transmit all night, on a 32-second frame that ARINC 635 ties to UTC, from directions WWV
cannot cover: Riverhead to the east, San Francisco and Molokai to the west, Reykjavík to the
north.  If each ground station keys its bursts at a steady offset from UTC, every squitter it
sends carries a timing mark.

dumphfdl already decodes these bursts, but it stamps each one with the host clock when it
detects the preamble, so its stamps carry our own pipeline's latency.  On ND those stamps fall
190–620 ms after the slot start, in clusters that move whenever the recorder restarts.  They
name the slot; they cannot time it.  This design adds the timing.

## What it does

hfdl-recorder keeps, for each squitter it expects from a chosen ground station, a short window
of the raw samples, labelled with UTC from radiod's RTP timestamp and hf-timestd's published
offset.  A batch analyser later correlates each window against the known HFDL preamble, timed
by the sample counter, and records one row per squitter:

  arrival − nominal slot start = the ground station's offset from UTC + the path's group delay
                                 + the receiving station's clock error

A separate offline solver uses those rows to answer whether HFDL improves UTC reconstruction.

## Goals

1. Time HFDL **squitters** (slot 0 of each ground station's frame, uplink only) from **two or
   three ground stations**, chosen as those all three test stations hear well.  On ND's log,
   Riverhead, San Francisco and Molokai look likely; Reykjavík remains a candidate.  The
   analyser may time every squitter it holds; the evaluation focuses on the chosen few.
2. Ship as part of **hfdl-recorder**, in a normal sigmond release.
3. Run alongside the existing decoding.  dumphfdl decodes as now and uploads to airframes.io
   when the operator enables it.
4. Keep a clear path to the **science**: Doppler from the 249 ms pre-key tone, and the channel
   impulse response (multipath) from the preamble and the probe symbols.  The first build saves
   what that later work needs; it does not have to do that work.

## Non-goals

- Aircraft (downlink) bursts.  Aircraft hold only ±50 ms and add an unknown range.
- Feeding hf-timestd's fusion.  That comes after the evaluation, and only if HFDL earns it.
- Doppler and multipath analysis in the first build (see goal 4).

## Anti-goals: failure even if it works

- **Any added stress on real-time processing.**  The station side does no signal processing in
  the sample path: it copies samples into a window and labels them.  Correlation runs later, in
  batches, at low CPU priority, or off the station.  It never shares a deadline with radiod,
  dumphfdl or hf-timestd.  Measure CPU on the Proxmox host, not in the VM.
- **Disturbing decoding.**  If the timing path fails, falls behind or fills its disk, dumphfdl
  keeps decoding and the airframes.io feed keeps flowing.  The timing path drops windows; it
  never blocks `on_samples`.
- **Sending anything off the station.**  Timing rows and windows stay local, as hfdl-recorder's
  outputs do today.  We pull test data back by hand.
- **Using the host clock for timing.**  Labels come from radiod's sample counters, never from
  the host clock.  The host clock may only disambiguate the RTP wrap.  See "Changed after the
  label gate" for the two labels each window carries.

## The signal (confirmed against dumphfdl 1.7 source)

- 1800 symbols per second; the burst sits on a 1440 Hz subcarrier above the channel's dial
  frequency.
- A burst opens with a 448-symbol pre-key (249 ms, constant phase), then a 531-symbol preamble:
  two 127-symbol A sequences, a 127-symbol M1 sequence (one of several, which names the data
  rate and interleaver), a 15-symbol M2 and nine 15-symbol T blocks (`hfdl.c`: `PREKEY_LEN`,
  `A_LEN`, `M1_LEN`, `M2_LEN`, `T_LEN`; the sequences themselves live in that file).
- Then 72 blocks (single slot) of 30 data symbols, each followed by 15 known probe symbols.
- Frame: 32 s, 13 slots of 32/13 s.  A squitter carries a 12-bit frame index and a 4-bit
  frame offset k.  On ND, `frame_index × 32 s + k × 32/13 s` placed every squitter from
  Riverhead (k=5), San Francisco (k=8) and Molokai (k=3) on one UTC grid.  Santa Cruz (3
  squitters) did not fit; the builder should find out why.
- Each squitter carries a `utc_sync` flag per ground station (1,806 true, 3 false on ND).  A
  burst from a station whose flag reads false gets marked, not timed into results.
- The ICAO spec bounds ground-station timing at ±25 ms.  One informal source claims tens of
  microseconds in practice.  Measuring which holds stands as the point of the test.

## Station side (in hfdl-recorder)

- `BandPipeline._on_samples` already receives every band's IQ.  Beside the write to dumphfdl's
  stdin, it hands the block (with its RTP timestamp) to a window keeper.
- The window keeper learns from dumphfdl's decoded squitters which (ground station, frequency,
  frame offset k) currently transmit, among the chosen ground stations.  Ground stations change
  frequencies through the night and announce them in their squitters.
- For each such squitter it expects, it keeps the samples from about 30 ms before the nominal
  slot start to the slot's end.  The first 750 ms cover the longest plausible path delay plus
  the pre-key and preamble; the rest carries the probes for the multipath science.  Slot boundaries come from ka9q-python's `SlotClock` (RTP-anchored,
  drift-immune), with a 32 s cadence; 32 × the band sample rate always comes out an integer.
  The slot start inside the frame (k × 32/13 s) generally falls between samples, so the window
  records the fractional position exactly rather than rounding it.
- The keeper narrows each window to one 3 kHz HFDL channel, at a few kS/s, before saving.  This
  stays a cheap mix-and-decimate.  If it ever costs measurable CPU on the host, the keeper saves
  the band's samples untouched and leaves the narrowing to the batch step.
- Each saved window carries: ground station, frequency, frame index, k, both labels of its
  first sample (see "Changed after the label gate"), the band's sample rate, the decimation applied, and
  the receiving station's timing tier at that moment (T6, T5, T4, T3 or T2).
- Storage, rough: a full slot (2.4 s) at 6 kS/s complex64 holds about 115 kB.  At 2,700 frames
  a day, three ground stations and up to two live frequencies each, about 2 GB a day.  See
  "Settled after the builder check" for retention.  The builder confirms the real figure before
  shipping.
- Off by default.  A config switch turns it on, naming the chosen ground stations.

## Batch analyser

- Runs on the station at low priority, or on pulled windows elsewhere; the same code either
  way.
- For each window, correlate against the pre-key plus the two A sequences (and M1, trying each
  candidate), with the pulse shape dumphfdl uses.  Take the peak, refine it between samples,
  and report the arrival in UTC.
- Subtract the nominal slot start.  Write one row per squitter: ground station, frequency,
  frame, k, arrival minus slot start, peak quality (SNR, peak width, any second peak), the
  `utc_sync` flag, the receiving station's tier, and the pre-key's carrier offset (Doppler).
  Rows go to hfdl-recorder's local output, in the form the sigmond sink accepts, as a new local
  table that hs-uploader has no route for.
- Keep the windows that the science path needs, as later work decides.  The pre-key gives
  Doppler; the preamble and probes give the multipath picture.  Saving the full slot would serve
  multipath better; the builder notes the storage cost of that option.
- **Prove it on synthetic bursts first.**  Build bursts from the real sequences, delay them by
  known amounts (whole and fractional samples), add noise, multipath and Doppler, and check the
  arrival the analyser reports.  Check by mutation: shift the input by a known amount and the
  answer must move by exactly that; break the correlation template and the tests must fail.

## Offline evaluation

Test stations, chosen for spread: **UACNJ** (T5), **ND** (T4, a LAN stratum-1 good to about
0.2 ms), **WB6CXC-7** (TS-1 present, so T6 should come; until it locks, T3 or T2, which cannot
score millisecond results).  Run for at least a week.  WB6CXC-7 rows count for scoring only
while that station holds T6 or T5.  Michael will ask Paul whether WB6CXC-7 has a LAN time
server.

The solver runs offline, on pulled rows, never on a station.  Four checks, in order:

1. **Does each ground station hold a steady offset?**  At a station with a trusted reference,
   each squitter gives offset + path delay.  Separate the two with geometry, the path delay's
   variation through the night, and the three receiving sites.  If the offset repeats night to
   night within about 0.1 ms, HFDL can carry absolute time.  If it wanders by milliseconds
   within the ±25 ms spec, HFDL drops to tracking change, plus the Doppler science.
2. **How tightly does one squitter time?**  The scatter of single arrivals against the
   reference, by hour.  This sets how many squitters a minute's estimate needs.
3. **Do the three sites agree?**  One ground station heard at three sites should give
   differences between sites that match their geometry.  A mismatch points to the labelling or
   the analyser, whatever the ionosphere does.
4. **With and without HFDL.**  Reconstruct each station's clock error overnight from WWV, WWVH
   and WWVB alone, then with HFDL added.  Score both against the station's reference.  Report
   hourly RMS, 95th-percentile error, and hours with no usable answer.  Learn ground-station
   offsets on the first nights; score only on later ones.  At ND, hide T4 from the
   reconstruction and use it only to score, so ND stands in for a station without GPS.

**Success:** check 1 holds, and adding HFDL clearly lowers the overnight 95th-percentile error
or fills hours WWV leaves empty.

## Decided by Michael

- Purpose: shore up UTC reconstruction from HF alone, after the loss of CHU and given WWV's,
  WWVH's and WWVB's weakness overnight.
- Ships in hfdl-recorder, released; runs beside decoding and the optional airframes.io upload.
- Test at UACNJ, ND and WB6CXC-7, for geographic spread.
- Squitters only, from two or three HFDL ground stations, for timing.
- Keep a path to Doppler and multipath for science.
- Evaluate by comparing accuracy with and without HFDL.
- Whatever puts the least stress on real-time processing; hence the offline evaluation.
- WB6CXC-7: TS-1 present, T6 expected; fall back to T3 or T2.
- Install hfdl-recorder at UACNJ and WB6CXC-7 (only ND runs it today), decoding only the two or
  three bands that carry the chosen ground stations at night.
- Deploy to the three test stations by git ahead of a blessed release; fold into the next one.
- If the label phase cannot reach 0.1 ms, wait on raising the RTP origin regression upstream
  until that phase shows it blocks HFDL.

## Claude's calls (Michael to check)

- Station side copies and labels only; correlation runs in batches at low priority.
- Narrow to one 3 kHz channel at the station, unless that costs measurable host CPU.
- Window from 30 ms before the nominal slot start to the slot's end.
- Off by default behind a config switch.
- Checks 1–3 added before Michael's with-and-without comparison.
- Rows stay local in a new sink table with no upload route.

## Settled after the builder check (Claude's calls)

- **The label comes first.**  The RTP origin regression stays open: each dynamic radiod channel
  counts RTP from its own start, and on ND two siblings at the same rate disagree by 20 ms.  The
  hf-timestd offset gets learned on hf-timestd's channels, so it may not carry over to an HFDL
  channel.  The plan's first phase measures each HFDL channel's label error against hf-timestd's
  channels on the same radiod, and proves it under 0.1 ms, or corrects it per channel, before
  anything gets timed.  If neither works, the work stops there and the regression goes upstream.
- **Receive-chain delay.**  The plan models radiod's filter delay for each HFDL band rate and
  checks the model on the rig with a known signal.  The solver also carries one constant per
  receiving station as a nuisance.  Check 1 asks whether each ground station's offset repeats
  night to night, which a per-station constant does not disturb.  Check 3 compares sites, where
  the model's residue shows up.
- **WWV, WWVH and WWVB inputs for check 4.**  hf-timestd keeps its history locally at all
  three stations.  The plan's first phase confirms that it holds a week of per-minute arrivals
  and the reference series, kept apart from the fused answer.  If it doesn't, an export task in
  hf-timestd joins the plan and must run before the test week starts.
- **Disk and pulls.**  Free space: ND 113 GB, UACNJ 66 GB, WB6CXC-7 43 GB.  That affords saving
  the full slot (pre-key through the last probe, about 2.4 s), which serves the multipath
  science: about 2 GB a day at 6 kS/s.  Keep five days on the station, and pull daily by a
  script on the devbox.

## Changed after the label gate (Michael, 2 November 2026)

The label gate failed as first written, and the cause turned out not to be the RTP origin
regression.  On ND, UACNJ and WB6CXC-7, a probe channel labelled through `acquire_anchor_utc`
disagreed with hf-timestd's channel by 0 to 6 ms, differently on every fresh channel (once
253 ms), at correlation 0.99998 or better.  radiod reads the GPS clock when it builds a status
packet but takes the RTP count from the last block, so a one-shot anchor inherits that packet's
lateness.  Labelled instead through each channel's counter map (hamsci-dsp `CounterMap`: RTP to
radiod's input-sample index, from the channel's own INPUT_SAMPLES and RTP_TIMESNAP), the same
two channels agreed to 0.5 µs, with no filter-delay offset to correct.

So each window carries two labels for its first sample:

- **The counter label**: its radiod input-sample index (fractional), with the counter map that
  produced it and the radiod run it belongs to.  Results get scored on this one.  hf-timestd
  logs, every minute at all three stations, a registration in the same index space
  (`t6_anchor_input_index`, `t6_anchor_utc_ns` in its T3 shadow minute rows).  The batch
  analyser or the offline solver converts index to UTC with it; the station converts nothing
  at millisecond grain.
- **The status-pair label**: UTC through `acquire_anchor_utc`, as first planned.  Kept beside
  the counter label as a record.  Their difference measures the status-path noise, window by
  window, for free.

The daily pull fetches hf-timestd's T3 shadow minute rows with the windows.  The per-channel
label correction file from the label check no longer applies: the counter label needs none.

## Left to the builder

- Module layout inside hfdl-recorder, and how the window keeper reads dumphfdl's squitter
  decodes (its JSON output file, or a second output).
- The correlation details: matched filter, interpolation, M1 search, Doppler pre-correction.
- The file format for windows and rows, within the sink's conventions.
- The offline solver's method, provided it scores against held-out nights and hides the
  reference it scores with.

## Related

- Plan: `tasks/plan-hfdl-codar-t3.md` (Michael's workspace), step 5 there.
- `docs/HFDL.md` in this repository.
- dumphfdl 1.7 source, `src/hfdl.c` and `src/spdu.c` (github.com/szpajder/dumphfdl).
- ka9q-python `slot_clock.py` and `rtp_recorder.py::rtp_to_utc`.
