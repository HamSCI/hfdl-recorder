# WWV, WWVH and WWVB inputs for check 4

Task 7 of the squitter timing plan, 2 November 2026.  Question: does hf-timestd already keep,
at ND, UACNJ and WB6CXC-7, a week of per-minute broadcast arrivals and the reference tier's
readings, apart from the fused answer?

## Answer: yes for WWV and WWVH; WWVB gives no arrivals

hf-timestd writes the T3 shadow record at every station running v3.85
(`hf_timestd/core/t3_shadow_record.py`).  The files sit in `/var/lib/timestd/t3_shadow/`, one
JSON line per row, one file per stream per UTC day, kept 14 days (`[timing.t3_shadow]
retention_days`, default 14).

- `broadcasts-<channel>-<YYYYMMDD>.jsonl`, one file per channel (SHARED_2500, 5000, 10000,
  15000; WWV_20000, 25000).  One row per broadcast per minute that the channel measured:
  `station` (WWV or WWVH), `arrival_rtp` in the channel's own counter (`rtp_domain`),
  `residual_ms`, `sigma_ms`, `fold_snr_db`, the hop model's `expected_delay_ms`,
  `model_sigma_ms`, `mode`, and the ionosonde source.  These rows come from the measurement
  itself, before fusion.
- `minutes-<YYYYMMDD>.jsonl`: one row per minute holding T3's published registration, and the
  reference readings from the same tick: `t6_offset_ms`, `t4_offset_ms`, `t2_offset_ms`, the
  judge's `t3_vs_adopted`, `t4_vs_adopted` and `t5_vs_adopted`, and `offset_signs`, which
  states each sign convention in the row itself.

Present on all three test stations.  The record began with the v3.85 update: ND and WB6CXC-7
have 1 and 2 November, UACNJ 2 November.  It takes 3 to 4 MB a day, so the 14 days cost
nothing.

WWVB: `/var/lib/timestd/wwvb/<date>.jsonl` holds decoder rows only: `pass` (bits, SNR, carrier
offset) and `frame` (the decoded minute, parity, `vs_wallclock_s`).  No row carries an arrival
time at millisecond grain, so WWVB cannot enter check 4 as a timing source today.  Check 4
compares HFDL against WWV and WWVH; adding WWVB arrivals belongs to hf-timestd's own work.

## Pull method

The daily devbox pull (Task 8) copies `/var/lib/timestd/t3_shadow/` from each station beside
the squitter rows.  Fourteen days of retention leaves slack if a pull fails.  No hf-timestd
export task needed.

## For the solver (Task 9)

- A broadcast row's `arrival_rtp` lives in its own channel's counter.  The minute row's
  `t3.rtp_domain` names the counter T3 registers in.  Solving across channels needs each
  channel's anchor; the RTP origin regression (per-channel origins) bears on this exactly as it
  does on the HFDL channels.  `residual_ms` already subtracts the hop model, in T3's own frame,
  and serves as the simpler input.
- The reference: ND uses `t4_offset_ms` (host minus the LAN GPS server), UACNJ T5
  (`t5_vs_adopted`), WB6CXC-7 T6 (`t6_offset_ms`) once it locks.

## Seen in passing

ND's minute row at 00:00Z on 2 November shows `t3_vs_adopted.shadow_residual_ns` of
−474.9 ms against T4, while `t3_minus_t6_ms` reads 0.012 ms.  One row only; worth a look by
whoever owns the T3 shadow run before anyone reads ND's T3 residuals as accuracy.
