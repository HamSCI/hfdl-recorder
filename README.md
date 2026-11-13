# hfdl-recorder

A sigmond-compliant High Frequency Data Link (HFDL) recorder for
[ka9q-radio](https://github.com/ka9q/ka9q-radio).

`hfdl-recorder` subscribes to per-band in-phase / quadrature (I/Q)
multicast streams from one or more `radiod` instances via
[ka9q-python](https://github.com/ka9q/ka9q-python), supervises one
[`dumphfdl`](https://github.com/szpajder/dumphfdl) subprocess per
enabled band (feeding it complex 32-bit float (CF32) I/Q via stdin),
and writes the decoded JavaScript Object Notation (JSON) traffic to
a local file per band — optionally pushing to `feed.airframes.io:5556`
over the Transmission Control Protocol (TCP).  Each decoded frame is
also forwarded to the local HamSCI (Ham Radio Science Citizen
Investigation) sink for scientific use.

It is one of the recorder clients in the HamSCI sigmond suite,
following the same Pattern A install layout and deploy ergonomics as
[psk-recorder](https://github.com/HamSCI/psk-recorder),
[wspr-recorder](https://github.com/HamSCI/wspr-recorder), and
[hf-timestd](https://github.com/HamSCI/hf-timestd).

## About the signal

For background on the HFDL system itself — how the worldwide
network is constructed, what information its transmissions carry,
and why an opportunistic HFDL receiver is scientifically useful —
see [`docs/HFDL.md`](docs/HFDL.md).  In short: HFDL is the
aeronautical industry's HF packet-data service, operated under
ARINC Specification 635 from approximately a dozen
GPS-disciplined ground stations worldwide, on 12 sub-bands
spanning roughly 2.8 to 22 MHz.  Every transmission carries the
ground station's identity and a UTC timestamp; many also carry
aircraft position reports, making each decoded frame an
opportunistic propagation observation with **both endpoints
known**.

## Pipeline: radiod → airframes.io + hamsci sink

```
   radiod                ka9q-python                hfdl-recorder daemon              dumphfdl
  (RX888,             (multicast subscription)    (one BandPipeline / band)         (one process / band)
   antenna)
      │                                                                                  │
      │  per-band I/Q multicast (one Real-time Transport Protocol (RTP) group per HFDL   │
      │  band — typically all on hfdl.local; F32LE complex I/Q at the band's native      │
      │  samprate)                                                                       │
      │                                                                                  │
      ├── HFDL21 @ 21964 kHz, 80 kS/s ───────► MultiStream ──► float32→CF32 ──► stdin ──►│ dumphfdl
      ├── HFDL13 @ 13310 kHz, 100 kS/s ──────► MultiStream ──► float32→CF32 ──► stdin ──►│ dumphfdl
      ├── HFDL11 @ 11287 kHz, 220 kS/s ──────► MultiStream ──► float32→CF32 ──► stdin ──►│ dumphfdl
      ├── HFDL10 @ 10061.5 kHz, 80 kS/s ─────► MultiStream ──► float32→CF32 ──► stdin ──►│ dumphfdl
      ├── HFDL8  @  8902.5 kHz, 160 kS/s ────► MultiStream ──► float32→CF32 ──► stdin ──►│ dumphfdl
      ├── HFDL6  @  6622 kHz, 192 kS/s ──────► MultiStream ──► float32→CF32 ──► stdin ──►│ dumphfdl
      └── HFDL5  @  5587 kHz, 277.2 kS/s ────► MultiStream ──► float32→CF32 ──► stdin ──►│ dumphfdl
                                                                                          │
                                                                                          ├──► /var/lib/hfdl-recorder/<rid>/<band>.json   (always-on)
                                                                                          ├──► feed.airframes.io:5556 over TCP            (opt-in)
                                                                                          └──► (decoded messages are not stored in the sink)
```

`dumphfdl` performs HFDL waveform demodulation (M-PSK + forward
error correction (FEC) + deinterleaving) and ACARS (Aircraft
Communications Addressing and Reporting System) upper-layer
parsing internally, writing one JSON object per decoded message to
each configured `--output` sink.  See [`docs/HFDL.md`](docs/HFDL.md)
§3 for the waveform / protocol detail and §4 for the per-frame
fields the recorder extracts into the local sink.

## HFDL band plan

The 12 HFDL bands and their I/Q requirements are encoded in
[`src/hfdl_recorder/bands.py`](src/hfdl_recorder/bands.py).  Sample
rates match the `samprate` declared in the `ka9q-radio` HFDL
fragment
([`config/fragments/hfdl.conf`](https://github.com/ka9q/ka9q-radio/blob/main/config/fragments/hfdl.conf));
the per-band channel-kHz lists are the worldwide active HFDL
sub-channels.  Full table with active channels is in
[`docs/HFDL.md`](docs/HFDL.md) §2.

Names match the `radiod` section names (`[HFDL21]`, `[HFDL13]`, …)
so an operator can grep for the same identifier in `radiod`'s
config and the recorder's config.

The default config template enables the seven highest-yield bands
across day/night propagation: `HFDL21`, `HFDL13`, `HFDL11`,
`HFDL10`, `HFDL8`, `HFDL6`, `HFDL5`.  HFDL15 (squitter-only) and
the lowest-yield bands are opt-in.

## Quick start

```bash
# First-run install (creates user, venv, builds libacars + liquid-dsp +
# dumphfdl from source, installs systemd unit)
sudo ./scripts/install.sh

# Edit /etc/hfdl-recorder/hfdl-recorder-config.toml — set station_id,
# radiod_status mDNS hostname, and which bands to enable.

# Validate the config
sudo -u hfdlrec hfdl-recorder validate --json

# Start it
sudo systemctl start hfdl-recorder@<radiod-id>

# Watch decodes land
tail -f /var/lib/hfdl-recorder/<radiod-id>/HFDL13.json
```

Under sigmond, hfdl-recorder belongs to the `dasi2` and `client`
profiles, so `smd bringup dasi2` (or `client`) installs it and runs
`config init`.
That init writes one `[[radiod]]` block for each radiod sigmond binds
and turns squitter timing on.  It overwrites only a new file or the
installer's untouched template (keeping a `.bak`); a config that names
a real radiod needs `--reconfig`.  `smd start` then starts one
`hfdl-recorder@<radiod>` instance, with its squitter timer, for each
radiod the config names.

## Feeding airframes.io

**Off by default.**  The feed needs an airframes.io account and a
station registered in the operator's name, so only the operator can
turn it on.  Nothing in hfdl-recorder, squitter timing included,
depends on it.

[Airframes](https://app.airframes.io) is the de-facto community
aggregator for crowdsourced HFDL traffic.  The live feed is
identified by a **station name that you register on their site**
— your amateur callsign alone is not the right value.  Sending
feeds with an unregistered or mismatched `airframes_station_id` will
silently fail (the server accepts the TCP connection but
discards the messages).

Setup:

1. **Create an account** at <https://app.airframes.io>.
2. **Register a station** under your account; choose a station
   name (e.g. `MH-KCOU-HFDL` — the convention is
   `<initials>-<nearest-airport>-HFDL`, but anything unique
   under your account works) and set the software to
   `HFDL-DUMPHFDL` (the protocol Airframes expects on port
   5556).
3. **Edit the config**:

   ```toml
   [sinks]
   airframes_io = true                     # enables the TCP push to feed.airframes.io:5556
   airframes_station_id = "MH-KCOU-HFDL"   # MUST match what you registered
   ```

   Leave `[station] station_id` as it is: it names the station in the
   local JSON, while
   dumphfdl reports `airframes_station_id` to airframes.io.
4. **Validate and restart**:

   ```bash
   sudo -u hfdlrec hfdl-recorder validate --json | jq .ok    # → true
   sudo systemctl restart hfdl-recorder@<radiod-id>
   ```
5. **Verify the live TCP feed**:

   ```bash
   sudo ss -tnp | grep feed.airframes.io
   # expect ESTAB connections, one per enabled band
   tail -f /var/log/hfdl-recorder/<radiod-id>-HFDL13.log
   # expect "output_tcp(feed.airframes.io:5556): connection established"
   ```
6. **Watch the dashboard** at
   `https://app.airframes.io/stations/<your-station-name>`.
   Decodes show up within ~15 minutes once the antenna picks up
   traffic.  If your station page stays empty for an extended
   period despite an `ESTAB` socket, double-check that
   `airframes_station_id` exactly matches the registered name (it's
   case-sensitive).

The Airframes web UI also offers an installer token (`bash <(curl
...) --token …`) — that's their bundled installer for stations
not running ka9q-radio already, and is **not needed here**.  Our
daemon talks to airframes directly via `dumphfdl`'s own `--output
decoded:json:tcp:…` sink.

## Squitter timing (off in the template, on under sigmond)

HFDL ground stations transmit a squitter at a fixed slot of a fixed 32 s
frame, and the slot sits on UTC.  The squitter-timing feature measures when
each squitter arrives, against the station's own registered sample clock.
That gives a propagation delay per ground station, per band, per squitter.

The config template ships `[timing_squitters] enabled = false`.  With it
off, the recorder saves nothing extra, and decoding runs as it always has.
A `config init` that sigmond drives writes `enabled = true`.

Turned on, each enabled band keeps a short window of IQ around each squitter
from the ground stations you list: 6 kS/s, about 2.5 s and 120 kB per window,
about 2 GB a day.  Each window file carries its own UTC label and the radiod
sample index of its first sample.  The windows land in
`/var/lib/hfdl-recorder/squitters/<radiod_id>/<YYYYMMDD>/`.  The recorder
deletes them after `retention_days`, and stops saving (never decoding) when
free space falls below `min_free_gb`.

`ground_stations` names the ground stations whose squitters get windows,
spelled as dumphfdl's system table spells them (`"Reykjavik, Iceland"`).  A band
can decode a ground station that the list leaves out, but its windows never get
saved.  So choose bands and ground stations together.  A station close to a
ground station may sit in its skip zone: UACNJ, about 100 km from Riverhead,
decoded three times as many frames from Reykjavik as from Riverhead on HFDL8 and
HFDL6 (2026-11-02).

The window keeper costs little.  Measured on the Proxmox host at AC0G-ND with
five bands (2026-11-03, 10-minute blocks, keeper on, off, on), it added about
0.017 cores to the recorder's 0.28.

dumphfdl's own JSON spool, `/var/lib/hfdl-recorder/<radiod_id>/<BAND>.json`,
holds every decoded frame and nothing rotates it: about 15 MB a day per station
at ND.  It stays that way, deliberately, until the timing study decides what to
keep.

A batch analyser times the saved windows.  It writes one row per window to
the local `hfdl.squitter_timing` table in sigmond's sink, and it converts
each arrival to UTC through hf-timestd's minute registration
(`/var/lib/timestd/t3_shadow`, read only).

To turn it on by hand on a station sigmond did not configure:

```bash
# 1. In /etc/hfdl-recorder/hfdl-recorder-config.toml:
#      [timing_squitters]
#      enabled = true
#    window_dir must stay under /var/lib/hfdl-recorder.
sudo systemctl restart hfdl-recorder@<radiod-id>
# 2. Once windows accumulate, time them every 15 minutes:
sudo systemctl enable --now hfdl-recorder-squitters@<radiod-id>.timer
```

The windows stay on local disk, and the sink marks every `squitter_timing`
row local, so hs-uploader never ships one.  The batch also writes sigmondb
rows: one arrival per window that yields one, and one capture per frequency
and analyser version per run.  hs-uploader ships those, in their own tars,
through the wsprdaemon.org gateways to sigmondb (pipelines
`hfdl-sigmondb-arrival` and `hfdl-sigmondb-capture` in `deploy.toml`, sent
only by an hs-uploader that offers `sigmondb-tar`).
The nightly `smd admin storage trim --all` covers no local table, so the
rows accumulate until an operator trims them, for example
`smd admin storage trim --target-db hfdl --target-table squitter_timing --max-age 30d --yes`.

Off the station, the same analyser runs without sigmond:
`hfdl-recorder squitter-analyze --dir <windows> --no-sink --jsonl rows.jsonl`.

## Documentation

- [`docs/HFDL.md`](docs/HFDL.md) — what HFDL is, how the
  worldwide network is constructed, what information its
  transmissions carry, and why it is valuable as an
  opportunistic propagation source for HamSCI.
- [`CLAUDE.md`](CLAUDE.md) — development briefing and
  architecture detail for working on the recorder itself.
- [`config/hfdl-recorder-config.toml.template`](config/hfdl-recorder-config.toml.template)
  — the full config schema with inline comments.
- The sigmond client contract is documented at
  [`sigmond/docs/CLIENT-CONTRACT.md`](https://github.com/HamSCI/sigmond/blob/main/docs/CLIENT-CONTRACT.md)
  in the orchestrator repo.

## License

MIT — see [LICENSE](LICENSE).
