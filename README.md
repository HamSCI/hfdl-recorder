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
                                                                                          └──► hfdl.spots in sigmond's local SQLite sink  (always-on when writable)
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

## Feeding airframes.io

[Airframes](https://app.airframes.io) is the de-facto community
aggregator for crowdsourced HFDL traffic.  The live feed is
identified by a **station name that you register on their site**
— your amateur callsign alone is not the right value.  Sending
feeds with an unregistered or mismatched `station_id` will
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
   [station]
   station_id = "MH-KCOU-HFDL"   # MUST match what you registered

   [sinks]
   airframes_io = true            # enables the TCP push to feed.airframes.io:5556
   ```
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
   `station_id` exactly matches the registered name (it's
   case-sensitive).

The Airframes web UI also offers an installer token (`bash <(curl
...) --token …`) — that's their bundled installer for stations
not running ka9q-radio already, and is **not needed here**.  Our
daemon talks to airframes directly via `dumphfdl`'s own `--output
decoded:json:tcp:…` sink.

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
