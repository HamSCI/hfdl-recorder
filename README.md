# hfdl-recorder

A sigmond-compliant HFDL (High Frequency Data Link) recorder for
[ka9q-radio](https://github.com/ka9q/ka9q-radio).

`hfdl-recorder` subscribes to per-band IQ multicast streams from one or
more `radiod` instances via [ka9q-python](https://github.com/ka9q/ka9q-python),
supervises one [`dumphfdl`](https://github.com/szpajder/dumphfdl) subprocess
per enabled band (feeding it CF32 IQ via stdin), and writes the decoded
JSON to a local file per band — optionally pushing to
`feed.airframes.io:5556` over TCP.

It is the fourth client in the HamSCI sigmond contract v0.6 family,
following the same Pattern A install layout and deploy ergonomics as
[psk-recorder](https://github.com/mijahauan/psk-recorder),
[wspr-recorder](https://github.com/mijahauan/wspr-recorder), and
[hf-timestd](https://github.com/mijahauan/hf-timestd).

## Pipeline: radiod → airframes.io

```
   radiod                ka9q-python                hfdl-recorder daemon              dumphfdl
  (RX888,             (multicast subscription)    (one BandPipeline / band)         (one process / band)
   antenna)
      │                                                                                  │
      │  per-band IQ multicast (one RTP group per HFDL band — typically all              │
      │  on hfdl.local; F32LE complex IQ at the band's native samprate)                  │
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
                                                                                          └──► feed.airframes.io:5556 over TCP            (opt-in)
```

dumphfdl performs HFDL waveform demodulation (M-PSK, FEC, deinterleaving)
and ACARS upper-layer parsing internally. It writes one JSON object per
decoded message to each configured `--output` sink. With
`sinks.airframes_io = true` the daemon adds
`--output decoded:json:tcp:address=feed.airframes.io,port=5556` to that
band's dumphfdl argv, and dumphfdl pushes each message directly — there
is no client-side aggregation.

## HFDL band plan

The 12 HFDL bands and their IQ requirements are encoded in
[`src/hfdl_recorder/bands.py`](src/hfdl_recorder/bands.py). Sample rates
match the `samprate` declared in the `ka9q-radio` HFDL fragment
([`config/fragments/hfdl.conf`](https://github.com/ka9q/ka9q-radio/blob/main/config/fragments/hfdl.conf));
the per-band channel-kHz lists are the worldwide active HFDL
sub-channels from `aux/start-hfdl.sh`.

| Band   | Center (kHz) | Samprate (kS/s) | Channels (kHz) |
|--------|-------------:|----------------:|---|
| HFDL2  |  2980        |  80             | 2941, 2944, 2992, 2998, 3007, 3016 |
| HFDL3  |  3477        |  50             | 3455, 3497 |
| HFDL4  |  4672        |  40             | 4654, 4660, 4681, 4687 |
| HFDL5  |  5587        | 277.2           | 5451, 5502, 5508, 5514, 5529, 5538, 5544, 5547, 5583, 5589, 5622, 5652, 5655, 5720 |
| HFDL6  |  6622        | 192             | 6529, 6532, 6535, 6559, 6565, 6589, 6596, 6619, 6628, 6646, 6652, 6661, 6712 |
| HFDL8  |  8902.5      | 160             | 8825, 8834, 8843, 8885, 8886, 8894, 8912, 8921, 8927, 8936, 8939, 8942, 8948, 8957, 8977 |
| HFDL10 | 10061.5      |  80             | 10027, 10030, 10060, 10063, 10066, 10075, 10081, 10084, 10087, 10093 |
| HFDL11 | 11287        | 220             | 11184, 11306, 11312, 11318, 11321, 11327, 11348, 11354, 11384, 11387 |
| HFDL13 | 13310        | 100             | 13264, 13270, 13276, 13303, 13312, 13315, 13321, 13324, 13342, 13351, 13354 |
| HFDL15 | 15025        |  12             | 15025 (squitter-only) |
| HFDL17 | 17944        | 100             | 17901, 17912, 17916, 17919, 17922, 17928, 17934, 17958, 17967, 17985 |
| HFDL21 | 21964        |  80             | 21928, 21931, 21934, 21937, 21949, 21955, 21982, 21990, 21997 |

Names match the radiod section names (`[HFDL21]`, `[HFDL13]`, …) so an
operator can grep for the same identifier in radiod's config and the
recorder's config.

The default config template enables the seven highest-yield bands across
day/night propagation: `HFDL21`, `HFDL13`, `HFDL11`, `HFDL10`, `HFDL8`,
`HFDL6`, `HFDL5`. HFDL15 (squitter-only) and the lowest-yield bands are
opt-in.

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

[Airframes](https://app.airframes.io) is the de-facto aggregator for
crowdsourced HFDL traffic. The live feed is identified by a **station
name that you register on their site** — your amateur callsign alone is
not the right value. Sending feeds with an unregistered or mismatched
`station_id` will silently fail (the server accepts the TCP connection
but discards the messages).

Setup:

1. **Create an account** at <https://app.airframes.io>.
2. **Register a station** under your account; choose a station name
   (e.g. `MH-KCOU-HFDL` — the convention is `<initials>-<nearest-airport>-HFDL`,
   but anything unique under your account works) and set the software
   to `HFDL-DUMPHFDL` (the protocol Airframes expects on port 5556).
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
6. **Watch the dashboard** at `https://app.airframes.io/stations/<your-station-name>`.
   Decodes show up within ~15 minutes once the antenna picks up
   traffic. If your station page stays empty for an extended period
   despite an `ESTAB` socket, double-check that `station_id` exactly
   matches the registered name (it's case-sensitive).

The Airframes web UI also offers an installer token (`bash <(curl ...) --token …`)
— that's their bundled installer for stations not running ka9q-radio
already, and is **not needed here**. Our daemon talks to airframes
directly via dumphfdl's own `--output decoded:json:tcp:…` sink.

## Documentation

- See [CLAUDE.md](CLAUDE.md) for development briefing and architecture.
- See [config/hfdl-recorder-config.toml.template](config/hfdl-recorder-config.toml.template)
  for the full config schema.
- The client contract v0.6 spec lives in
  [sigmond/docs/CLIENT-CONTRACT.md](https://github.com/mijahauan/sigmond/blob/main/docs/CLIENT-CONTRACT.md).

## License

MIT — see [LICENSE](LICENSE).
