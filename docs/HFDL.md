# HFDL: the High Frequency Data Link

This document describes the High Frequency Data Link (HFDL) system —
what it is, how the worldwide network is constructed, what
information its transmissions carry, and why an opportunistic
HFDL receiver is scientifically valuable to the HamSCI (Ham Radio
Science Citizen Investigation) community.

`README.md` covers what `hfdl-recorder` is and how to install it.
This file is the background on the *signal* the recorder
receives.

## Contents

1. [What HFDL is](#1-what-hfdl-is)
2. [Network architecture](#2-network-architecture)
3. [Air-interface and protocol stack](#3-air-interface-and-protocol-stack)
4. [What's in a transmission](#4-whats-in-a-transmission)
5. [HFDL as an opportunistic signal for HamSCI](#5-hfdl-as-an-opportunistic-signal-for-hamsci)
6. [References](#6-references)

---

## 1. What HFDL is

HFDL is the aviation industry's high-frequency (HF) packet-data
service for aircraft operating beyond the reach of VHF Data Link
(VDL) and Inmarsat / Iridium satellite links — primarily oceanic,
polar, and remote-area routes.  Each transmission carries a short
data packet between an aircraft and a ground station, using a
narrowband (≈2.4 kHz) HF channel chosen from a global frequency
plan that spans roughly 2.8 to 22 MHz.

The HFDL air interface is specified in **ARINC Specification
635**.  The service has been continuously operated since the late
1990s by Aeronautical Radio Inc. (ARINC) — now part of Collins
Aerospace — and is the principal HF-based ACARS (Aircraft
Communications Addressing and Reporting System) carrier in
commercial aviation.

Typical aircraft use cases:

- routine **position reports** in oceanic airspace (where ATC
  surveillance is procedural),
- **dispatch traffic** between flight crews and airline
  operations control,
- **free-text messages** to and from the cockpit,
- **OOOI** events (Out-of-gate, Off-ground, On-ground,
  In-gate),
- **departure and arrival clearances**, **company telex**, and
  similar short messages otherwise carried by VHF ACARS or
  satellite ACARS.

The link is bidirectional: ground stations transmit downlinks to
aircraft, aircraft transmit uplinks (in the air-traffic sense) on
the same band but in different slots.

## 2. Network architecture

### Ground-station constellation

A small number of HF ground stations — approximately a dozen
sites worldwide — give effectively global coverage by exploiting
HF skywave propagation.  Documented station locations span every
inhabited continent and several remote sites: examples include
San Francisco, Molokai, Reykjavík, Riverhead (New York), Auckland,
Hat Yai (Thailand), Shannon, Johannesburg, Barrow (Alaska),
Albrook (Panama), Al Muharraq (Bahrain), Santa Cruz (Bolivia),
Krasnoyarsk, Agana (Guam), and Canarias (Spain).  Each site is
identified by a numeric **ground-station ID** that appears in
every transmission and is the primary key into the location
registry (`/var/lib/hfdl-recorder/systable.conf`).

The fixed-and-known geometry of these stations is the foundation
of HFDL's value to ionospheric science: every transmission
implicitly anchors one endpoint of a propagation path at a
precisely known location.

### Frequency plan

Each ground station is licensed to operate on a subset of about
60 logical channels grouped into 12 **bands** named after their
nominal centre frequency in MHz (HFDL2 through HFDL21).  Channel
selection within a band adapts to propagation conditions: a
station may move its active assignments as the ionosphere changes
through the day, and aircraft scan for whichever frequency is
currently open.  The current global channel list is encoded in
[`src/hfdl_recorder/bands.py`](../src/hfdl_recorder/bands.py).

Worldwide active channels — by band:

| Band   | Centre (kHz) | Sample rate (kS/s) | Active channels (kHz) |
|--------|-------------:|-------------------:|---|
| HFDL2  |  2980        |  80                | 2941, 2944, 2992, 2998, 3007, 3016 |
| HFDL3  |  3477        |  50                | 3455, 3497 |
| HFDL4  |  4672        |  40                | 4654, 4660, 4681, 4687 |
| HFDL5  |  5587        | 277.2              | 5451, 5502, 5508, 5514, 5529, 5538, 5544, 5547, 5583, 5589, 5622, 5652, 5655, 5720 |
| HFDL6  |  6622        | 192                | 6529, 6532, 6535, 6559, 6565, 6589, 6596, 6619, 6628, 6646, 6652, 6661, 6712 |
| HFDL8  |  8902.5      | 160                | 8825, 8834, 8843, 8885, 8886, 8894, 8912, 8921, 8927, 8936, 8939, 8942, 8948, 8957, 8977 |
| HFDL10 | 10061.5      |  80                | 10027, 10030, 10060, 10063, 10066, 10075, 10081, 10084, 10087, 10093 |
| HFDL11 | 11287        | 220                | 11184, 11306, 11312, 11318, 11321, 11327, 11348, 11354, 11384, 11387 |
| HFDL13 | 13310        | 100                | 13264, 13270, 13276, 13303, 13312, 13315, 13321, 13324, 13342, 13351, 13354 |
| HFDL15 | 15025        |  12                | 15025 (squitter-only) |
| HFDL17 | 17944        | 100                | 17901, 17912, 17916, 17919, 17922, 17928, 17934, 17958, 17967, 17985 |
| HFDL21 | 21964        |  80                | 21928, 21931, 21934, 21937, 21949, 21955, 21982, 21990, 21997 |

In practice the **higher bands** (HFDL13 – HFDL21) carry the
bulk of daytime traffic and the **lower bands** (HFDL5 – HFDL10)
the bulk of nighttime traffic, matching the diurnal cycle of HF
propagation.  HFDL15 is reserved for ground-station squitters
only.

### Timing

The ground-station clocks are GPS-disciplined.  Each ground
station's transmissions are emitted on a precise TDMA-style slot
boundary and carry a UTC timestamp accurate to the sub-second
level.  This is a non-trivial property for ionospheric science:
the receive-side propagation delay, frequency offset, and clock
behaviour can all be referenced to a known UTC anchor in every
single frame.

## 3. Air-interface and protocol stack

### Physical layer

- **Modulation:** differentially-encoded M-PSK (M ∈ {2, 4, 8})
  at a fixed **symbol rate of 1800 baud**, chosen adaptively per
  link based on signal-to-noise ratio.  Bit rates of 300, 600,
  1200, and 1800 bit/s are advertised by the ground station and
  selected per transmission.
- **Channel bandwidth:** ≈2.4 kHz (a single SSB-equivalent
  channel; the recorder samples a band-wide window so it can
  follow channel-hopping within the band without
  reconfiguration).
- **Forward error correction:** convolutional code with
  interleaving; the residual frame error rate after deinterleave
  + decode is what `dumphfdl` ultimately presents as a "good"
  frame.

### Link layer

Access is **TDMA-like**, with each ground station running a
fixed slot grid on its assigned channels.  Frames carry a slot
number; aircraft transmit only in slots that the ground station
has allocated to them in response to an earlier logon.

Two frame classes dominate observed traffic:

- **Squitter** frames — broadcast by every ground station every
  32 seconds on every assigned frequency.  A squitter advertises
  the station's ID, the current UTC time, and the frequency
  assignments currently active across the system.  Squitters are
  receivable without any aircraft in the air — they are the
  HFDL system's beacon traffic.
- **Data** frames — uplinks and downlinks carrying ACARS payloads,
  logon/logoff, link-quality measurements, and
  performance-management messages.

### Application layer (ACARS-over-HFDL)

The payload of a data frame is usually an **ACARS** message in
the format defined by ARINC 618 / 620: a structured short text
record identifying the aircraft (24-bit ICAO address, flight
number, registration) and carrying one of several hundred
standardised message labels.  Examples of common labels:

- `H1` — uplink to the cockpit (general)
- `B6` — position / progress report
- `4N` — free-text downlink
- `BE` — clearance request
- `15` — weather request

`dumphfdl` decodes the HFDL framing, calls into **libacars** for
upper-layer parsing, and emits one JSON object per successfully
decoded frame.  HamSCI-relevant fields are extracted and
forwarded to sigmond's local sink (see §4).

## 4. What's in a transmission

For each decoded HFDL frame, `hfdl-recorder` extracts the
following into the `hfdl.spots` table of the local sigmond
sink:

### Link-layer fields (every frame)

| Field            | Meaning |
|------------------|---|
| `time`           | UTC timestamp of frame reception (sub-second). |
| `band_name`      | HFDL band (`HFDL13`, etc.). |
| `station_id`     | Numeric ground-station ID present in every frame. |
| `frequency` / `frequency_mhz` | The sub-channel the frame was carried on. |
| `bit_rate`       | The M-PSK speed selected for this frame (300 / 600 / 1200 / 1800 bit/s). |
| `sig_level`      | In-band signal level reported by the dumphfdl receiver (dB). |
| `noise_level`    | In-band noise floor (dB).  `sig_level − noise_level` is the per-frame SNR. |
| `freq_skew`      | Residual frequency offset of the recovered carrier (Hz).  Includes any Doppler shift and station / receiver oscillator drift. |
| `slot`           | TDMA slot number within the squitter frame grid. |
| `direction`      | `uplink` (aircraft → ground), `downlink` (ground → aircraft), or `squitter` (ground beacon). |
| `ground_station` | Human-readable station name resolved from `systable.conf`. |

### ACARS application-layer fields (when present)

| Field            | Meaning |
|------------------|---|
| `icao_addr`      | 24-bit ICAO aircraft address. |
| `flight`         | Operational flight number (e.g. `UAL836`). |
| `aircraft_reg`   | Aircraft registration (tail number). |
| `acars_label`    | Two-character ACARS message label (`H1`, `B6`, etc.). |
| `acars_message`  | Free-text body of the ACARS message. |
| `position_lat`, `position_lon`, `position_alt_ft` | Reported aircraft position, when the ACARS payload includes one (typical for `B6`-class messages). |
| `raw_json`       | The original `dumphfdl` JSON line, preserved verbatim for re-parse. |

### What this gives you per frame

Every successfully decoded squitter is a complete record of one
end of a propagation path: **(ground station, frequency, time,
received SNR, frequency offset)**.

Every successfully decoded aircraft frame additionally pins **the
other end** of that path — usually within a few hundred kilometres
when the ACARS payload is a position report.

## 5. HFDL as an opportunistic signal for HamSCI

HFDL satisfies most of the criteria that make a signal valuable
for opportunistic ionospheric observation:

1. **Known transmitter geometry.**  Ground-station locations are
   public and fixed.  Once the station ID is decoded, the
   transmit endpoint of the propagation path is known to within
   antenna-pattern uncertainty.

2. **Known frequencies.**  The 12-band global channel plan is
   stable and published.  A receiver can probe propagation
   simultaneously across the full HF range (2.8–22 MHz) with a
   single recorder configuration.

3. **GPS-disciplined timing.**  Every frame carries a UTC
   timestamp accurate to better than a second, and the
   physical-layer slot boundaries are GPS-disciplined.  Doppler
   and clock-offset analyses are well-anchored.

4. **24/7 operation.**  HFDL is a commercial service in
   continuous use.  Ground-station squitters are emitted every
   32 seconds on every assigned frequency, so the data rate of
   propagation observations is high even when no aircraft is in
   view.

5. **Richer telemetry than a beacon.**  Each frame surfaces
   bit rate (adaptive modulation order → coarse link-quality
   indicator), signal and noise levels (per-frame SNR),
   frequency skew (per-frame Doppler / drift), and slot timing.
   That's substantially more per-frame metadata than a WSPR or
   FT8 spot carries.

6. **Free both endpoints when aircraft are heard.**  An ACARS
   position report turns a single decode into a
   two-endpoint great-circle propagation observation — useful
   for path-attribution work, multi-hop characterisation, and
   for studies that need long oblique geometries the amateur
   beacon networks under-sample (oceanic paths, polar paths).

7. **Multi-band visibility.**  Running all 12 sub-band recorders
   in parallel surfaces, in near-real-time, which frequencies
   are open to which ground stations at which times.  This is
   directly the diurnal-MUF / propagation-mode information that
   ionospheric workers want to characterise.

8. **Complements amateur beacons.**  HFDL ground stations sit on
   the commercial side of the HF spectrum, well away from the
   amateur bands used by WSPR / FT8.  A station running both
   `hfdl-recorder` and `wspr-recorder` (or `psk-recorder`) gets
   a non-overlapping pair of propagation-observation channels
   covering effectively the entire HF range.

### Caveats and what HFDL is *not*

- **Not amateur-band.**  HFDL is licensed commercial
  aeronautical traffic.  Receiving and decoding it is permitted
  for personal observation and research in most jurisdictions,
  but **republishing the content** of aircraft data frames may
  carry legal restrictions in some countries; consult local
  rules before mirroring messages publicly.  Squitter content
  is operationally analogous to ATIS / VOLMET broadcasts and
  generally less sensitive.
- **Not a calibrated link budget.**  `sig_level` and
  `noise_level` are dumphfdl's internal estimates, not absolute
  power into the antenna.  They are very useful as relative
  indicators across frames on the same receiver but should not
  be cross-compared between stations without calibration.
- **Aircraft-density-limited.**  Squitters give continuous
  ground-station-side observations, but aircraft frames cluster
  on busy oceanic routes and are absent over quiet regions.
- **No raw IQ archive.**  This recorder pipes IQ directly into
  `dumphfdl` and writes only decoded frames.  Studies that need
  the IQ for re-analysis must capture it separately upstream
  (e.g. via `pcmrecord` on the same ka9q-radio multicast).

### Suggested data products

The combination of `hfdl.spots` + sigmond's local sink lets a
station build, with no further integration work:

- **Diurnal MUF probes** per ground station — squitter receive
  rate vs frequency vs UTC time of day.
- **Path-quality time series** per great-circle (ground-station,
  receiver) pair — per-frame SNR + bit-rate adaptation.
- **Long-oblique propagation campaigns** when aircraft position
  reports are decoded — both endpoints known, frequency known,
  time known.
- **HFDL-vs-WSPR / FT8 cross-validation** — running the
  amateur-band recorder in parallel and matching propagation
  events across the two services.

## 6. References

- **ARINC 635** — HF Data Link Protocols.  The authoritative
  air-interface specification.
- **ARINC 620** / **ARINC 618** — ACARS message format.
- **`dumphfdl`** — open-source HFDL decoder by Tomasz Lemiech
  ([github.com/szpajder/dumphfdl](https://github.com/szpajder/dumphfdl)).
- **`libacars`** — open-source ACARS application-layer parser
  ([github.com/szpajder/libacars](https://github.com/szpajder/libacars)).
- **Airframes** ([app.airframes.io](https://app.airframes.io)) —
  community-run aggregator for crowdsourced HFDL traffic; the
  `hfdl-recorder` `sinks.airframes_io` option feeds this
  service.
- **HamSCI** ([hamsci.org](https://hamsci.org)) — Ham Radio
  Science Citizen Investigation; the umbrella under which this
  recorder's data products contribute to ionospheric research.
