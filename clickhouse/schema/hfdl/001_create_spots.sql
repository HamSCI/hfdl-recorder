-- hfdl-recorder: hfdl.spots — HFDL frames decoded by dumphfdl.
--
-- Wire format: one compact JSON object per line, written by dumphfdl's
-- `--output decoded:json:file:path=<path>` sink (libacars la_json_*).
-- Top-level shape:
--   {"hfdl":{"app":{...},"station":"...","t":{"sec":N,"usec":N},
--            "freq":N,"bit_rate":N,"sig_level":F,"noise_level":F,
--            "freq_skew":F,"slot":"X", ...payload subobjects ...}}
--
-- ACARS-bearing frames also include nested `acars`/`spdu`/`mpdu`/`lpdu`
-- objects with airframe metadata (icao24, flight, label, message text).
-- We extract the high-signal fields here and preserve the full frame in
-- `raw_json` so a future parser upgrade doesn't lose information.

CREATE TABLE IF NOT EXISTS hfdl.spots
(
    -- common header
    time               DateTime64(3, 'UTC')   CODEC(Delta(8), ZSTD(1)),
    host_call          LowCardinality(String) CODEC(LZ4),
    host_grid          LowCardinality(String) CODEC(LZ4),
    radiod_id          LowCardinality(String) CODEC(LZ4),
    instance           LowCardinality(String) CODEC(LZ4),
    band_name          LowCardinality(String) CODEC(LZ4),
    processing_version LowCardinality(String) CODEC(LZ4),

    -- dumphfdl metadata block (always present)
    station_id         LowCardinality(String) CODEC(LZ4),       -- operator-side station id
    frequency          Int64                  CODEC(Delta(8), ZSTD(3)),
    frequency_mhz      Float64                CODEC(Delta(8), ZSTD(3)),
    bit_rate           Int32                  CODEC(T64, ZSTD(1)),
    sig_level          Float32                CODEC(Delta(4), ZSTD(3)),
    noise_level        Float32                CODEC(Delta(4), ZSTD(3)),
    freq_skew          Float32                CODEC(Delta(4), ZSTD(3)),
    slot               LowCardinality(String) CODEC(LZ4),       -- typically "A"|"B"

    -- best-effort parse of nested payloads (nullable / empty when absent)
    direction          LowCardinality(String) CODEC(LZ4),       -- "uplink" | "downlink" | ""
    ground_station     LowCardinality(String) CODEC(LZ4),       -- e.g. "11"|"Krasnoyarsk"
    icao_addr          Nullable(UInt32)       CODEC(T64, ZSTD(1)),  -- ICAO 24-bit hex addr
    flight             LowCardinality(String) CODEC(LZ4),       -- airline code + number
    aircraft_reg       LowCardinality(String) CODEC(LZ4),       -- tail number when present
    acars_label        LowCardinality(String) CODEC(LZ4),       -- ACARS label (e.g. "DM","SG")
    acars_message      String                 CODEC(ZSTD(3)),   -- ACARS body text

    -- position-bearing frames (HFNPDU position reports)
    position_lat       Nullable(Float32)      CODEC(Delta(4), ZSTD(3)),
    position_lon       Nullable(Float32)      CODEC(Delta(4), ZSTD(3)),
    position_alt_ft    Nullable(Int32)        CODEC(T64, ZSTD(1)),

    -- raw frame for re-parse (canonical record)
    raw_json           String                 CODEC(ZSTD(6)),

    ingested_at        DateTime DEFAULT now() CODEC(Delta(4), ZSTD(1))
)
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (host_call, ground_station, time, frequency)
SETTINGS index_granularity = 32768;
