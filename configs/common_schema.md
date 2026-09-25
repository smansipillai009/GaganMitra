# Common telemetry schema (Role A owns this)
#
# Every data source (ESA-ADB, synthetic, future Indian data) must be converted
# into rows that match this shape before anything downstream (Role B's models,
# Role C's adapters, Role D's dashboard) touches them. This is what makes the
# "one engine, many missions" claim true rather than aspirational.
#
# Required columns after adaptation:
#   timestamp       - datetime, sorted ascending, one series per channel
#   satellite_id     - string, e.g. "esa_mission_1" or "demo_sat_01"
#   mission_id        - string, matches a mission config file's mission_id
#   channel_name     - standardized name, e.g. "battery_voltage" (NOT the
#                        source's raw name like "BAT_V" — that mapping lives
#                        in the mission config, see esa_adb.yaml / synthetic_demo_sat.yaml)
#   value             - float
#   unit               - string, e.g. "volts"
#   subsystem          - string, one of: power, thermal, adcs, comms, payload, obc
#   operating_mode    - string or null if unknown, e.g. "nominal", "eclipse", "safe_mode"
#   criticality        - string, one of: low, medium, high
#   source              - string, one of: esa_adb, synthetic, indian_transfer
#
# A source adapter's job (Role C) is ONLY to produce a DataFrame with exactly
# these columns. Nothing downstream should need to know which adapter produced
# the data.
#
# If a field is genuinely unavailable for a given source (e.g. ESA-ADB may not
# label operating_mode for every channel), set it to null / "unknown" — never
# silently omit the column. Downstream code should be able to rely on the
# column always existing, even if its value is sometimes unknown.
