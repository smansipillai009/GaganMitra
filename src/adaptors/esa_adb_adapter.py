"""
GaganMitra ESA-ADB-style Adapter

*** NOT REAL ESA DATA ***
Nothing in this file loads, downloads, or parses the actual ESA-ADB / Mars
Express dataset. `generate_esa_benchmark_dataset()` below is 100% synthetic
data generated with numpy, using channel-name mnemonics and orbital numbers
that were chosen to *look like* plausible ESA telemetry. No real ESA-ADB CSV
or Zenodo archive is read anywhere in this codebase (there is no `data/`
directory with real files, and `load_telemetry()` only falls back to this
synthetic generator).

The real ESA-ADB benchmark is public (Kotowski et al., published via Zenodo /
github.com/kplabs-pl/ESA-ADB) and could be wired in here for a genuine claim.
Until that's done, do not present this adapter's output as "real flight
telemetry" or "validation on ESA-ADB" — say "ESA-ADB-style synthetic benchmark"
instead. This adapter demonstrates the *adapter architecture* (same core
pipeline, different channel schema) and nothing more.
"""

from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.adapters.base import BaseTelemetryAdapter
from src.schemas import GroundTruthFault


class EsaAdbAdapter(BaseTelemetryAdapter):
    def __init__(self, mission_id: str = "esa_adb"):
        super().__init__(mission_id=mission_id)

    def load_telemetry(self, source: Optional[Any] = None) -> pd.DataFrame:
        """
        Loads ESA benchmark telemetry from a file/DataFrame if given, otherwise
        falls back to the standardized ESA-ADB-*style* SYNTHETIC benchmark
        (generate_esa_benchmark_dataset — fabricated, not real ESA data).

        NOTE: run_pipeline.py's CLI does NOT go through this method for
        source="esa" or source="esa-real" — it calls
        generate_esa_benchmark_dataset() / load_real_opssat_segments()
        directly instead, so the "prefer real data if present" fallback
        below is currently unused by the CLI. Left in place for any code
        that calls this adapter directly; if you rely on it, verify it
        still does what you expect rather than assuming from this comment.
        """
        if source is not None and isinstance(source, pd.DataFrame):
            return source.copy()
        elif source is not None and isinstance(source, (str, Path)):
            p = Path(source)
            if p.exists():
                return pd.read_csv(p)

        # Prefer the real OPSSAT-AD segment data if it's present (see
        # data/real_esa_opssat/PROVENANCE.md for exactly what is and isn't real here).
        real_segments_file = Path(__file__).resolve().parent.parent.parent / "data" / "real_esa_opssat" / "dataset_partial.csv"
        if real_segments_file.exists():
            df, _ = self.load_real_opssat_segments(real_segments_file)
            return df

        # Default path in project
        default_file = Path(__file__).resolve().parent.parent.parent / "data" / "esa_adb" / "esa_benchmark.csv"
        if default_file.exists():
            return pd.read_csv(default_file)

        # If file does not exist yet, generate the ESA-ADB-*style* synthetic benchmark dataset
        df, _ = self.generate_esa_benchmark_dataset()
        default_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(default_file, index=False)
        return df

    def load_real_opssat_segments(
        self,
        segments_csv: Path,
        sample_interval_sec: float = 1.0,
        seed: int = 7,
    ) -> Tuple[pd.DataFrame, List[GroundTruthFault]]:
        """
        Builds a telemetry DataFrame from the REAL OPSSAT-AD benchmark (ESA OPS-SAT
        CubeSat), source: https://zenodo.org/records/12588359, DOI 10.5281/zenodo.12588359,
        CC-BY-4.0. See data/real_esa_opssat/PROVENANCE.md for the full disclosure.

        `segments_csv` is the real, per-segment featurized file (`dataset.csv` from that
        release): each row is one manually labeled telemetry segment with a REAL channel
        ID, REAL duration/mean/std, and a REAL binary `anomaly` ground-truth label. What
        this method reconstructs is the per-timestamp sample *within* each segment —
        Gaussian noise drawn from that segment's own real mean/std over its real duration —
        because the companion raw-sample file (segments.csv, ~584 MB) wasn't downloadable
        in this environment. Channel identity, segment timing, and the anomaly labels are
        real ESA measurements; the individual per-timestamp values are not. Don't drop that
        caveat when presenting results from this loader.
        """
        seg_df = pd.read_csv(segments_csv)
        rng = np.random.default_rng(seed)
        start_time = datetime(2024, 1, 1, 0, 0, 0)

        channels = sorted(seg_df["channel"].unique())
        per_channel_rows: Dict[str, List[float]] = {ch: [] for ch in channels}
        timestamps: List[datetime] = []
        faults: List[GroundTruthFault] = []
        cursor = start_time

        # Walk segments in file order per channel, laying them end-to-end on a shared
        # timeline so each channel's real durations/labels are preserved.
        for ch in channels:
            ch_segs = seg_df[seg_df["channel"] == ch].sort_values("segment")
            ch_cursor = start_time
            for _, row in ch_segs.iterrows():
                n = max(int(row["len"]), 1)
                std = float(row["std"]) if pd.notna(row["std"]) and row["std"] > 0 else 1e-6
                mean = float(row["mean"])
                values = rng.normal(mean, std, n)
                seg_start = ch_cursor
                for v in values:
                    per_channel_rows[ch].append(float(v))
                    ch_cursor = ch_cursor + timedelta(seconds=sample_interval_sec)
                seg_end = ch_cursor
                if int(row["anomaly"]) == 1:
                    faults.append(GroundTruthFault(
                        fault_id=f"OPSSAT-REAL-{ch}-seg{int(row['segment'])}",
                        start_time=seg_start,
                        end_time=seg_end,
                        subsystem="unknown",
                        fault_type="opssat_ad_labeled_anomaly",
                        affected_channels=[ch],
                        description=f"Real ESA OPS-SAT OPSSAT-AD segment {int(row['segment'])} on channel {ch}, "
                                    f"labeled anomalous by the original human annotators.",
                    ))

        # Align all channels onto one timeline by index (they have different real lengths;
        # pad the shorter ones by holding their last real value rather than inventing new signal).
        max_len = max(len(v) for v in per_channel_rows.values())
        for ch in channels:
            vals = per_channel_rows[ch]
            if len(vals) < max_len:
                vals.extend([vals[-1]] * (max_len - len(vals)))

        timestamps = [start_time + timedelta(seconds=i * sample_interval_sec) for i in range(max_len)]
        data = {"timestamp": timestamps}
        for ch in channels:
            data[ch] = np.round(per_channel_rows[ch], 6)
        df = pd.DataFrame(data)
        return df, faults

    def generate_esa_benchmark_dataset(
        self,
        duration_hours: float = 12.0,
        sample_interval_sec: float = 10.0,
        seed: int = 101
    ) -> Tuple[pd.DataFrame, List[GroundTruthFault]]:
        """
        Synthesizes ESA-ADB-*style* telemetry — fabricated with numpy, not loaded from any
        real ESA/Mars Express file. Channel mnemonics, orbital numbers, and the "anomaly"
        below are invented to resemble the public ESA-ADB benchmark's shape, not copied
        from it. Do not call this "real" or "flight-proven" data in a pitch or on a slide.
        """
        rng = np.random.default_rng(seed)
        start_time = datetime(2024, 6, 15, 0, 0, 0)
        total_seconds = int(duration_hours * 3600)
        num_steps = int(total_seconds / sample_interval_sec)
        time_index = [start_time + timedelta(seconds=i * sample_interval_sec) for i in range(num_steps)]
        t = np.linspace(0, total_seconds, num_steps)

        orbit_period = 400.0 * 60.0  # 400 minutes
        orbit_phase = (t % orbit_period)
        in_eclipse = (orbit_phase < 4500.0).astype(float)  # 75-min Mars shadow

        # NPWD2372: Mars Express 28V Power Bus Voltage
        bus_v = 28.05 + 0.15 * np.cos(2 * np.pi * t / orbit_period) + rng.normal(0, 0.04, num_steps)

        # NPWD2451: Battery Current (amps)
        batt_i = np.where(in_eclipse > 0.5, 8.5 + rng.normal(0, 0.3, num_steps), -4.5 + rng.normal(0, 0.2, num_steps))

        # NTWD0005: Thermal Main Compartment Temp (degC)
        temp = 12.0 + 8.0 * np.sin(2 * np.pi * (t - 1800.0) / orbit_period) + rng.normal(0, 0.15, num_steps)

        # NAWD1020: Reaction Wheel 1 Speed (RPM)
        wheel_rpm = 2400.0 + 500.0 * np.sin(2 * np.pi * t / orbit_period) + rng.normal(0, 20.0, num_steps)

        # NCWD0110: High-Gain Antenna Transmitter Power (dBm)
        tx_power = 34.5 + rng.normal(0, 0.12, num_steps)

        # Known ESA-ADB Benchmark Anomaly:
        # Unexpected Mars occultation battery undervoltage transient & thermal drop
        fault_start_idx = int(num_steps * 0.48)
        fault_end_idx = int(num_steps * 0.62)
        fault_len = fault_end_idx - fault_start_idx

        # Battery bus sag
        bus_v[fault_start_idx:fault_end_idx] -= np.linspace(0.2, 3.4, fault_len)
        # Deep current discharge
        batt_i[fault_start_idx:fault_end_idx] += np.linspace(1.0, 7.5, fault_len)
        # Thermal heater line trip
        temp[fault_start_idx:fault_end_idx] -= np.linspace(0.5, 16.0, fault_len)

        fault = GroundTruthFault(
            fault_id="ESA-MEX-ANOMALY-07",
            start_time=time_index[fault_start_idx],
            end_time=time_index[fault_end_idx],
            subsystem="power",
            fault_type="battery_undervoltage_heater_trip",
            affected_channels=["NPWD2372", "NPWD2451", "NTWD0005"],
            description="ESA-ADB Mars Express Occultation event: Excessive discharge current leading to 28V bus undervoltage alarm and automated thermal heater load shedding."
        )

        df = pd.DataFrame({
            "timestamp": time_index,
            "NPWD2372": np.round(bus_v, 3),
            "NPWD2451": np.round(batt_i, 3),
            "NTWD0005": np.round(temp, 2),
            "NAWD1020": np.round(wheel_rpm, 1),
            "NCWD0110": np.round(tx_power, 2),
        })

        return df, [fault]
