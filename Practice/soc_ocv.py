"""
SOC and OCV helpers for the NASA battery CSVs.

Typical use (from a notebook in Practice/):
    from soc_ocv import load_meta, load_test, add_soc, build_ocv_curve, ocv_lookup
    meta = load_meta()
    soc_grid, ocv_grid = build_ocv_curve(meta, "B0005", cycle=0)
    df = add_soc(load_test(meta, "B0005", "discharge", 10), kind="discharge")
    df["OCV"] = ocv_lookup(df["SOC"], soc_grid, ocv_grid)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "cleaned_dataset"


# ----------------------------------------------------------------------------- loading
def load_meta():
    """Read metadata.csv, clean numeric columns, add a per-cell cycle counter."""
    meta = pd.read_csv(ROOT / "metadata.csv")
    for c in ["Capacity", "Re", "Rct"]:                       # a few rows hold '[]' or complex strings
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta = meta.sort_values(["battery_id", "test_id"]).reset_index(drop=True)
    meta["cycle"] = meta.groupby(["battery_id", "type"]).cumcount()   # 0,1,2,... per cell and type
    return meta


def load_test(meta, battery, kind, cycle):
    """Return the cleaned time series for one test (kind = 'charge' or 'discharge')."""
    row = meta[(meta.battery_id == battery) & (meta.type == kind) & (meta.cycle == cycle)]
    if row.empty:
        raise ValueError(f"no {kind} cycle {cycle} for {battery}")
    row = row.iloc[0]
    df = pd.read_csv(ROOT / "data" / row.filename)
    df = _preprocess(df)
    df.attrs.update(battery=battery, kind=kind, cycle=cycle, filename=row.filename,
                    capacity=row.Capacity, ambient=row.ambient_temperature)
    return df


def _preprocess(df):
    """Make the raw CSV safe for integration."""
    df = df.apply(pd.to_numeric, errors="coerce")           # everything numeric, junk -> NaN
    df = df.dropna(subset=["Time", "Voltage_measured", "Current_measured"])
    df = df.sort_values("Time")
    df = df[df["Time"].diff().fillna(1) > 0]                  # drop repeated timestamps
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- SOC
def _charge_passed_ah(current, time_s):
    """Cumulative charge in Ah by trapezoidal integration of current (A) over time (s)."""
    dt_h = np.diff(time_s, prepend=time_s[0]) / 3600.0
    i_mid = np.concatenate([[current[0]], 0.5 * (current[1:] + current[:-1])])
    return np.cumsum(i_mid * dt_h)


def add_soc(df, kind, capacity=None):
    """
    Add 'Ah' (charge passed) and 'SOC' (0..1) columns by coulomb counting.

    discharge: SOC starts at 1 and falls.  Uses |Current_load| because the sign
               convention differs between cells (B0005 logs -2 A, B0047 logs +1 A).
    charge:    SOC starts at 0 and rises.  Current_measured is positive while charging;
               brief negative spikes in some files are clipped to zero.
    capacity:  Ah that defines SOC = 1. Defaults to the test's own throughput so the
               curve spans 0..1 exactly; pass a fixed value to compare cycles on one scale.
    """
    t = df["Time"].to_numpy(float)
    if kind == "discharge":
        i = np.abs(df["Current_load"].to_numpy(float))
        ah = _charge_passed_ah(i, t)
        q = capacity if capacity else ah[-1]
        soc = 1.0 - ah / q
    elif kind == "charge":
        i = df["Current_measured"].to_numpy(float).clip(min=0)
        ah = _charge_passed_ah(i, t)
        q = capacity if capacity else ah[-1]
        soc = ah / q
    else:
        raise ValueError("kind must be 'charge' or 'discharge'")
    df = df.copy()
    df["Ah"] = ah
    df["SOC"] = np.clip(soc, 0.0, 1.0)
    return df


# ----------------------------------------------------------------------------- OCV
def build_ocv_curve(meta, battery, cycle=0, n_points=101):
    """
    Estimate OCV vs SOC for one cell from discharge `cycle` and the full charge
    that follows it (charge cycle+1; charge 0 is only a partial top-up).

    The charge curve sits above OCV by the charge overpotential and the discharge
    curve below it by the discharge overpotential; their average cancels most of it.
    Returns (soc_grid, ocv_grid) ready for np.interp.
    """
    dis = add_soc(load_test(meta, battery, "discharge", cycle), "discharge")
    chg = add_soc(load_test(meta, battery, "charge", cycle + 1), "charge")
    # keep only the parts where current is flowing: drops the rest period after the
    # discharge cutoff (voltage rebounds while SOC stays 0) and the CV tail of the charge
    dis = dis[dis["Current_load"].abs() > 0.5 * dis["Current_load"].abs().max()]
    chg = chg[chg["Current_measured"] > 0.5 * chg["Current_measured"].max()]

    soc_grid = np.linspace(0, 1, n_points)
    v_dis = np.interp(soc_grid, dis["SOC"][::-1].to_numpy(), dis["Voltage_measured"][::-1].to_numpy())
    v_chg = np.interp(soc_grid, chg["SOC"].to_numpy(), chg["Voltage_measured"].to_numpy())
    ocv_grid = 0.5 * (v_dis + v_chg)
    return soc_grid, ocv_grid


def ocv_lookup(soc, soc_grid, ocv_grid):
    """Interpolate OCV at the given SOC values (array-like)."""
    return np.interp(np.asarray(soc, float), soc_grid, ocv_grid)
