import os
import glob
import math
import requests
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

# ================== CONFIGURATION ==================

# Region of interest (Gulf of Naples, can be changed)
LAT_MIN, LAT_MAX = 40.3, 41.1
LON_MIN, LON_MAX = 13.8, 14.8

# Baseline period for climatology / threshold
# Set these to match your new choice, e.g. full record:
# BASELINE_START = "1984-01-01"
# BASELINE_END   = "2024-12-31"
BASELINE_START = "1984-01-01"
BASELINE_END   = "2013-12-31"  # example 30-year baseline

# Marine heatwave definition
PERCENTILE = 0.9        # 90th percentile
MIN_EVENT_LENGTH = 5    # minimum event duration (days)

# Time coverage (years) to use
START_YEAR = 1984
END_YEAR   = 2024   # inclusive

# PSL base URL for direct file download (global yearly fields)
PSL_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"

# Output (relative to where you run the script)
PLOTS_DIR  = "plots"
TABLES_DIR = "tables"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ====================================================
# 0) TKINTER FOLDER SELECTION
# ====================================================

def select_data_folder(prompt: str) -> str:
    """
    Use a Tkinter dialog to let the user choose a folder.
    This is where the yearly OISST .nc files will live.
    """
    root = tk.Tk()
    root.withdraw()  # hide the main window

    folder = filedialog.askdirectory(title=prompt)
    if not folder:
        raise ValueError("No folder selected. Aborting.")

    return folder


# ====================================================
# 1) DOWNLOAD YEARLY FILES (ONE AT A TIME, SKIP EXISTING)
# ====================================================

def download_year_from_psl(year: int, raw_dir: str, overwrite: bool = False) -> str:
    """
    Download one yearly OISST V2 high-res daily mean file from PSL
    and save it under raw_dir. Returns the local file path.

    Example remote file:
    https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.1984.nc
    """
    os.makedirs(raw_dir, exist_ok=True)

    fname = f"sst.day.mean.{year}.nc"
    url = f"{PSL_BASE_URL}/{fname}"
    out_path = os.path.join(raw_dir, fname)

    # Skip if already present and overwrite=False
    if os.path.exists(out_path) and not overwrite:
        print(f"✓ {year} already exists → skipping download")
        return out_path

    print(f"↓ Downloading {year} from PSL:")
    print(f"  {url}")

    # Stream download with progress bar
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB

        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"  {fname}",
            leave=True,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"  → saved to {out_path}")
    return out_path


def ensure_all_years_downloaded(start_year: int, end_year: int, raw_dir: str):
    """
    Check raw_dir for existing yearly OISST files.
    Only download the years that are missing.
    """
    os.makedirs(raw_dir, exist_ok=True)

    expected_years = set(range(start_year, end_year + 1))
    existing_years = set()

    # Scan folder for files of the form sst.day.mean.YYYY.nc
    for fname in os.listdir(raw_dir):
        if fname.startswith("sst.day.mean.") and fname.endswith(".nc"):
            parts = fname.split(".")
            # expected pattern: sst day mean YYYY nc
            if len(parts) >= 4:
                try:
                    y = int(parts[3])
                    existing_years.add(y)
                except ValueError:
                    pass

    missing_years = sorted(expected_years - existing_years)

    print("\n==========================================")
    print(" OISST DATASET CHECK")
    print("==========================================")
    print(f"Years requested: {start_year}–{end_year}")
    print(f"Existing files:  {sorted(existing_years) if existing_years else '(none)'}")
    print(f"Missing files:   {missing_years if missing_years else 'none'}")
    print("==========================================\n")

    # Download only the missing ones
    for y in missing_years:
        download_year_from_psl(y, raw_dir=raw_dir, overwrite=False)

    if not missing_years:
        print("✓ All requested years already downloaded — nothing to do.")


# ====================================================
# 2) LOAD LOCAL FILES AND BUILD GULF-OF-NAPLES TS
# ====================================================

def load_sst_timeseries_for_region(
    raw_dir: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> xr.DataArray:
    """
    Load multiple locally stored SST netCDF files and directly extract
    the area-averaged time series for a given region.

    NOTE: yearly files are global OISST; we subset to AOI locally.
    """
    pattern = os.path.join(raw_dir, "sst.day.mean.*.nc")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    print("Local SST files found:")
    for f in file_list:
        print("  ", f)

    # Open first file to detect SST variable and longitude convention
    first_ds = xr.open_dataset(file_list[0])

    # Detect SST variable name
    for cand in ["sst", "SST", "sea_surface_temperature", "analysed_sst"]:
        if cand in first_ds.data_vars:
            sst_var = cand
            break
    else:
        raise ValueError("Could not find SST variable in first dataset. "
                         "Check your netCDF files.")

    # Handle longitude convention using first file
    if float(first_ds.lon.max()) > 180:
        # 0–360 convention -> shift our lon range
        lon_min_mod = (lon_min + 360) % 360
        lon_max_mod = (lon_max + 360) % 360
    else:
        lon_min_mod, lon_max_mod = lon_min, lon_max

    ts_list = []

    # Loop over all files, keep only the region + mean
    for f in file_list:
        print(f"  Processing {os.path.basename(f)}...")
        ds = xr.open_dataset(f)
        sst = ds[sst_var]

        # Subset the region
        sst_sub = sst.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min_mod, lon_max_mod)
        )

        # Area-averaged SST for this file
        ts = sst_sub.mean(dim=("lat", "lon"))

        # Load values into memory, then close this file
        ts = ts.load()
        ds.close()

        ts_list.append(ts)

    # Concatenate all time segments into one long time series
    sst_ts = xr.concat(ts_list, dim="time")

    # Kelvin -> Celsius if needed
    if float(sst_ts.mean()) > 100:
        sst_ts = sst_ts - 273.15

    # Ensure time is sorted
    sst_ts = sst_ts.sortby("time")

    print(
        f"\nFinal SST TS: {str(sst_ts.time.values[0])} → "
        f"{str(sst_ts.time.values[-1])} "
        f"({sst_ts.time.size} days)"
    )

    return sst_ts


# ====================================================
# 3) MHW COMPUTATION
# ====================================================

def compute_climatology_and_threshold(
    sst_ts: xr.DataArray,
    baseline_start: str,
    baseline_end: str,
    percentile: float = 0.9
):
    """
    Compute daily climatological mean and percentile threshold
    based on a chosen baseline period.
    """
    sst_base = sst_ts.sel(time=slice(baseline_start, baseline_end))

    if sst_base.time.size < 365 * 10:
        print("Warning: baseline period is quite short; "
              "climatology/threshold may be noisy.")

    doy = sst_base["time"].dt.dayofyear

    clim_mean = sst_base.groupby(doy).mean("time")
    clim_thresh = sst_base.groupby(doy).quantile(percentile, dim="time")

    clim_mean = clim_mean.rename({"dayofyear": "doy"})
    clim_thresh = clim_thresh.rename({"dayofyear": "doy"})

    full_doy = sst_ts["time"].dt.dayofyear
    clim_full = clim_mean.sel(doy=full_doy)
    thresh_full = clim_thresh.sel(doy=full_doy)

    return clim_full, thresh_full


def detect_marine_heatwaves(
    sst_ts: xr.DataArray,
    threshold_ts: xr.DataArray,
    min_length: int = 5
) -> pd.DataFrame:
    """
    Detect marine heatwaves: SST > threshold for at least min_length days.
    """
    sst_vals = sst_ts.values
    thr_vals = threshold_ts.values
    time_vals = pd.to_datetime(sst_ts.time.values)

    above = sst_vals > thr_vals

    events = []
    in_event = False
    start_idx = None

    for i, is_above in enumerate(above):
        if is_above and not in_event:
            in_event = True
            start_idx = i
        elif not is_above and in_event:
            end_idx = i - 1
            duration = end_idx - start_idx + 1

            if duration >= min_length:
                event_sst = sst_vals[start_idx:end_idx+1]
                event_thr = thr_vals[start_idx:end_idx+1]
                intensity = event_sst - event_thr
                max_intensity = float(intensity.max())
                mean_intensity = float(intensity.mean())

                events.append({
                    "start": time_vals[start_idx],
                    "end": time_vals[end_idx],
                    "duration_days": duration,
                    "max_intensity_degC": max_intensity,
                    "mean_intensity_degC": mean_intensity
                })

            in_event = False
            start_idx = None

    # If the series ends during an event
    if in_event:
        end_idx = len(above) - 1
        duration = end_idx - start_idx + 1
        if duration >= min_length:
            event_sst = sst_vals[start_idx:end_idx+1]
            event_thr = thr_vals[start_idx:end_idx+1]
            intensity = event_sst - event_thr
            max_intensity = float(intensity.max())
            mean_intensity = float(intensity.mean())
            events.append({
                "start": time_vals[start_idx],
                "end": time_vals[end_idx],
                "duration_days": duration,
                "max_intensity_degC": max_intensity,
                "mean_intensity_degC": mean_intensity
            })

    return pd.DataFrame(events)


def summarize_events_by_year(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate MHW events by year (using event start date).
    """
    if events_df.empty:
        return pd.DataFrame()

    events_df = events_df.copy()
    events_df["year"] = events_df["start"].dt.year

    summary = events_df.groupby("year").agg(
        n_events=("duration_days", "count"),
        total_mhw_days=("duration_days", "sum"),
        max_intensity_degC=("max_intensity_degC", "max"),
        mean_intensity_degC=("mean_intensity_degC", "mean"),
        longest_event_days=("duration_days", "max")
    ).reset_index()

    return summary


def plot_time_series_with_events(
    sst_ts: xr.DataArray,
    clim_ts: xr.DataArray,
    thresh_ts: xr.DataArray,
    events_df: pd.DataFrame,
    out_path: str,
    region_name: str = "Selected Region"
):
    """
    Plot full SST time series, climatology, threshold + shade MHW events.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    time_vals = sst_ts.time.values

    ax.plot(time_vals, sst_ts, label="SST (°C)", linewidth=1)
    ax.plot(time_vals, clim_ts, label="Climatology (baseline mean)", linestyle="--")
    ax.plot(time_vals, thresh_ts, label=f"{int(PERCENTILE * 100)}th percentile threshold", linestyle=":")

    for i, ev in events_df.iterrows():
        ax.axvspan(ev["start"], ev["end"], alpha=0.2,
                   color="tab:blue",
                   label="MHW event" if i == 0 else None)

    ax.set_title(f"Marine Heatwaves – {region_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (°C)")
    ax.legend()
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# ====================================================
# 4) SIGNIFICANCE TESTS (1–2–3) + PLOTS
# ====================================================

def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_kendall_test(series: pd.Series) -> dict:
    """
    Non-parametric Mann–Kendall trend test + Kendall's tau.
    Returns dict with tau, S, varS, Z, p.
    """
    s = series.dropna()
    n = len(s)
    if n < 3:
        return {
            "n": n,
            "mk_tau": float("nan"),
            "mk_S": float("nan"),
            "mk_varS": float("nan"),
            "mk_Z": float("nan"),
            "mk_p": float("nan"),
        }

    y = s.values

    # S statistic
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[j] > y[i]:
                S += 1
            elif y[j] < y[i]:
                S -= 1

    # Tie correction
    unique, counts = np.unique(y, return_counts=True)
    tie_term = 0
    for c in counts:
        if c > 1:
            tie_term += c * (c - 1) * (2*c + 5)

    varS = (n * (n - 1) * (2*n + 5) - tie_term) / 18.0

    if S > 0:
        Z = (S - 1) / math.sqrt(varS)
    elif S < 0:
        Z = (S + 1) / math.sqrt(varS)
    else:
        Z = 0.0

    p = 2.0 * (1.0 - _normal_cdf(abs(Z)))
    tau = S / (0.5 * n * (n - 1))

    return {
        "n": n,
        "mk_tau": tau,
        "mk_S": float(S),
        "mk_varS": float(varS),
        "mk_Z": float(Z),
        "mk_p": float(p),
    }


def sen_slope(series: pd.Series, x_years: pd.Series) -> float:
    """
    Sen's slope (median of all pairwise slopes).
    Slope is in units of 'per year' if x_years is in years.

    x_years must be a Series indexed the same way as series.
    """
    s = series.dropna()
    x = x_years.loc[s.index].values
    y = s.values
    n = len(s)
    if n < 2:
        return float("nan")

    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)

    if not slopes:
        return float("nan")
    return float(np.median(slopes))


def linear_trend(series: pd.Series, x_years: pd.Series) -> dict:
    """
    Simple OLS linear trend y = a + b * x.
    Returns slope per year, intercept, R^2, p (approx normal).
    """
    s = series.dropna()
    x = x_years.loc[s.index].values.astype(float)
    y = s.values.astype(float)
    n = len(s)
    if n < 3:
        return {
            "lin_slope": float("nan"),
            "lin_intercept": float("nan"),
            "lin_R2": float("nan"),
            "lin_p": float("nan"),
        }

    # Fit y = b*x + a
    b, a = np.polyfit(x, y, 1)
    y_hat = a + b * x

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    R2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    # Standard error of slope
    df = n - 2
    if df <= 0:
        return {
            "lin_slope": float(b),
            "lin_intercept": float(a),
            "lin_R2": float(R2),
            "lin_p": float("nan"),
        }

    s_err2 = ss_res / df
    x_mean = x.mean()
    ssx = ((x - x_mean) ** 2).sum()
    if ssx == 0:
        se_b = float("nan")
        p = float("nan")
    else:
        se_b = math.sqrt(s_err2 / ssx)
        if se_b == 0:
            p = float("nan")
        else:
            t_stat = b / se_b
            # Approximate using normal distribution
            p = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))

    return {
        "lin_slope": float(b),
        "lin_intercept": float(a),
        "lin_R2": float(R2),
        "lin_p": float(p),
    }


def pettitt_test(series: pd.Series) -> dict:
    """
    Pettitt change-point test.
    Returns change_index (int, position in sorted series),
    change_value (x coordinate), and p-value.

    We assume x-axis is sorted (e.g., years).
    """
    s = series.dropna()
    n = len(s)
    if n < 2:
        return {
            "pettitt_index": None,
            "pettitt_x": None,
            "pettitt_p": float("nan"),
        }

    y = s.values
    K = [0] * n
    for t in range(n):
        sum_sign = 0
        for i in range(t + 1):
            for j in range(t + 1, n):
                if y[j] > y[i]:
                    sum_sign += 1
                elif y[j] < y[i]:
                    sum_sign -= 1
        K[t] = sum_sign

    absK = [abs(val) for val in K]
    K_max = max(absK)
    t_index = absK.index(K_max)

    # Approximate p-value
    p = 2.0 * math.exp((-6.0 * (K_max ** 2)) / (n**3 + n**2))

    # Get corresponding x-coordinate (e.g. year)
    x_vals = s.index.values
    x_change = x_vals[t_index]

    return {
        "pettitt_index": int(t_index),
        "pettitt_x": x_change,
        "pettitt_p": float(p),
    }


def plot_trend_series(years: np.ndarray,
                      values: np.ndarray,
                      metric_label: str,
                      out_name: str,
                      lin_slope: float,
                      lin_intercept: float,
                      lin_p: float,
                      mk_p: float,
                      pettitt_x):
    """
    Generic plot for a yearly series with:
    - scatter + line
    - linear trend line
    - vertical Pettitt change-point (if any)
    - text with slope and p-values
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Time series
    ax.plot(years, values, "-o", label=metric_label)

    # Trend line
    if not math.isnan(lin_slope) and not math.isnan(lin_intercept):
        x_line = np.array([years.min(), years.max()])
        y_line = lin_intercept + lin_slope * x_line
        ax.plot(x_line, y_line, "--", label="Linear trend")

    # Pettitt change-point
    if pettitt_x is not None:
        try:
            cx = int(pettitt_x)
            ax.axvline(cx, color="red", linestyle=":", label=f"Pettitt change ({cx})")
        except Exception:
            pass

    ax.set_xlabel("Year")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} – Trend & Change-point")

    # Annotation text
    txt = f"slope = {lin_slope:.4g} /yr\n" \
          f"lin p = {lin_p:.3g}\n" \
          f"MK p = {mk_p:.3g}"
    ax.text(0.01, 0.99, txt, transform=ax.transAxes,
            va="top", ha="left", bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved trend plot to {out_path}")


def run_significance_tests(sst_ts: xr.DataArray,
                           summary_df: pd.DataFrame):
    """
    1) Trend tests on annual mean/min/max SST (with plots).
    2) Trend tests on yearly MHW metrics (summary_df, with plots).
    3) Pettitt change-point tests on each of those series.

    Saves:
    - tables/trend_significance_sst.csv
    - tables/trend_significance_mhw_metrics.csv
    - plots/trend_*.png
    """
    # ---- 1. Annual SST metrics ----
    # Use "YE" (year-end) instead of deprecated "A"
    sst_annual_mean = sst_ts.resample(time="YE").mean()
    sst_annual_max  = sst_ts.resample(time="YE").max()
    sst_annual_min  = sst_ts.resample(time="YE").min()

    years_sst = pd.to_datetime(sst_annual_mean.time.values).year
    idx_sst = pd.Index(years_sst, name="year")

    sst_mean_series = pd.Series(sst_annual_mean.values, index=idx_sst, name="annual_mean_sst")
    sst_max_series  = pd.Series(sst_annual_max.values,  index=idx_sst, name="annual_max_sst")
    sst_min_series  = pd.Series(sst_annual_min.values,  index=idx_sst, name="annual_min_sst")

    sst_metrics = {
        "annual_mean_sst": sst_mean_series,
        "annual_max_sst":  sst_max_series,
        "annual_min_sst":  sst_min_series,
    }

    results_sst = []
    for metric_name, series in sst_metrics.items():
        years = series.index.to_series()

        mk_res = mann_kendall_test(series)
        sen = sen_slope(series, years)
        lin_res = linear_trend(series, years)
        pt_res = pettitt_test(series)

        row = {
            "metric": metric_name,
            "n": mk_res["n"],
            "mk_tau": mk_res["mk_tau"],
            "mk_S": mk_res["mk_S"],
            "mk_varS": mk_res["mk_varS"],
            "mk_Z": mk_res["mk_Z"],
            "mk_p": mk_res["mk_p"],
            "sen_slope_per_year": sen,
        }
        row.update(lin_res)
        row["pettitt_index"] = pt_res["pettitt_index"]
        row["pettitt_x"] = pt_res["pettitt_x"]
        row["pettitt_p"] = pt_res["pettitt_p"]

        results_sst.append(row)

        # --- Plot for this SST metric ---
        plot_trend_series(
            years=series.index.values.astype(int),
            values=series.values,
            metric_label=metric_name,
            out_name=f"trend_sst_{metric_name}.png",
            lin_slope=lin_res["lin_slope"],
            lin_intercept=lin_res["lin_intercept"],
            lin_p=lin_res["lin_p"],
            mk_p=mk_res["mk_p"],
            pettitt_x=pt_res["pettitt_x"],
        )

    results_sst_df = pd.DataFrame(results_sst)
    sst_out_path = os.path.join(TABLES_DIR, "trend_significance_sst.csv")
    results_sst_df.to_csv(sst_out_path, index=False)
    print(f"Saved SST trend significance table to {sst_out_path}")

    # ---- 2. MHW metrics per year ----
    if summary_df is None or summary_df.empty:
        print("No MHW summary_df provided or it is empty; skipping MHW trend tests.")
        return

    mhw_metrics = [c for c in summary_df.columns if c != "year"]

    results_mhw = []
    for metric_name in mhw_metrics:
        # series indexed by year, e.g. 1985, 1986, ...
        series = summary_df.set_index("year")[metric_name]
        years_idx = series.index.to_series()  # IMPORTANT: same index as series

        mk_res = mann_kendall_test(series)
        sen = sen_slope(series, years_idx)
        lin_res = linear_trend(series, years_idx)
        pt_res = pettitt_test(series)

        row = {
            "metric": metric_name,
            "n": mk_res["n"],
            "mk_tau": mk_res["mk_tau"],
            "mk_S": mk_res["mk_S"],
            "mk_varS": mk_res["mk_varS"],
            "mk_Z": mk_res["mk_Z"],
            "mk_p": mk_res["mk_p"],
            "sen_slope_per_year": sen,
        }
        row.update(lin_res)
        row["pettitt_index"] = pt_res["pettitt_index"]
        row["pettitt_x"] = pt_res["pettitt_x"]
        row["pettitt_p"] = pt_res["pettitt_p"]

        results_mhw.append(row)

        # --- Plot for this MHW metric ---
        plot_trend_series(
            years=series.index.values.astype(int),
            values=series.values,
            metric_label=f"MHW {metric_name}",
            out_name=f"trend_mhw_{metric_name}.png",
            lin_slope=lin_res["lin_slope"],
            lin_intercept=lin_res["lin_intercept"],
            lin_p=lin_res["lin_p"],
            mk_p=mk_res["mk_p"],
            pettitt_x=pt_res["pettitt_x"],
        )

    results_mhw_df = pd.DataFrame(results_mhw)
    mhw_out_path = os.path.join(TABLES_DIR, "trend_significance_mhw_metrics.csv")
    results_mhw_df.to_csv(mhw_out_path, index=False)
    print(f"Saved MHW trend significance table to {mhw_out_path}")


# ====================================================
# 5) MAIN
# ====================================================

def main():
    # 0) Ask user where the yearly OISST .nc files are / should be stored
    print("Please select the folder where OISST yearly NetCDF files are stored (or will be downloaded):")
    raw_dir = select_data_folder("Select / create folder for OISST yearly .nc files")

    # 1) Ensure all global yearly files are present locally (skip those already there)
    ensure_all_years_downloaded(START_YEAR, END_YEAR, raw_dir)

    # 2) Build Gulf-of-Naples SST time series from local files
    sst_ts = load_sst_timeseries_for_region(
        raw_dir,
        LAT_MIN, LAT_MAX,
        LON_MIN, LON_MAX
    )

    # 3) Climatology + threshold
    print("Computing climatology and threshold...")
    clim_ts, thresh_ts = compute_climatology_and_threshold(
        sst_ts,
        BASELINE_START,
        BASELINE_END,
        PERCENTILE
    )

    # 4) Detect MHWs
    print("Detecting marine heatwaves...")
    events_df = detect_marine_heatwaves(
        sst_ts,
        thresh_ts,
        MIN_EVENT_LENGTH
    )
    events_csv_path = os.path.join(TABLES_DIR, "marine_heatwaves_events.csv")
    events_df.to_csv(events_csv_path, index=False)
    print(f"Saved events table to {events_csv_path}")

    # 5) Summarize by year
    print("Summarizing events by year...")
    summary_df = summarize_events_by_year(events_df)
    summary_csv_path = os.path.join(TABLES_DIR, "marine_heatwaves_yearly_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved yearly summary to {summary_csv_path}")
    print(summary_df)

    # 6) Plot full time series + MHW shading
    print("Plotting SST time series with MHW events...")
    plot_path = os.path.join(PLOTS_DIR, "sst_mhw_timeseries.png")
    plot_time_series_with_events(
        sst_ts,
        clim_ts,
        thresh_ts,
        events_df,
        plot_path,
        region_name="Gulf of Naples"
    )

    # 7) Significance tests (1–2–3) + trend plots
    print("Running trend & change-point significance tests (with plots)...")
    run_significance_tests(sst_ts, summary_df)

    print("Done.")


if __name__ == "__main__":
    main()
