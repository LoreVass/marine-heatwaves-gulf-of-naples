"""
Marine heatwave detection workflow for the Gulf of Naples
using NOAA OISST v2 high-resolution daily SST (1984–2024).

Main steps
----------
1. Download yearly global OISST NetCDF files from NOAA/PSL
   into `data/raw/` (only missing years are fetched).
2. Build an area-averaged SST time series for the Gulf of Naples.
3. Compute seasonal climatology and percentile-based threshold
   over a user-defined baseline period.
4. Detect marine heatwave (MHW) events following Hobday et al. (2016).
5. Summarise events by year and save tables to `tables/`.
6. Generate diagnostic plots:
   - full SST time series with MHW shading
   - trend and change-point analysis for SST and MHW metrics.

Assumptions & conventions
-------------------------
- Input data: NOAA OISST v2 high-resolution daily fields from PSL.
- Units: Kelvin or °C (auto-converted to °C if mean > 100).
- Calendar: Gregorian; time axis handled via xarray/pandas.
- Region: Gulf of Naples (configurable via LAT/LON bounds).
- Longitudes: automatic handling of 0–360 vs −180–180 grids.

Outputs are designed to be reproducible and easily reusable
for further analysis or publication-quality figures.
"""

import os
import glob
import math
import requests
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ================== USER CONFIGURATION ==================

# Region of interest (ROI) for the Gulf of Naples.
# Lat/Lon bounds are in degrees, using a regular lat/lon grid.
LAT_MIN, LAT_MAX = 40.3, 41.1
LON_MIN, LON_MAX = 13.8, 14.8

# Baseline period used to compute:
#   - daily climatological mean SST
#   - percentile-based threshold for MHW definition
# This can be any sub-period within the full record.
BASELINE_START = "1984-01-01"
BASELINE_END   = "2013-12-31"  # example 30-year baseline (WMO-style)

# Marine heatwave definition following Hobday et al. (2016):
#   - SST above the percentile threshold
#   - for at least MIN_EVENT_LENGTH consecutive days.
PERCENTILE = 0.9        # 90th percentile threshold
MIN_EVENT_LENGTH = 5    # minimum event duration [days]

# Temporal coverage for the analysis (inclusive years).
# These years will be checked/downloaded from PSL.
START_YEAR = 1984
END_YEAR   = 2024

# NOAA PSL base URL for global yearly OISST fields.
# Each file contains one year of daily SST on a regular grid.
PSL_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"

# Directory where raw OISST NetCDF files are stored.
# This is relative to the working directory where you run the script
# (typically the repository root).
RAW_DATA_DIR = os.path.join("data", "raw")

# Output directories for derived products.
PLOTS_DIR  = "plots"    # figures: time series, trends, etc.
TABLES_DIR = "tables"   # CSV tables: events, yearly summaries, tests
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# ====================================================
# 1) DOWNLOAD YEARLY FILES (ONE AT A TIME, SKIP EXISTING)
# ====================================================

def download_year_from_psl(year: int, raw_dir: str, overwrite: bool = False) -> str:
    """
    Download a single yearly OISST v2 high-resolution daily mean file
    from the NOAA PSL server into `raw_dir`.

    Parameters
    ----------
    year : int
        Year to download (e.g. 1984).
    raw_dir : str
        Local directory where the resulting NetCDF file will be stored.
    overwrite : bool, optional
        If True, re-download even if the file already exists.

    Returns
    -------
    str
        Local path to the downloaded (or existing) NetCDF file.

    Notes
    -----
    Example URL:
    https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.1984.nc
    """
    os.makedirs(raw_dir, exist_ok=True)

    fname = f"sst.day.mean.{year}.nc"
    url = f"{PSL_BASE_URL}/{fname}"
    out_path = os.path.join(raw_dir, fname)

    # Skip file if it already exists and overwrite is False
    if os.path.exists(out_path) and not overwrite:
        print(f"✓ {year} already exists → skipping download")
        return out_path

    print(f"↓ Downloading {year} from PSL:")
    print(f"  {url}")

    # Stream download with a progress bar for large files
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks

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
    Check which yearly OISST files are already present in `raw_dir`
    and download only the missing years.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive range of years to ensure (e.g. 1984–2024).
    raw_dir : str
        Directory where NetCDF files are stored.

    Side effects
    ------------
    - Prints a summary of existing vs missing years.
    - Downloads missing yearly files.
    """
    os.makedirs(raw_dir, exist_ok=True)

    expected_years = set(range(start_year, end_year + 1))
    existing_years = set()

    # Look for files named like: sst.day.mean.YYYY.nc
    for fname in os.listdir(raw_dir):
        if fname.startswith("sst.day.mean.") and fname.endswith(".nc"):
            parts = fname.split(".")
            if len(parts) >= 4:
                try:
                    y = int(parts[3])
                    existing_years.add(y)
                except ValueError:
                    # Ignore unexpected file name patterns
                    pass

    missing_years = sorted(expected_years - existing_years)

    print("\n==========================================")
    print(" OISST DATASET CHECK")
    print("==========================================")
    print(f"Years requested: {start_year}–{end_year}")
    print(f"Existing files:  {sorted(existing_years) if existing_years else '(none)'}")
    print(f"Missing files:   {missing_years if missing_years else 'none'}")
    print("==========================================\n")

    # Download only the years that are missing
    for y in missing_years:
        download_year_from_psl(y, raw_dir=raw_dir, overwrite=False)

    if not missing_years:
        print("✓ All requested years already downloaded — nothing to do.")


# ====================================================
# 2) LOAD LOCAL FILES AND BUILD GULF-OF-NAPLES TIME SERIES
# ====================================================

def load_sst_timeseries_for_region(
    raw_dir: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> xr.DataArray:
    """
    Build an area-averaged SST time series for a given lat/lon box
    from multiple yearly NetCDF files stored locally.

    Parameters
    ----------
    raw_dir : str
        Directory containing OISST NetCDF files (sst.day.mean.YYYY.nc).
    lat_min, lat_max : float
        Latitude bounds of the region of interest [deg].
    lon_min, lon_max : float
        Longitude bounds of the region of interest [deg].

    Returns
    -------
    xarray.DataArray
        1D SST time series (time dimension), in °C, spatially averaged
        over the region.

    Notes
    -----
    - Automatically detects whether the dataset uses 0–360 or −180–180
      longitude convention and adjusts the ROI bounds accordingly.
    - If input SST is in Kelvin (mean > 100), values are converted to °C.
    """
    pattern = os.path.join(raw_dir, "sst.day.mean.*.nc")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    print("Local SST files found:")
    for f in file_list:
        print("  ", f)

    # Open first file to inspect SST variable name and longitude convention
    first_ds = xr.open_dataset(file_list[0])

    # Detect SST variable name commonly used in gridded products
    for cand in ["sst", "SST", "sea_surface_temperature", "analysed_sst"]:
        if cand in first_ds.data_vars:
            sst_var = cand
            break
    else:
        raise ValueError("Could not find SST variable in first dataset. "
                         "Check your NetCDF files for variable naming.")

    # Adjust longitude bounds if dataset uses 0–360 convention
    if float(first_ds.lon.max()) > 180:
        lon_min_mod = (lon_min + 360) % 360
        lon_max_mod = (lon_max + 360) % 360
    else:
        lon_min_mod, lon_max_mod = lon_min, lon_max

    ts_list = []

    # Loop over yearly files, extract regional mean time series
    for f in file_list:
        print(f"  Processing {os.path.basename(f)}...")
        ds = xr.open_dataset(f)
        sst = ds[sst_var]

        # Subset the region of interest
        sst_sub = sst.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min_mod, lon_max_mod)
        )

        # Spatial mean over the selected box
        ts = sst_sub.mean(dim=("lat", "lon"))

        # Load values into memory, then close file to free resources
        ts = ts.load()
        ds.close()

        ts_list.append(ts)

    # Concatenate yearly time fragments into a continuous series
    sst_ts = xr.concat(ts_list, dim="time")

    # Convert Kelvin → Celsius if needed (robust check on mean value)
    if float(sst_ts.mean()) > 100:
        sst_ts = sst_ts - 273.15

    # Ensure time axis is strictly increasing
    sst_ts = sst_ts.sortby("time")

    print(
        f"\nFinal SST TS: {str(sst_ts.time.values[0])} → "
        f"{str(sst_ts.time.values[-1])} "
        f"({sst_ts.time.size} days)"
    )

    return sst_ts


# ====================================================
# 3) MARINE HEATWAVE COMPUTATION
# ====================================================

def compute_climatology_and_threshold(
    sst_ts: xr.DataArray,
    baseline_start: str,
    baseline_end: str,
    percentile: float = 0.9
):
    """
    Compute daily climatological mean SST and the percentile-based
    MHW threshold over a baseline period.

    Parameters
    ----------
    sst_ts : xarray.DataArray
        Full SST time series (°C).
    baseline_start, baseline_end : str
        ISO dates defining the baseline period (e.g. "1984-01-01").
    percentile : float, optional
        Percentile for the MHW threshold (e.g. 0.9 for 90th).

    Returns
    -------
    clim_full : xarray.DataArray
        Climatological mean SST for each day of the full record.
    thresh_full : xarray.DataArray
        Percentile threshold time series aligned with `sst_ts`.

    Notes
    -----
    - Uses day-of-year (DOY) grouping, ignoring leap-day issues.
    - If the baseline is very short (< ~10 years), a warning is printed.
    """
    sst_base = sst_ts.sel(time=slice(baseline_start, baseline_end))

    if sst_base.time.size < 365 * 10:
        print("Warning: baseline period is quite short; "
              "climatology/threshold may be noisy.")

    doy = sst_base["time"].dt.dayofyear

    # DOY climatology and threshold from baseline
    clim_mean = sst_base.groupby(doy).mean("time")
    clim_thresh = sst_base.groupby(doy).quantile(percentile, dim="time")

    clim_mean = clim_mean.rename({"dayofyear": "doy"})
    clim_thresh = clim_thresh.rename({"dayofyear": "doy"})

    # Map DOY climatology/threshold back to full time axis
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
    Detect marine heatwave (MHW) events based on a threshold time series.

    Parameters
    ----------
    sst_ts : xarray.DataArray
        SST time series (°C).
    threshold_ts : xarray.DataArray
        Threshold time series (same shape as sst_ts).
    min_length : int, optional
        Minimum number of consecutive days above threshold
        to qualify as an event.

    Returns
    -------
    pandas.DataFrame
        One row per event, with columns:
        - start (datetime64)
        - end (datetime64)
        - duration_days
        - max_intensity_degC
        - mean_intensity_degC

    Notes
    -----
    - Intensity is defined as SST − threshold.
    - Short gaps and merging rules can be extended if needed.
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
            # Start a new potential MHW
            in_event = True
            start_idx = i
        elif not is_above and in_event:
            # End of an event candidate
            end_idx = i - 1
            duration = end_idx - start_idx + 1

            if duration >= min_length:
                # Extract event-specific SST and threshold
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

            # Reset event flag
            in_event = False
            start_idx = None

    # Handle the case where the series ends while still in an event
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
    Aggregate MHW events by calendar year (based on event start date).

    Parameters
    ----------
    events_df : pandas.DataFrame
        Event-level table returned by `detect_marine_heatwaves`.

    Returns
    -------
    pandas.DataFrame
        One row per year, including:
        - n_events
        - total_mhw_days
        - max_intensity_degC
        - mean_intensity_degC
        - longest_event_days
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
    Plot full SST time series with climatology, threshold,
    and shaded marine heatwave events.

    Parameters
    ----------
    sst_ts : xarray.DataArray
        SST time series (°C).
    clim_ts : xarray.DataArray
        Climatological mean SST aligned with sst_ts.
    thresh_ts : xarray.DataArray
        Percentile threshold aligned with sst_ts.
    events_df : pandas.DataFrame
        Event-level table from `detect_marine_heatwaves`.
    out_path : str
        Path where the PNG figure will be saved.
    region_name : str, optional
        Label used in the plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    time_vals = sst_ts.time.values

    ax.plot(time_vals, sst_ts, label="SST (°C)", linewidth=1)
    ax.plot(time_vals, clim_ts, label="Climatology (baseline mean)", linestyle="--")
    ax.plot(
        time_vals,
        thresh_ts,
        label=f"{int(PERCENTILE * 100)}th percentile threshold",
        linestyle=":"
    )

    # Shade each detected MHW event
    for i, ev in events_df.iterrows():
        ax.axvspan(
            ev["start"],
            ev["end"],
            alpha=0.2,
            color="tab:blue",
            label="MHW event" if i == 0 else None
        )

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
# 4) SIGNIFICANCE TESTS (TRENDS + CHANGE-POINTS) & PLOTS
# ====================================================

def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (helper for p-values)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_kendall_test(series: pd.Series) -> dict:
    """
    Mann–Kendall non-parametric trend test and Kendall's tau.

    Parameters
    ----------
    series : pandas.Series
        Time series indexed by time (e.g. years).

    Returns
    -------
    dict
        Keys: n, mk_tau, mk_S, mk_varS, mk_Z, mk_p.
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

    # S statistic (pairwise sign comparison)
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[j] > y[i]:
                S += 1
            elif y[j] < y[i]:
                S -= 1

    # Tie correction for variance
    unique, counts = np.unique(y, return_counts=True)
    tie_term = 0
    for c in counts:
        if c > 1:
            tie_term += c * (c - 1) * (2 * c + 5)

    varS = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

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
    Sen's slope estimator (median of all pairwise slopes).

    Parameters
    ----------
    series : pandas.Series
        Time series (e.g. yearly metric).
    x_years : pandas.Series
        Time coordinate expressed as numeric years, indexed
        consistently with `series`.

    Returns
    -------
    float
        Slope per unit of x_years (usually per year).
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
    Ordinary Least Squares (OLS) linear trend: y = a + b * x.

    Parameters
    ----------
    series : pandas.Series
        Time series values.
    x_years : pandas.Series
        Numeric time axis (e.g. years), with same index as `series`.

    Returns
    -------
    dict
        Keys: lin_slope, lin_intercept, lin_R2, lin_p.
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
            # Use normal approximation for p-value
            p = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))

    return {
        "lin_slope": float(b),
        "lin_intercept": float(a),
        "lin_R2": float(R2),
        "lin_p": float(p),
    }


def pettitt_test(series: pd.Series) -> dict:
    """
    Pettitt non-parametric change-point test.

    Parameters
    ----------
    series : pandas.Series
        Time series (e.g. yearly metric) sorted by time.

    Returns
    -------
    dict
        Keys: pettitt_index, pettitt_x, pettitt_p.
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

    # Approximate p-value for Pettitt statistic
    p = 2.0 * math.exp((-6.0 * (K_max ** 2)) / (n**3 + n**2))

    # Get corresponding time coordinate (e.g. year)
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
    Generic utility to plot a yearly metric with:
    - raw values,
    - linear trend line,
    - Pettitt change-point (if detected),
    - annotation of slope and p-values.

    Parameters
    ----------
    years : array-like
        Year values (x-axis).
    values : array-like
        Metric values (y-axis).
    metric_label : str
        Label for axis and legend.
    out_name : str
        File name of the output PNG (relative to PLOTS_DIR).
    lin_slope, lin_intercept : float
        Parameters of the linear fit.
    lin_p : float
        p-value of the linear trend.
    mk_p : float
        p-value of Mann–Kendall test.
    pettitt_x :
        Location of Pettitt change-point (e.g. year) or None.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Time series
    ax.plot(years, values, "-o", label=metric_label)

    # Linear trend line (if valid)
    if not math.isnan(lin_slope) and not math.isnan(lin_intercept):
        x_line = np.array([years.min(), years.max()])
        y_line = lin_intercept + lin_slope * x_line
        ax.plot(x_line, y_line, "--", label="Linear trend")

    # Pettitt change-point as vertical line
    if pettitt_x is not None:
        try:
            cx = int(pettitt_x)
            ax.axvline(cx, color="red", linestyle=":", label=f"Pettitt change ({cx})")
        except Exception:
            pass

    ax.set_xlabel("Year")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} – Trend & Change-point")

    # Text box with key statistics
    txt = (
        f"slope = {lin_slope:.4g} /yr\n"
        f"lin p = {lin_p:.3g}\n"
        f"MK p = {mk_p:.3g}"
    )
    ax.text(
        0.01,
        0.99,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

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
    Perform trend and change-point analysis on:
    1) Annual SST metrics (mean, max, min).
    2) Yearly marine heatwave metrics from `summary_df`.

    For each metric, the following are computed:
    - Mann–Kendall test (non-parametric trend).
    - Sen's slope (robust trend estimate).
    - OLS linear trend (slope, R², p-value).
    - Pettitt change-point test.

    Results
    -------
    - tables/trend_significance_sst.csv
    - tables/trend_significance_mhw_metrics.csv
    - plots/trend_*.png
    """
    # ---- 1. Annual SST metrics ----
    # Use "YE" (year-end) resampling to avoid deprecation warnings.
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

        # Plot for this SST metric
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
        # Series indexed by year (e.g. 1985, 1986, ...)
        series = summary_df.set_index("year")[metric_name]
        years_idx = series.index.to_series()

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

        # Plot for this MHW metric
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
# 5) MAIN WORKFLOW
# ====================================================

def main():
    """
    Orchestrate the full marine heatwave analysis:
      - ensure raw data is present (download if needed),
      - build regional SST time series,
      - derive climatology and threshold,
      - detect and summarise MHW events,
      - produce plots and significance tests.
    """
    # Use fixed data/raw folder inside the repository
    raw_dir = RAW_DATA_DIR
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Using raw data folder: {os.path.abspath(raw_dir)}")

    # 1) Ensure all requested years are available locally
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

    # 7) Trend and change-point analysis
    print("Running trend & change-point significance tests (with plots)...")
    run_significance_tests(sst_ts, summary_df)

    print("Done.")


if __name__ == "__main__":
    main()
