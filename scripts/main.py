import os
import glob
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog



# ================== CONFIGURATION ==================

# Region of interest (here: Gulf of Naples, but can be changed)
LAT_MIN, LAT_MAX = 40.3, 41.1
LON_MIN, LON_MAX = 13.8, 14.8

# Baseline period for climatology / threshold
BASELINE_START = "2014-01-01"
BASELINE_END   = "2019-12-31"

# Marine heatwave definition
PERCENTILE = 0.9        # 90th percentile
MIN_EVENT_LENGTH = 5    # minimum event duration (days)

# Output folders
PLOTS_DIR = "plots"
TABLES_DIR = "tables"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
# ====================================================

def select_data_folder():
    """
    Open a tkinter window to select the folder containing .nc files.
    Returns a folder path as a string.
    """
    root = tk.Tk()
    root.withdraw()  # hide the root window
    folder = filedialog.askdirectory(title="Select Folder with SST .nc Files")

    if not folder:
        raise ValueError("No folder selected.")

    return folder


def load_sst_timeseries_for_region(
    folder_pattern,
    lat_min, lat_max,
    lon_min, lon_max
) -> xr.DataArray:
    """
    Load multiple SST netCDF files and directly extract
    the area-averaged time series for a given region.

    This is much more memory-efficient than loading the full globe.
    """
    file_list = sorted(glob.glob(folder_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files found for pattern: {folder_pattern}")

    print("Files found:")
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
        raise ValueError("Could not find SST variable in first dataset. Check your netCDF files.")

    # Handle longitude convention using first file
    if float(first_ds.lon.max()) > 180:
        lon_min_mod = (lon_min + 360) % 360
        lon_max_mod = (lon_max + 360) % 360
    else:
        lon_min_mod, lon_max_mod = lon_min, lon_max

    ts_list = []

    # Loop over all files, but only keep the region + mean
    for f in file_list:
        ds = xr.open_dataset(f)
        sst = ds[sst_var]

        sst_sub = sst.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min_mod, lon_max_mod)
        )

        # Area-averaged SST for this file
        ts = sst_sub.mean(dim=("lat", "lon"))
        ts_list.append(ts)

    # Concatenate all time segments into one long time series
    sst_ts = xr.concat(ts_list, dim="time")

    # Kelvin -> Celsius if needed
    if float(sst_ts.mean()) > 100:
        sst_ts = sst_ts - 273.15

    # Ensure time is sorted
    sst_ts = sst_ts.sortby("time")

    return sst_ts



#def load_sst_dataset(folder_pattern):

    """
    Load multiple SST netCDF files (e.g., one per year) and merge them
    into a single xarray DataArray with dimensions (time, lat, lon).

    Parameters
    ----------
    folder_pattern : str
        Glob pattern matching all SST netCDF files (e.g. 'data/*.nc').

    Returns
    -------
    sst_all : xarray.DataArray
        Merged SST field in degrees Celsius.
    """
    file_list = sorted(glob.glob(folder_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files found for pattern: {folder_pattern}")

    print("Files found:")
    for f in file_list:
        print("  ", f)

    datasets = [xr.open_dataset(f) for f in file_list]

    # Detect SST variable from the first dataset
    first_ds = datasets[0]
    for cand in ["sst", "SST", "sea_surface_temperature", "analysed_sst"]:
        if cand in first_ds.data_vars:
            sst_var = cand
            break
    else:
        raise ValueError("Could not find SST variable in first dataset. "
                         "Check your netCDF files.")

    # Extract SST and concatenate along time
    sst_list = [ds[sst_var] for ds in datasets]
    sst_all = xr.concat(sst_list, dim="time")

    # Convert Kelvin → Celsius if needed (simple heuristic)
    if float(sst_all.mean()) > 100:
        sst_all = sst_all - 273.15

    # Ensure time is sorted
    sst_all = sst_all.sortby("time")

    return sst_all


def subset_region_mean(sst: xr.DataArray,
                       lat_min: float, lat_max: float,
                       lon_min: float, lon_max: float) -> xr.DataArray:
    """
    Subset SST to a lat/lon box and return the spatial mean time series.

    Returns
    -------
    sst_ts : xarray.DataArray
        1D time series of area-averaged SST.
    """
    # Handle 0–360 vs -180–180 longitude conventions
    if float(sst.lon.max()) > 180:
        lon_min_mod = (lon_min + 360) % 360
        lon_max_mod = (lon_max + 360) % 360
    else:
        lon_min_mod, lon_max_mod = lon_min, lon_max

    sst_sub = sst.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min_mod, lon_max_mod)
    )

    sst_ts = sst_sub.mean(dim=("lat", "lon"))
    return sst_ts


def compute_climatology_and_threshold(
    sst_ts: xr.DataArray,
    baseline_start: str,
    baseline_end: str,
    percentile: float = 0.9
):
    """
    Compute daily climatology and percentile-based threshold.

    Returns
    -------
    clim_full : xarray.DataArray
        Climatological mean SST mapped onto the full time axis.
    thresh_full : xarray.DataArray
        Percentile threshold mapped onto the full time axis.
    """
    sst_base = sst_ts.sel(time=slice(baseline_start, baseline_end))

    if sst_base.time.size < 365 * 2:
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
    Detect marine heatwave events where SST exceeds threshold for at
    least 'min_length' consecutive days.

    Returns
    -------
    events_df : pandas.DataFrame
        Table with one row per MHW event.
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

    events_df = pd.DataFrame(events)
    return events_df


def summarize_events_by_year(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate marine heatwave events by calendar year.

    Returns
    -------
    summary : pandas.DataFrame
        One row per year with counts and intensity metrics.
    """
    if events_df.empty:
        return pd.DataFrame()

    # Assign a 'year' based on the start date of each event
    events_df = events_df.copy()
    events_df["year"] = events_df["start"].dt.year

    # Total MHW days per event already encoded in duration_days
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
    out_path: str
):
    """
    Plot SST, climatology, threshold and highlight marine heatwave events.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    time_vals = sst_ts.time.values

    ax.plot(time_vals, sst_ts, label="SST (°C)", linewidth=1)
    ax.plot(time_vals, clim_ts, label="Climatology (baseline mean)", linestyle="--")
    ax.plot(time_vals, thresh_ts, label=f"{int(PERCENTILE * 100)}th percentile threshold", linestyle=":")

    # Shade MHW events
    for i, ev in events_df.iterrows():
        ax.axvspan(ev["start"], ev["end"], alpha=0.2,
                   color="tab:blue",
                   label="MHW event" if i == 0 else None)

    ax.set_title("Marine Heatwaves – Gulf of Naples")
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (°C)")
    ax.legend()
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    print("Select the folder containing your SST .nc files...")
    data_folder = select_data_folder()
    data_pattern = os.path.join(data_folder, "*.nc")

    print("Loading SST and extracting regional time series...")
    sst_ts = load_sst_timeseries_for_region(
        data_pattern,
        LAT_MIN, LAT_MAX,
        LON_MIN, LON_MAX
    )

    print("Computing climatology and threshold...")
    clim_ts, thresh_ts = compute_climatology_and_threshold(
        sst_ts,
        BASELINE_START,
        BASELINE_END,
        PERCENTILE
    )

    print("Detecting marine heatwaves...")
    events_df = detect_marine_heatwaves(
        sst_ts,
        thresh_ts,
        MIN_EVENT_LENGTH
    )

    events_csv_path = os.path.join(TABLES_DIR, "marine_heatwaves_events.csv")
    events_df.to_csv(events_csv_path, index=False)
    print(f"Saved events table to {events_csv_path}")

    print("Summarizing events by year...")
    summary_df = summarize_events_by_year(events_df)
    summary_csv_path = os.path.join(TABLES_DIR, "marine_heatwaves_yearly_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved yearly summary to {summary_csv_path}")
    print(summary_df)

    print("Plotting time series...")
    plot_path = os.path.join(PLOTS_DIR, "sst_mhw_timeseries.png")
    plot_time_series_with_events(sst_ts, clim_ts, thresh_ts, events_df, plot_path)

    print("Done.")



if __name__ == "__main__":
    main()
