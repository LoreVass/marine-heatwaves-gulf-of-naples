# 🌊 Marine Heatwaves Detection in the Gulf of Naples  
### NOAA OISST v2.1 Daily SST (1984–2024)

This repository contains a complete Python workflow for:

- 📥 Downloading **daily NOAA OI SST v2.1** data (1984–2024) automatically  
- 🗺 Subsetting SST to a **region of interest** (Gulf of Naples)  
- 📈 Computing **climatology and heatwave thresholds** based on a user-defined baseline  
- 🔥 Detecting **marine heatwaves (MHW)** using the Hobday et al. (2016) definition  
- 📊 Exporting results, figures, and processed datasets for downstream analysis  

This project is intended for environmental data science, marine ecology, and long-term SST monitoring.

---

## 📁 Repository Structure

root/
│
├── scripts/
│ └── mhw_detection_noaa_oisst.py # Main Python script
│
├── data/
│ ├── raw/ # NetCDF files auto-downloaded
│ └── processed/ # Cleaned & subsetted datasets
│
├── outputs/
│ ├── figures/ # Plots: SST, anomalies, MHWs
│ └── tables/ # Event metadata, climatology
│
└── README.md

---

## 🔧 Requirements

Install all dependencies:

```bash
pip install numpy pandas xarray netCDF4 matplotlib tqdm requests tk
```
```Or via Conda:

conda install -c conda-forge numpy pandas xarray netcdf4 matplotlib tqdm requests
```
📍 Region of Interest (Gulf of Naples)

LAT_MIN, LAT_MAX = 40.3, 41.1
LON_MIN, LON_MAX = 13.8, 14.8


These values can be modified to process any region globally.
📅 Climatology / Baseline Period

The baseline for threshold computation can be set here:

BASELINE_START = "1984-01-01"
BASELINE_END   = "2013-12-31"  # Default 30-year climatology

🌡 Marine Heatwave Definition (Hobday et al., 2016)

A marine heatwave occurs when:

SST exceeds the 90th percentile for

≥ 5 consecutive days

Short gaps < 2 days are merged

Parameters:

PERCENTILE = 0.9
MIN_EVENT_LENGTH = 5

🚀 How to Run the Script

From terminal / command prompt:

python mhw_detection_noaa_oisst.py


Workflow steps:

Download all required NOAA OISST files (1984–2024)

Merge into a single dataset

Subset to the ROI

Compute climatology and 90th percentile threshold

Detect marine heatwaves

Export plots & results

📊 Outputs
1️⃣ Figures (outputs/figures/)

SST time series

Seasonal climatology

Temperature anomalies

Highlighted MHW periods

2️⃣ Tables (outputs/tables/)

CSV containing:

| Start | End | Duration | Mean Intensity | Max Intensity | Cumulative Intensity |

3️⃣ Processed NetCDF files (data/processed/)

Subset SST

Climatology + threshold

Dataset with MHW flags

📥 NOAA Data Source

Daily SST files are downloaded from:

https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/


Downloaded files are stored in:

data/raw/noaa_oisst/


Files already present will be skipped automatically.

✏️ Customization

You can easily modify:

Region coordinates

Baseline climatology period

Marine heatwave parameters

Output directories

Plotting settings

📚 References

Hobday et al. (2016). A hierarchical approach to defining marine heatwaves. Progress in Oceanography.

NOAA OISST v2.1 User Guide.

Reynolds et al. (2007). Daily high-resolution blended analyses for sea surface temperature.

🤝 Contributing

Contributions are welcome. Ideas for improvements include:

Parallel NOAA downloading

Multi-region comparison

Trend detection (Theil–Sen, Mann–Kendall)

Additional visualizations

📜 License

MIT License — free to use, modify, and distribute.
