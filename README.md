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

```
root/
│
├── scripts/
│   └── mhw_detection_noaa_oisst.py        # Main Python script
│
├── data/
│   ├── raw/                               # NetCDF files auto-downloaded
│   └── processed/                         # Cleaned & subsetted datasets
│
├── outputs/
│   ├── figures/                           # Plots: SST, anomalies, MHWs
│   └── tables/                            # Event metadata, climatology
│
└── README.md
```

---

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install numpy pandas xarray netCDF4 matplotlib tqdm requests tk
```

Or using Conda:

```bash
conda install -c conda-forge numpy pandas xarray netcdf4 matplotlib tqdm requests
```

---

## 📍 Region of Interest (Gulf of Naples)

```python
LAT_MIN, LAT_MAX = 40.3, 41.1
LON_MIN, LON_MAX = 13.8, 14.8
```

Modify these values to analyse any region worldwide.

---

## 📅 Climatology / Baseline Period

```python
BASELINE_START = "1984-01-01"
BASELINE_END   = "2013-12-31"
```

---

## 🌡 Marine Heatwave Definition (Hobday et al., 2016)

A marine heatwave occurs when:

- SST exceeds the **90th percentile**
- For **≥ 5 consecutive days**
- Short interruptions (<2 days) are merged

Parameters:

```python
PERCENTILE = 0.9
MIN_EVENT_LENGTH = 5
```

---

## 🚀 How to Run the Script

From terminal:

```bash
python mhw_detection_noaa_oisst.py
```

Workflow:

1. Download required NOAA OISST files (1984–2024)  
2. Merge datasets  
3. Subset to the region of interest  
4. Compute climatology and thresholds  
5. Detect marine heatwaves  
6. Export figures + tables  

---

## 📊 Outputs

### 1️⃣ Figures (`outputs/figures/`)
- SST time series  
- Climatology  
- Anomalies  
- Marine heatwave periods  

### 2️⃣ Tables (`outputs/tables/`)
Event metadata CSV with:

| Start | End | Duration | Mean Intensity | Max Intensity | Cumulative Intensity |

### 3️⃣ Processed NetCDF files (`data/processed/`)
- Subset SST  
- Climatology & threshold  
- Dataset with MHW flags  

---

## 📥 NOAA Data Source

Daily SST files are downloaded from:

```
https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/
```

Saved to:

```
data/raw/noaa_oisst/
```

Existing files are skipped automatically.

---

## ✏️ Customization

You can modify:

- Region coordinates  
- Climatology baseline  
- MHW threshold + minimum event duration  
- Output directory structure  
- Plotting options  

---

## 📚 References

- Hobday et al. (2016). *A hierarchical approach to defining marine heatwaves.* Progress in Oceanography.  
- NOAA OISST v2.1 User Guide  
- Reynolds et al. (2007). *Daily high-resolution blended analyses for sea surface temperature.*

---

## 🤝 Contributing

Contributions are welcome. Future additions may include:

- Multi-region MHW comparison  
- Trend detection (Theil–Sen, Mann–Kendall)  
- Parallelised NOAA downloading  
- Additional visualisation modules  

---

## 📜 License

MIT License — free to use, modify, and distribute.
