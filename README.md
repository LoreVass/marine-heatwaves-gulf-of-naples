# Marine Heatwaves in the Gulf of Naples (2014–2024)

This project analyzes **marine heatwaves (MHWs)** in the Gulf of Naples using 
daily sea surface temperature (SST) data from **NOAA OISST v2.1**.  
A Python script processes multiple years of SST data, computes a regional 
climatology, detects heatwave events following the Hobday et al. (2016) 
definition, and generates:

- A **detailed event catalogue** (per-event start/end, duration, intensity)
- A **yearly summary** (number of events, total MHW days, max intensity)
- A **time-series plot** with climatology, threshold, and shaded MHW periods
- A **Jupyter Notebook** for visualization and interpretation

The repository demonstrates geospatial data handling, climatology calculation, 
and event detection for environmental and marine science applications.

---

## 🔍 Study Area

**Region:** Gulf of Naples, Tyrrhenian Sea (Central Mediterranean)

Bounding box:
- **Latitude:** 40.3°N → 41.1°N  
- **Longitude:** 13.8°E → 14.8°E  

NOAA OISST provides NaN over land, so only ocean SST values are used.

---

## 🌡️ Methodology (Hobday et al., 2016)

A marine heatwave occurs when:

1. SST exceeds the **90th percentile** of the daily climatology  
2. For **≥ 5 consecutive days**  
3. Climatology baseline used here: **2014–2019**  
   (earliest available SST years)

**Processing steps:**
1. Load multi-year OISST netCDF files  
2. Subset to the Gulf of Naples box  
3. Compute daily climatology + 90th percentile threshold  
4. Detect heatwave events  
5. Generate yearly statistics  
6. Save tables + time-series plot  

---

## 📂 Repository Structure

marine-heatwaves-gulf-of-naples/
│
├── notebooks/
│ └── mhw_gulf_of_naples.ipynb
│
├── scripts/
│ └── main.py
│
├── tables/
│ ├── marine_heatwaves_events.csv
│ └── marine_heatwaves_yearly_summary.csv
│
├── plots/
│ └── sst_mhw_timeseries.png
│
├── data/
│ └── README.md
│
├── requirements.txt
└── README.md


---

## 📊 Summary of Results (2014–2024)

Key findings for the Gulf of Naples:

- **Strong rise in heatwave persistence**  
  → from ~90 days in 2014 → **219 days in 2024**

- **Higher intensity extremes**  
  → anomalies > **2.2°C** after 2022

- **Longer events**  
  → maximum duration grew from ~10–20 days to **45–63 days** post-2022

- **Multi-season heatwaves**  
  → events occur in winter, spring, summer, autumn

The Gulf of Naples appears to have entered a **persistent marine heatwave regime**.

---

## ▶️ Running the Analysis

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run the automated detector

python scripts/main.py

You will be asked to select a folder containing `.nc` files.

### 3. Explore the results in the Jupyter Notebook

jupyter notebook notebooks/mhw_gulf_of_naples.ipynb


---

## 📖 References

- Hobday et al. (2016), *A hierarchical approach to defining marine heatwaves*  
- NOAA OISST v2.1 dataset  
- Copernicus Mediterranean Ocean Monitoring reports  

---

## 📜 License
MIT License.

