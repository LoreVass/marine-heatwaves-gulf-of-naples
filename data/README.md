## 📂 Data Instructions

This folder is intentionally empty.

To run the full marine heatwave detector using raw SST data, download the **NOAA OISST v2.1 Daily SST (AVHRR-Only)** NetCDF files from:

🔗 https://www.ncei.noaa.gov/products/optimum-interpolation-sst

---

### **Required years**
- **1984–2024** (full historical record for long-term analysis)

If you plan to compute a climatology baseline within this window (e.g., **1984–2013**, or any 30-year period), make sure the required years are included.

---

### **📁 How to Organize the Data**

Place all `.nc` files inside the following directory:

```
data/raw/
```

The script **main.py** will automatically detect and load all NetCDF files using a glob pattern.

---

### **📦 Example Directory Structure**

```
marine-heatwaves-gulf-of-naples/
│
├── data/
│   ├── raw/
│   │   ├── sst.day.mean.1984.nc
│   │   ├── sst.day.mean.1985.nc
│   │   ├── sst.day.mean.1986.nc
│   │   ├── ...
│   │   ├── sst.day.mean.2023.nc
│   │   └── sst.day.mean.2024.nc
│
└── README.md
```

---

### **📄 Required Data Format**

- File type: **NetCDF (.nc)**
- Temporal resolution: **Daily**
- Required variables:
  - `sst`
  - `lat`
  - `lon`
  - `time`

---

### ⚠️ Note

Raw SST data is **not included** in this repository because of NOAA licensing restrictions and file size.
