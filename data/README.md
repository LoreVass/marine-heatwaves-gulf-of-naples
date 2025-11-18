# Data Instructions

This folder is intentionally empty.

To run the full marine heatwave detector using raw SST data, download
NOAA OISST v2.1 **Daily SST (AVHRR-Only)** netCDF files from:

https://www.ncei.noaa.gov/products/optimum-interpolation-sst

### Required years:
2014–2024  
(If a 2010–2019 baseline is desired, download 2010–2013.)

### How to organise the data:
Place all `.nc` files into a folder such as:

data/raw/


The script `main.py` will automatically read  
**all .nc files in the selected folder** using a glob pattern.

Example:

marine-heatwaves-gulf-of-naples/
│
├── data/
│ ├── raw/
│ │ ├── sst.day.mean.2014.nc
│ │ ├── sst.day.mean.2015.nc
│ │ └── ...
│ └── README.md


### Format required:
- NetCDF (`.nc`)
- Daily temporal resolution
- Variables: `sst`, `lat`, `lon`, `time`

This repository does **not** include raw SST data due to NOAA licensing and size.


