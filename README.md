# This repository contains the developed Python3 scripts to perform S-ESN

### Step 1. Download Data

1. The wind speed residual data at 3,173 knots

The wind speed residual data $Y_t(\mathbf{s}^\ast)$ at 3,173 knots from 2013 to 2016 (Feb. 29 in the leap year 2016 is removed) is stored as "wind_residual.nc" (netCDF4 file, 849 MB). Please download it via https://repository.kaust.edu.sa/handle/10754/667127 and save it to the directory "./data".

(Optional) If you would like to see the S-ESN forecasts at all the 53,333 locations over Saudi Arabia, the two following datasets are needed.

2. The wind speed residual data at all 53,333 locations
The wind speed residual data $Y_t(\mathbf{s})$ at 53,333 knots from 2013 to 2016 (Feb. 29 in the leap year 2016 is removed) is stored as "wind_residual_all_locations.nc" (netCDF4 file, 14 GB). Please download it via https://repository.kaust.edu.sa/handle/10754/667127 and save it to the directory "./data".

  - **Due to data size limitation in the repository, the files to be downloaded are actually:**
    - wind_residual_all_locations.nc.partaa (5 GB)
    - wind_residual_all_locations.nc.partab (5 GB)
    - wind_residual_all_locations.nc.partac (4 GB)
  - Download all of them, save them to "./data" and run the following command to merge
  ```bash
  cat  wind_residual_all_locations.nc.part* > wind_residual_all_locations.nc
  ```


3. The precomputed kriging weight
The precomputed kriging weight for the spatial interpolation is stored as "kriging_weight.nc" (netCDF4 file, 16 GB). Please download it via https://repository.kaust.edu.sa/handle/10754/667127 and save it to the directory "./data".

  - **Due to data size limitation in the repository, the files to be downloaded are actually:**
    - kriging_weight.nc.partaa (5 GB)
    - kriging_weight.nc.partab (5 GB)
    - kriging_weight.nc.partac (5 GB)
    - kriging_weight.nc.partad (134 MB)
  - Download all of them, save them to "./data" and run the following command to merge
  ```bash
  cat  kriging_weight.nc.part* > kriging_weight.nc
  ```
  
(Optional) If you would like to analyze the generated wind power, the following dataset is needed.

4. Related information about the turbines in the 75 wind farms
Wind farm related information is stored as "wind_farm_data.nc" (netCDF4 file, 24K). Please download it via https://repository.kaust.edu.sa/handle/10754/667127 and save it to the directory "./data".

### Step 2. Run the Jupyter Notebook Wind_Forecast.ipynb
The Jupyter Notebook *Wind_Forecast.ipynb* contains the script to perform the S-ESN, where necessary comments are provided.

### List of files (directories)and description

| File or Directory | Description |
| :-------------:   |:-------------:|
| data | Directory containing needed data|
| src | Directory containing python source code for ESN |
| data/wind_residual.nc | Data of wind residuals at knots, to be downloaded |
| data/wind_residual_all_locations.nc | Data of wind residuals at all locations, to be downloaded |
| data/kriging weight.nc | Data of precomputed kriging weight, to be downloaded |
| data/wind_farm_data.nc | Data of wind farm related information, to be downloaded |
| src/model.py | Class of ESN model  | 
| src/index.py | Wrapper for index |
| src/data.py  | Wrapper for data  |
| Wind_Forecast.ipynb | Jupyter Notebook to perform the S-ESN |