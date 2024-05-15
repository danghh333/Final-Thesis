# %%
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import glob


lat_han = 21.02357910187751
lon_han = 105.85387723079896

all_data = []

for file in glob.glob('*.nc'):
    print(f"Processing: {file}")
    with Dataset(file, 'r') as data:
        # Main Variables
        pressure = data.variables['pmsl'][:]
        temp = data.variables['t2m'][:]
        humidity = data.variables['q2m'][:]
        rain = data.variables['rain'][:]
        wind1 = data.variables['u10m'][:]
        wind2 = data.variables['v10m'][:]

        # Time convert
        time = data.variables['time'][:]
        time_units = data.variables['time'].units
        time_datetime = num2date(time, units=time_units)

        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]


        # squared diff of lat and lon
        sq_diff_lat = (lat - lat_han)**2
        sq_diff_lon = (lon - lon_han)**2

        # Identifying the index of the minimum value for lon and lat
        min_index_lat = sq_diff_lat.argmin()
        min_index_lon = sq_diff_lon.argmin()

        file_data = pd.DataFrame({
            'time': time_datetime,
            'pressure': pressure[:, min_index_lat, min_index_lon],
            'temp': temp[:, min_index_lat, min_index_lon],
            'humidity': humidity[:, min_index_lat, min_index_lon],
            'rain': rain[:, min_index_lat, min_index_lon],
            'wind1': wind1[:, min_index_lat, min_index_lon],
            'wind2': wind2[:, min_index_lat, min_index_lon]
        })
        all_data.append(file_data) 

combined_data = pd.concat(all_data)
combined_data.to_csv('weather_data.csv', index=False)


