# %%
from folium.plugins import HeatMap
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from branca.colormap import linear
import branca
from folium.plugins import TimestampedGeoJson
from datetime import datetime

# %%
# Load the data
data = pd.read_csv('merged_data_cleaned.csv')
data['time'] =  pd.to_datetime(data['time'])

selected_date = pd.to_datetime("2024-02-19")

df_filtered = data[data['time'].dt.date == selected_date.date()].copy()

features = ['pressure', 'temp', 'rain', 'humidity', 'wind1', 'wind2', 'Month', 'Hour', 'PM2.5_lag_3h', 'PM2.5_lag_6h', 'PM2.5_lag_9h']
normalizer = MinMaxScaler()

# Normalize and standardize with the same scalers used in training
X = df_filtered[features]
X_normalized = normalizer.fit_transform(X)

# Load model and make predictions
best_model = load('xgb_model.joblib')
predicted_pm25 = best_model.predict(X_normalized)
df_filtered['Predicted PM2.5'] = predicted_pm25

# Determine air quality category
def air_quality_category(conc):
    if conc <= 12:
        return 'Good', 'green'
    elif 12.1 <= conc <= 35.4:
        return 'Moderate', 'yellow'
    elif 35.5 <= conc <= 55.4:
        return 'Unhealthy for sensitive', 'orange'
    elif 55.5 <= conc <= 150.4:
        return 'Unhealthy', 'red'
    elif 150.5 <= conc <= 250.4:
        return 'Very unhealthy', 'purple'
    else:
        return 'Hazardous', 'brown'

# Initialize Folium map (center on station coordinates)
pm_station = [21.02357910187751, 105.85387723079896]
map_pm25 = folium.Map(location=pm_station, zoom_start=13, zoom_control=False)

# Add a marker for the station
folium.Marker(
    location=pm_station,
    popup="US Embassy in Hanoi (19/21 Hai Ba Trung St., Hoan Kiem Dist., Hanoi)",
    icon=folium.Icon(color="blue")
).add_to(map_pm25)

"""
# Create a circle for the 1km radius around the station
folium.Circle(
    location=pm_station,
    radius=1000,  # 1000 meters = 1km
    color="blue",
    fill=False,
    weight=2,
).add_to(map_pm25)
"""
features = []
for index, row in df_filtered.iterrows():
    timestamp = row['time']
    conc = row['Predicted PM2.5']
    category, color = air_quality_category(conc)

    # Convert Timestamp to string in the desired format
    timestamp_str = timestamp.strftime('%Y/%m/%d %H:%M:%S')  # Adjust format as needed

    features.append({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': pm_station[::-1],  # Folium uses [lon, lat]
        },
        'properties': {
            'time': timestamp_str,  # Use the string representation
            'style': {
                'color': color,
                'fillColor': color,
                'fillOpacity': 0.5,  
                'weight': 0  
            },
            'icon': 'circle',
            'iconstyle': {
                'radius': 56,  # Radius for the PM2.5 circle within the 1km radius
            },
            'popup': f'{timestamp_str} - PM2.5: {conc:.2f}, Air Quality: {category}'
        }
    })

TimestampedGeoJson(
    {'type': 'FeatureCollection', 'features': features},
    period='PT3H',
    duration='PT1H',
    add_last_point=True,
    auto_play=True,
    loop=True,
    transition_time=1000,
    max_speed=50,
    loop_button=True,
    date_options='YYYY/MM/DD HH:mm:ss',
).add_to(map_pm25)

def get_info_html(conc, timestamp):
    category, color = air_quality_category(conc)
    timestamp_str = timestamp.strftime("%m/%d/%Y %H:%M")
    date_str = datetime.strptime(timestamp_str, '%m/%d/%Y %H:%M').strftime('%d/%m/%Y %H:%M')
    return f"""
    <div id="info-box" style="position: fixed; top: 10px; left: 10px; width: 250px; height: 45px; 
                background-color: white; border: 2px solid grey; z-index: 9999; padding: 10px;">
        <strong>Station: US Embassy in Hanoi</strong><br>
    </div>
    """

# Add a custom function to add the info box to the map
def add_info_box(map_obj, conc, timestamp):
    info_html = get_info_html(conc, timestamp)
    map_obj.get_root().html.add_child(folium.Element(info_html))

# Add initial info box
initial_conc = df_filtered['Predicted PM2.5'].iloc[0]
initial_timestamp = df_filtered['time'].iloc[0]
add_info_box(map_pm25, initial_conc, initial_timestamp)

# Add PM2.5 indicator bar
indicator_bar_html = """
<div style="position: fixed; top: 10px; right: 10px; width: 200px; height: auto; 
             background-color: white; border: 2px solid grey; z-index: 9999; padding: 10px;">
    <div style="background-color: green; width: 100%; padding: 5px; box-sizing: border-box;"><strong>Good</strong></div>
    <div style="background-color: yellow; width: 100%; padding: 5px; box-sizing: border-box;"><strong>Moderate</strong></div>
    <div style="background-color: orange; width: 100%; padding: 5px; box-sizing: border-box;"><strong>Unhealthy for Sensitive</strong></div>
    <div style="background-color: red; color: white; width: 100%; padding: 5px; box-sizing: border-box;"><strong>Unhealthy</strong></div>
    <div style="background-color: purple; color: white; width: 100%; padding: 5px; box-sizing: border-box;"><strong>Very Unhealthy</strong></div>
    <div style="background-color: brown; color: white; width: 100%; padding: 5px; box-sizing: border-box;"><strong>Hazardous</strong></div>
</div>
"""

map_pm25.get_root().html.add_child(folium.Element(indicator_bar_html))
map_pm25.add_child(folium.plugins.ScrollZoomToggler())  # Add a button to toggle zoom on/off

# JavaScript to disable zoom 
disable_zoom_js = """
map.scrollWheelZoom.disable();
map.doubleClickZoom.disable();
map.touchZoom.disable();
"""

map_pm25._repr_html_()  # This is needed for the JS to be executed
map_pm25.get_root().script.add_child(folium.Element(disable_zoom_js))

# Save the animated map
map_pm25.save('static/index.html')



