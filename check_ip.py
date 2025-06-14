from ping3 import ping, verbose_ping
import geoip2.database
import pandas as pd
import folium
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from folium.plugins import MarkerCluster
import scipy.stats as stats
import subprocess
from branca.element import Template, MacroElement
import re

# Fetch my location coordinates
def get_my_coordinates():
    try:
        res = requests.get('http://ip-api.com/json/')
        data = res.json()
        lat = data.get('lat')
        lon = data.get('lon')
        city = data.get('city')
        country = data.get('country')
        print(f"Your coordinates: {lat}, {lon} in {city}, {country}")
        return lat, lon
    except Exception as e:
        print(f"Error fetching your coordinates: {e}")
        return None, None
my_lat, my_lon = get_my_coordinates()
MY_COORDINATES = (my_lat, my_lon)

with open("ipadd.txt", "r") as file:
    IP_LIST = [line.strip() for line in file if line.strip()]


# Load the GeoIP database
reader = geoip2.database.Reader('GeoLite2-City.mmdb')
data = []

print()
print("GeoIP is working")


# KERJAAN WONG EDAN - counting trace route hops
def get_hop_count(ip):
    try:
        result = subprocess.run(['tracert', '-d', '-h', '30', ip], capture_output=True, text=True, timeout=30)
        output = result.stdout.splitlines()
        hops = [line for line in output if re.match(r'^\s*\d+\s', line)]
        return len(hops)
    except Exception as e:
        print(f"Error running traceroute for {ip}: {e}")
        return None


for ip in IP_LIST:
    try:
        lattency = ping(ip, unit='ms')
        geo = reader.city(ip)
        location = (geo.location.latitude, geo.location.longitude)
        response = reader.city(ip)
        distance_km = geodesic(MY_COORDINATES, location).km
        hops = get_hop_count(ip)
        data.append({
            "IP": ip,
            "Latitude": location[0],
            "Longitude": location[1],
            "Country": geo.country.name,
            "City": geo.city.name,
            "Latency_ms": round(lattency, 2) if lattency is not None else None,
            "Distance_km": round(distance_km, 2),
            "Hop_count": hops
        })

    except Exception as e:
        print(f"Error processing IP {ip}: {e}")
    
reader.close()

df = pd.DataFrame(data)
print(df)
print("End of GeoIP")
print()


# visualize the results on a map
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=2)
marker_cluster = MarkerCluster().add_to(m)
df_clean = df.dropna(subset=['Latitude', 'Longitude'])


for _, row in df_clean.iterrows():
    popup_text = (
        f"IP: {row['IP']}<br>"
        f"Country: {row['Country']}<br>"
        f"City: {row['City']}<br>"
        f"Latency: {row['Latency_ms']} ms" if pd.notnull(row['Latency_ms']) else "Latency: N/A"
    )
    latency = row['Latency_ms']
    if pd.notnull(latency):
        if latency < 50:
            color = 'green'
        elif latency < 150:
            color = 'orange'
        else:
            color = 'red'
    else:
        color = 'grey'

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup_text
        
    ).add_to(marker_cluster)
             
# Save the map to an HTML file
m.save('ip_map.html')


# Latency VS distance calculation
df = pd.DataFrame(data).dropna()
print("Latency vs Distance DataFrame: The pingable addresses")
print(df)
print()
# save to csv
df.to_csv('PING-report.csv', index=False)


# Pearson correlation (Data Analysis)
correlation = df['Latency_ms'].corr(df['Distance_km'])
print()
print(f"Pearson correlation between Latency and Distance: {round(correlation, 2)}")

df_no_nan = df.dropna(subset=['Latency_ms', 'Country'])


# Use IQR
def detect_outliers_iqr(group):
    Q1 = group['Latency_ms'].quantile(0.25)
    Q3 = group['Latency_ms'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    group['Is_Outlier_IQR'] = (group['Latency_ms'] < lower_bound) | (group['Latency_ms'] > upper_bound)
    
    print(f"IQR: {IQR}")
    print(f"Q1: {lower_bound}")
    print(f"Q3: {upper_bound}")
    return group
detect_outliers_iqr(df)



df_corr = df.dropna(subset=['Latency_ms', 'Hop_Count'])
corr = df_corr['Latency_ms'].corr(df_corr['Hop_Count'])
print(f"\nPearson correlation between Latency and Hop Count: {round(corr, 3)}")

# Plotting the correlation between Latency and Hop Count
plt.figure(figsize=(10, 6))
sns.regplot(data=df_corr, x='Hop_Count', y='Latency_ms', scatter_kws={"s": 80}, line_kws={"color": "red"})
plt.title('Latency vs Hop Count')
plt.xlabel('Hop Count')
plt.ylabel('Latency (ms)')
plt.tight_layout()
plt.grid()


# Plotting the results
plt.figure(figsize=(10, 6)) 
sns.regplot(data=df, x='Distance_km', y='Latency_ms', scatter_kws={"s": 80}, line_kws={"color": "red"})
plt.title('Latency vs Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Latency (ms)')
plt.grid()
plt.tight_layout()
plt.show()


# latency boxplot base on countries
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Country', y='Latency_ms')
plt.xticks(rotation=45)
plt.title('Latency Distribution by Country')
plt.grid()
plt.show()


