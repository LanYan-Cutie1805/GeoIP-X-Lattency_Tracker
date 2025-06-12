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
import subprocess
import re
import scipy.stats as stats

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

with open("ip_list.txt", "r") as file:
    IP_LIST = [line.strip() for line in file if line.strip()]


# Load the GeoIP database
reader = geoip2.database.Reader('GeoLite2-City.mmdb')
data = []

print()
print("GeoIP is working")

# KERJAAN WONG EDAN - counting trace route hops
def get_hop_count(ip):
    try:
        result = subprocess.run(['traceroute', '-d', '-h', '30', ip], capture_output=True, text=True, timeout=20)
        hops = [line for line in result.stdout.splitlines('\n') if line.strip().startswith(str(len(hops)+ 1 ))]
        trace_output = result.stdout.splitlines()
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
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_text,
        
    ).add_to(marker_cluster)
             
# Save the map to an HTML file
m.save('ip_map.html')


# Latency VS distance calculation
df = pd.DataFrame(data).dropna()
print(df)

# Pearson correlation (Data Analysis)
correlation = df['Latency_ms'].corr(df['Distance_km'])
print()
print(f"Pearson correlation between Latency and Distance: {round(correlation, 3)}")

# Regional outlier detection (within country)
def detect_outliers_zscore(group, threshold=1.95):
    mean = group['Latency_ms'].mean()
    std = group['Latency_ms'].std()
    group['z_score'] = (group['Latency_ms'] - mean) / std
    group['Is_Outlier'] = np.abs(group['z_score']) > threshold
    return group

df_no_nan = df.dropna(subset=['Latency_ms', 'Country'])

# Apply z-score outlier detection
df_outliers = df_no_nan.groupby('Country', group_keys=False).apply(detect_outliers_zscore)

regional_outliers = df_outliers[df_outliers['Is_Outlier'] == True]
print("\nRegional Outliers Detected:")
print(regional_outliers[['IP', 'Country', 'City', 'Latency_ms', 'Distance_km']])

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

# QQ plot for Latency vs Distance
stats.probplot(df['Latency_ms'], dist="norm", plot=plt)
plt.title('QQ Plot of Latency')
plt.show()


# latency base on countries
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Country', y='Latency_ms')
plt.xticks(rotation=45)
plt.title('Latency Distribution by Country')
plt.grid()
plt.show()