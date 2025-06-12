# GeoIP-X-Lattency_Tracker
This project visualizes the latency and hop count of multiple IP addresses using GeoIP data and network traceroutes. It plots geolocated IPs on an interactive map and analyzes relationships between geographic distance, latency, and traceroute hop counts using correlation and statistical techniques like Pearson correlation and IQR outlier detection.


## 📌 Features

- Geolocation of IP addresses using MaxMind GeoLite2.
- Latency measurement via ICMP ping.
- Hop count estimation using `traceroute`.
- Interactive map with marker clustering via Folium.
- Correlation analysis between distance, latency, and hop count.
- Outlier detection using z-score or IQR methods.

---

## 🚀 How to Run
1. Install the required library
pip install -r requirements.txt

2. Download the MaxMind GeoLite2 Database
Register at https://www.maxmind.com/en/geolite2/signup to download the free GeoLite2 City database.
Extract the .mmdb file and rename it (or ensure it's named) GeoLite2-City.mmdb.
Place it in the root directory of the project.

3. The IP List is on the ip_list.txt
Feel free to modify those
