import io
import json
import math
import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from flask import Flask, jsonify, render_template_string, request, send_file
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import schedule
from scipy.spatial import cKDTree

# ==========================================
# KONFIGURASI SISTEM (5 PROVINSI)
# ==========================================
DB_NAME = "hotspot_multi.db"
DEFAULT_NASA_KEY = "cf5d1449942e6c851eba10358106ea6f"

PROVINSI_BBOX = {
    "JAMBI": "101.15,-2.50,104.55,-0.75",
    "SUMSEL": "102.05,-4.95,106.15,-1.65",
    "JATIM": "110.85,-8.80,114.65,-6.75",
    "KALTENG": "110.75,-3.55,115.85,0.75",
    "KALTIM": "113.80,-2.45,119.05,2.85",
}

SENTINEL_POINTS = [
    # Jambi
    {
        "prov": "JAMBI",
        "kawasan": "TN Berbak (Pesisir Gambut)",
        "lat": -1.352,
        "lon": 104.102,
        "gambut": 1,
    },
    {
        "prov": "JAMBI",
        "kawasan": "HL Gambut Bram Itam",
        "lat": -1.120,
        "lon": 103.180,
        "gambut": 1,
    },
    {
        "prov": "JAMBI",
        "kawasan": "PBPH PT REKI (Hutan Harapan)",
        "lat": -2.210,
        "lon": 103.350,
        "gambut": 0,
    },
    # Sumatera Selatan
    {
        "prov": "SUMSEL",
        "kawasan": "PBPH Gambut OKI (Air Sugihan)",
        "lat": -3.350,
        "lon": 105.420,
        "gambut": 1,
    },
    {
        "prov": "SUMSEL",
        "kawasan": "SM Padang Sugihan",
        "lat": -2.950,
        "lon": 105.050,
        "gambut": 1,
    },
    {
        "prov": "SUMSEL",
        "kawasan": "TN Sembilang Pesisir",
        "lat": -2.150,
        "lon": 104.750,
        "gambut": 1,
    },
    # Jawa Timur
    {
        "prov": "JATIM",
        "kawasan": "TN Bromo Tengger Semeru",
        "lat": -7.940,
        "lon": 112.950,
        "gambut": 0,
    },
    {
        "prov": "JATIM",
        "kawasan": "TN Baluran (Savana Bekol)",
        "lat": -7.820,
        "lon": 114.420,
        "gambut": 0,
    },
    {
        "prov": "JATIM",
        "kawasan": "KPH Perhutani Bojonegoro",
        "lat": -7.280,
        "lon": 111.750,
        "gambut": 0,
    },
    # Kalimantan Tengah
    {
        "prov": "KALTENG",
        "kawasan": "TN Sebangau (Kubah Gambut)",
        "lat": -2.350,
        "lon": 113.850,
        "gambut": 1,
    },
    {
        "prov": "KALTENG",
        "kawasan": "Eks-PLG Blok A Pulang Pisau",
        "lat": -2.750,
        "lon": 114.250,
        "gambut": 1,
    },
    {
        "prov": "KALTENG",
        "kawasan": "TN Tanjung Puting",
        "lat": -2.950,
        "lon": 111.950,
        "gambut": 1,
    },
    # Kalimantan Timur
    {
        "prov": "KALTIM",
        "kawasan": "Tahura Bukit Soeharto / IKN",
        "lat": -0.980,
        "lon": 117.020,
        "gambut": 0,
    },
    {
        "prov": "KALTIM",
        "kawasan": "TN Kutai (Sangkima)",
        "lat": 0.250,
        "lon": 117.350,
        "gambut": 0,
    },
    {
        "prov": "KALTIM",
        "kawasan": "Konsesi Tambang Kutai Kartanegara",
        "lat": -0.450,
        "lon": 116.850,
        "gambut": 0,
    },
]

PROV_CENTERS = {
    "ALL": [-1.5, 114.0, 5],
    "JAMBI": [-1.6, 103.0, 8],
    "SUMSEL": [-3.1, 104.5, 8],
    "JATIM": [-7.7, 112.8, 8],
    "KALTENG": [-2.2, 113.5, 7],
    "KALTIM": [0.3, 116.8, 7],
}


# ==========================================
# 1. DATABASE LOKAL
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS hotspot_clusters (
            cluster_id TEXT PRIMARY KEY,
            provinsi TEXT,
            latitude REAL,
            longitude REAL,
            kabupaten TEXT,
            fungsi_kawasan TEXT,
            nama_kawasan TEXT,
            is_gambut INTEGER,
            first_detected DATE,
            last_detected DATE,
            consecutive_padam_days INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS hotspot_daily_log (
            cluster_id TEXT,
            log_date DATE,
            satellite_source TEXT,
            status TEXT,
            confidence REAL,
            PRIMARY KEY (cluster_id, log_date, satellite_source),
            FOREIGN KEY (cluster_id) REFERENCES hotspot_clusters (cluster_id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fire_risk_forecast (
            forecast_id TEXT PRIMARY KEY,
            provinsi TEXT,
            nama_kawasan TEXT,
            latitude REAL,
            longitude REAL,
            forecast_date DATE,
            temp_max REAL,
            rh_min REAL,
            precip_sum REAL,
            wind_max REAL,
            is_gambut INTEGER,
            risk_score REAL,
            risk_category TEXT,
            confidence_pct REAL,
            confidence_label TEXT,
            dry_days_projected INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    c.execute(
        "INSERT OR REPLACE INTO app_settings (key, value) VALUES ('nasa_key', ?)",
        (DEFAULT_NASA_KEY,),
    )
    conn.commit()
    conn.close()


def get_nasa_key():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT value FROM app_settings WHERE key = 'nasa_key'")
    row = c.fetchone()
    conn.close()
    return row[0] if row else DEFAULT_NASA_KEY


def set_nasa_key(new_key):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO app_settings (key, value) VALUES ('nasa_key', ?)",
        (new_key,),
    )
    conn.commit()
    conn.close()


# ==========================================
# 2. OVERLAY KAWASAN 5 PROVINSI
# ==========================================
def overlay_kawasan_multi(df_points):
    gdf = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points.longitude, df_points.latitude),
        crs="EPSG:4326",
    )

    def classify_row(row):
        prov = row["provinsi"]
        lat, lon = row["latitude"], row["longitude"]

        if prov == "JAMBI":
            if lon <= 101.90 and lat <= -1.40:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Kerinci Seblat (TNKS)",
                    0,
                    "Kerinci / Merangin",
                ])
            elif 102.30 <= lon <= 102.85 and -1.40 <= lat <= -0.90:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Bukit Tigapuluh (TNBT)",
                    0,
                    "Tebo / Tanjab Barat",
                ])
            elif 102.50 <= lon <= 102.85 and -2.05 <= lat <= -1.75:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Bukit Duabelas (TNBD)",
                    0,
                    "Sarolangun / Batanghari",
                ])
            elif lon >= 104.00 and -1.60 <= lat <= -1.15:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Berbak (TNB)",
                    1,
                    "Tanjung Jabung Timur",
                ])
            elif 103.00 <= lon <= 103.45 and -1.25 <= lat <= -0.90:
                return pd.Series([
                    "Hutan Lindung (HL)",
                    "HL Gambut Bram Itam",
                    1,
                    "Tanjung Jabung Barat",
                ])
            elif 103.15 <= lon <= 103.55 and -2.40 <= lat <= -2.00:
                return pd.Series([
                    "PBPH (Restorasi Ekosistem)",
                    "PBPH PT REKI (Hutan Harapan Jambi)",
                    0,
                    "Batanghari / Sarolangun",
                ])
            elif 103.35 <= lon <= 103.85 and -1.65 <= lat <= -1.15:
                return pd.Series([
                    "PBPH (Hutan Tanaman)",
                    "PBPH PT Wirakarya Sakti (WKS)",
                    1,
                    "Muaro Jambi",
                ])
            else:
                is_g = 1 if (lon >= 103.50 and lat >= -1.70) else 0
                return pd.Series([
                    f"APL (Lahan {'Gambut' if is_g else 'Mineral'})",
                    "Areal Perkebunan / Lahan Budidaya",
                    is_g,
                    "Muaro Jambi / Tanjabbar",
                ])

        elif prov == "SUMSEL":
            if lon <= 103.15 and -3.40 <= lat <= -2.30:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Kerinci Seblat (TNKS Sumsel)",
                    0,
                    "Musi Rawas / Muratara",
                ])
            elif 104.35 <= lon <= 105.10 and -2.45 <= lat <= -1.70:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Sembilang",
                    1,
                    "Banyuasin",
                ])
            elif 104.80 <= lon <= 105.35 and -3.20 <= lat <= -2.60:
                return pd.Series([
                    "KSA (Suaka Margasatwa)",
                    "SM Padang Sugihan",
                    1,
                    "Ogan Komering Ilir",
                ])
            elif 103.65 <= lon <= 104.15 and -2.65 <= lat <= -2.25:
                return pd.Series([
                    "KSA (Suaka Margasatwa)",
                    "SM Dangku",
                    0,
                    "Musi Banyuasin",
                ])
            elif lon >= 105.15 and -3.75 <= lat <= -2.75:
                return pd.Series([
                    "PBPH (Hutan Tanaman)",
                    "PBPH HTI Gambut OKI",
                    1,
                    "Ogan Komering Ilir",
                ])
            elif 103.20 <= lon <= 103.60 and -2.60 <= lat <= -2.35:
                return pd.Series([
                    "PBPH (Restorasi Ekosistem)",
                    "PBPH PT REKI (Hutan Harapan Sumsel)",
                    0,
                    "Musi Banyuasin",
                ])
            else:
                is_g = 1 if lon >= 104.70 else 0
                return pd.Series([
                    f"APL (Lahan {'Gambut' if is_g else 'Mineral'})",
                    "Perkebunan Sawit / Karet Rakyat",
                    is_g,
                    "OKI / Banyuasin",
                ])

        elif prov == "JATIM":
            if 112.80 <= lon <= 113.15 and -8.20 <= lat <= -7.85:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Bromo Tengger Semeru",
                    0,
                    "Malang / Probolinggo",
                ])
            elif 114.25 <= lon <= 114.55 and -7.95 <= lat <= -7.70:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Baluran",
                    0,
                    "Situbondo",
                ])
            elif 113.60 <= lon <= 114.05 and -8.65 <= lat <= -8.30:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Meru Betiri",
                    0,
                    "Jember / Banyuwangi",
                ])
            elif 112.45 <= lon <= 112.75 and -7.85 <= lat <= -7.60:
                return pd.Series([
                    "KPA (Tahura)",
                    "Tahura Raden Soerjo",
                    0,
                    "Mojokerto / Batu",
                ])
            elif (111.35 <= lon <= 112.25 and -7.65 <= lat <= -7.05) or (
                111.70 <= lon <= 112.50 and -8.25 <= lat <= -7.75
            ):
                return pd.Series([
                    "KPH Perhutani (HP/HL)",
                    "Hutan Produksi Perum Perhutani",
                    0,
                    "Bojonegoro / Madiun",
                ])
            else:
                return pd.Series([
                    "APL (Lahan Mineral)",
                    "Tegalan / Sawah / Pemukiman",
                    0,
                    "Kediri / Malang / Blitar",
                ])

        elif prov == "KALTENG":
            if 113.25 <= lon <= 114.15 and -3.25 <= lat <= -2.00:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Sebangau (Kubah Gambut)",
                    1,
                    "Palangka Raya / Pulang Pisau",
                ])
            elif 111.70 <= lon <= 112.35 and -3.35 <= lat <= -2.55:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Tanjung Puting",
                    1,
                    "Kotawaringin Barat",
                ])
            elif 114.00 <= lon <= 114.85 and -3.10 <= lat <= -2.30:
                return pd.Series([
                    "Kawasan Hidrologis Gambut (KHG)",
                    "Kawasan Eks-PLG Sejuta Hektar",
                    1,
                    "Pulang Pisau / Kapuas",
                ])
            elif 112.00 <= lon <= 113.80 and -2.50 <= lat <= -1.50:
                return pd.Series([
                    "PBPH / Perkebunan",
                    "Areal Konsesi Hutan / Sawit Kalteng",
                    1,
                    "Kotawaringin Timur",
                ])
            else:
                is_g = 1 if lat <= -1.80 else 0
                return pd.Series([
                    f"APL (Lahan {'Gambut' if is_g else 'Mineral'})",
                    "Lahan Budidaya / Masyarakat Kalteng",
                    is_g,
                    "Katingan / Seruyan / Kapuas",
                ])

        elif prov == "KALTIM":
            if 116.70 <= lon <= 117.20 and -1.25 <= lat <= -0.75:
                return pd.Series([
                    "KPA (Tahura) / Kawasan IKN",
                    "Tahura Bukit Soeharto & Lanskap IKN",
                    0,
                    "Kutai Kartanegara / PPU",
                ])
            elif 117.10 <= lon <= 117.60 and 0.10 <= lat <= 0.65:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Kutai",
                    0,
                    "Kutai Timur / Bontang",
                ])
            elif 116.80 <= lon <= 117.00 and -1.18 <= lat <= -1.05:
                return pd.Series([
                    "Hutan Lindung (HL)",
                    "HL Sungai Wain (HLSW)",
                    0,
                    "Balikpapan",
                ])
            elif 115.50 <= lon <= 117.00 and -0.80 <= lat <= 0.50:
                return pd.Series([
                    "PBPH / Konsesi",
                    "Konsesi HTI & Tambang Mahakam",
                    0,
                    "Kutai Kartanegara / Kutai Barat",
                ])
            else:
                return pd.Series([
                    "APL (Lahan Mineral)",
                    "Perkebunan / Pemukiman Kaltim",
                    0,
                    "Berau / Paser / Kutai Timur",
                ])

        return pd.Series(["Lainnya", "Area Luar Wilayah", 0, "Kabupaten"])

    gdf[["fungsi_kawasan", "nama_kawasan", "is_gambut", "kabupaten"]] = (
        gdf.apply(classify_row, axis=1)
    )
    return gdf


# ==========================================
# 3. SPATIAL CLUSTERING (~500m)
# ==========================================
def match_or_create_clusters_multi(conn, new_gdf, threshold_km=0.5):
    c = conn.cursor()
    matched_ids = []

    for prov in ["JAMBI", "SUMSEL", "JATIM", "KALTENG", "KALTIM"]:
        prov_gdf = new_gdf[new_gdf["provinsi"] == prov]
        if prov_gdf.empty:
            continue

        c.execute(
            "SELECT cluster_id, latitude, longitude FROM hotspot_clusters"
            " WHERE is_active = 1 AND provinsi = ?",
            (prov,),
        )
        existing = c.fetchall()

        if not existing:
            for idx, row in prov_gdf.iterrows():
                cid = f"HS-{prov}-{int(datetime.now().timestamp())}-{idx}"
                c.execute(
                    """
                    INSERT INTO hotspot_clusters (cluster_id, provinsi, latitude, longitude, kabupaten, fungsi_kawasan, nama_kawasan, is_gambut, first_detected, last_detected, consecutive_padam_days, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
                """,
                    (
                        cid,
                        prov,
                        row.latitude,
                        row.longitude,
                        row.kabupaten,
                        row.fungsi_kawasan,
                        row.nama_kawasan,
                        row.is_gambut,
                        row.date,
                        row.date,
                    ),
                )
                matched_ids.append((cid, row.name))
            conn.commit()
            continue

        ex_ids = [e[0] for e in existing]
        ex_coords = [[e[1], e[2]] for e in existing]
        tree = cKDTree(np.array(ex_coords) * 111.0)

        for idx, row in prov_gdf.iterrows():
            pt = np.array([row.latitude, row.longitude]) * 111.0
            dist, index = tree.query(pt, distance_upper_bound=threshold_km)

            if dist < threshold_km:
                cid = ex_ids[index]
                c.execute(
                    "UPDATE hotspot_clusters SET last_detected = ?,"
                    " consecutive_padam_days = 0 WHERE cluster_id = ?",
                    (row.date, cid),
                )
            else:
                cid = f"HS-{prov}-{int(datetime.now().timestamp())}-{idx}"
                c.execute(
                    """
                    INSERT INTO hotspot_clusters (cluster_id, provinsi, latitude, longitude, kabupaten, fungsi_kawasan, nama_kawasan, is_gambut, first_detected, last_detected, consecutive_padam_days, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
                """,
                    (
                        cid,
                        prov,
                        row.latitude,
                        row.longitude,
                        row.kabupaten,
                        row.fungsi_kawasan,
                        row.nama_kawasan,
                        row.is_gambut,
                        row.date,
                        row.date,
                    ),
                )
            matched_ids.append((cid, row.name))

    conn.commit()
    id_dict = {orig_idx: cid for cid, orig_idx in matched_ids}
    return [id_dict.get(idx, f"HS-GEN-{idx}") for idx in new_gdf.index]


# ==========================================
# 4. PENARIKAN NASA & OPEN-METEO
# ==========================================
def fetch_satellite_data_multi(target_date_str, target_prov="ALL"):
    map_key = get_nasa_key()
    prov_keys = (
        ["JAMBI", "SUMSEL", "JATIM", "KALTENG", "KALTIM"]
        if target_prov == "ALL"
        else [target_prov]
    )
    combined = []

    for p in prov_keys:
        bbox = PROVINSI_BBOX[p]
        for api_code, sat_lbl in [
            ("VIIRS_NOAA20_NRT", "NOAA-20"),
            ("VIIRS_SNPP_NRT", "SNPP"),
            ("MODIS_NRT", "Terra/Aqua"),
        ]:
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{api_code}/{bbox}/1/{target_date_str}"
            try:
                resp = requests.get(url, timeout=9)
                if resp.status_code == 200 and "latitude" in resp.text:
                    df = pd.read_csv(io.StringIO(resp.text))
                    for _, r in df.iterrows():
                        c_raw = str(r.get("confidence", "n")).lower()
                        c_val = (
                            90.0
                            if c_raw == "h"
                            else (
                                55.0
                                if c_raw == "n"
                                else (20.0 if c_raw == "l" else float(c_raw))
                            )
                        )
                        combined.append({
                            "provinsi": p,
                            "latitude": float(r["latitude"]),
                            "longitude": float(r["longitude"]),
                            "confidence": c_val,
                            "satellite_source": sat_lbl,
                        })
            except Exception:
                pass
    return combined


def update_fire_risk_forecast_multi():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    for item in SENTINEL_POINTS:
        lat, lon = item["lat"], item["lon"]
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,relative_humidity_2m_min,precipitation_sum,wind_speed_10m_max"
            "&timezone=Asia%2FJakarta&forecast_days=5"
        )
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json().get("daily", {})
                times = data.get("time", [])
                t_max = data.get("temperature_2m_max", [])
                rh_min = data.get("relative_humidity_2m_min", [])
                precip = data.get("precipitation_sum", [])
                wind = data.get("wind_speed_10m_max", [])

                dry_count = 0
                for i in range(len(times)):
                    fc_date = times[i]
                    tm = t_max[i] if i < len(t_max) else 33.0
                    rm = rh_min[i] if i < len(rh_min) else 55.0
                    pr = precip[i] if i < len(precip) else 0.0
                    wn = wind[i] if i < len(wind) else 12.0

                    if pr < 1.0:
                        dry_count += 1
                    else:
                        dry_count = 0

                    temp_score = min(max((tm - 28.0) * 8.0, 0), 40)
                    rh_score = min(max((75.0 - rm) * 1.2, 0), 30)
                    dry_score = 25 if pr < 1.0 else (10 if pr < 5.0 else 0)
                    peat_multiplier = 1.25 if item["gambut"] == 1 else 1.05

                    raw_score = (
                        (temp_score + rh_score + dry_score)
                        * peat_multiplier
                        * (1.0 + (wn / 60.0))
                    )
                    score = min(round(raw_score, 1), 100.0)

                    conf_raw = 100.0 / (1.0 + math.exp(-0.11 * (score - 60.0)))
                    conf_pct = min(max(round(conf_raw, 1), 5.0), 98.0)

                    if conf_pct >= 85:
                        conf_label = "Sangat Tinggi (High Confidence)"
                    elif conf_pct >= 65:
                        conf_label = "Tinggi (Medium-High)"
                    elif conf_pct >= 40:
                        conf_label = "Sedang (Moderate)"
                    else:
                        conf_label = "Rendah (Low)"

                    if score >= 80:
                        cat = "Ekstrim"
                    elif score >= 60:
                        cat
