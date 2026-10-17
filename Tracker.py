import io
import json
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
DEFAULT_NASA_KEY = "DEMO_KEY"

PROVINSI_BBOX = {
    "JAMBI": "101.15,-2.50,104.55,-0.75",
    "SUMSEL": "102.05,-4.95,106.15,-1.65",
    "JATIM": "110.85,-8.80,114.65,-6.75",
    "KALTENG": "110.75,-3.55,115.85,0.75",
    "KALTIM": "113.80,-2.45,119.05,2.85",
}

# 15 Titik Sentinel Acuan Peringatan Dini H-5 di Kawasan Kritis
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
        "kawasan": "PBPH PT REKI (Hutan Harapan Jambi)",
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
        "kawasan": "Tahura Bukit Soeharto / Delineasi IKN",
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
        "kawasan": "Konsesi HTI / Tambang Kutai Kartanegara",
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
            risk_category TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    c.execute(
        "INSERT OR IGNORE INTO app_settings (key, value) VALUES ('nasa_key', ?)",
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
# 2. OVERLAY KAWASAN SPESIFIK 5 PROVINSI
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

        # Jambi
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

        # Sumatera Selatan
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

        # Jawa Timur
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

        # Kalimantan Tengah
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

        # Kalimantan Timur
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
# 4. KONEKTOR PENARIKAN DATA NASA & CUACA H-5
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
    """Mengambil data 5 hari ke depan dan menganalisis 3 elemen pembakar."""
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

                for i in range(len(times)):
                    fc_date = times[i]
                    tm = t_max[i] if i < len(t_max) else 33.0
                    rm = rh_min[i] if i < len(rh_min) else 55.0
                    pr = precip[i] if i < len(precip) else 0.0
                    wn = wind[i] if i < len(wind) else 12.0

                    # 3 ELEMEN PEMBAKAR:
                    # 1. Panas (Suhu Udara Puncak Siang Hari)
                    temp_score = min(max((tm - 28.0) * 8.0, 0), 40)
                    # 2. Desikasi / Oksigen (Kelembapan Rendah + Angin Kencang)
                    rh_score = min(max((75.0 - rm) * 1.2, 0), 30)
                    # 3. Ketersediaan Bahan Bakar Kering (Defisit Curah Hujan)
                    dry_score = 25 if pr < 1.0 else (10 if pr < 5.0 else 0)

                    # Pengali tipe bahan bakar (Gambut vs Mineral)
                    peat_multiplier = 1.25 if item["gambut"] == 1 else 1.05

                    raw_score = (
                        (temp_score + rh_score + dry_score)
                        * peat_multiplier
                        * (1.0 + (wn / 60.0))
                    )
                    score = min(round(raw_score, 1), 100.0)

                    if score >= 80:
                        cat = "Ekstrim"
                    elif score >= 60:
                        cat = "Tinggi"
                    elif score >= 35:
                        cat = "Sedang"
                    else:
                        cat = "Rendah"

                    fid = f"FC-{item['prov']}-{fc_date}-{int(lat*100)}-{int(lon*100)}"
                    c.execute(
                        """
                        INSERT OR REPLACE INTO fire_risk_forecast
                        (forecast_id, provinsi, nama_kawasan, latitude, longitude, forecast_date, temp_max, rh_min, precip_sum, wind_max, is_gambut, risk_score, risk_category)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            fid,
                            item["prov"],
                            item["kawasan"],
                            lat,
                            lon,
                            fc_date,
                            tm,
                            rm,
                            pr,
                            wn,
                            item["gambut"],
                            score,
                            cat,
                        ),
                    )
        except Exception:
            pass

    conn.commit()
    conn.close()


def sync_hotspots_multi(target_date_str):
    init_db()
    raw_points = fetch_satellite_data_multi(target_date_str, target_prov="ALL")
    if not raw_points:
        return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    df_raw = pd.DataFrame(raw_points)
    df_raw["date"] = target_date_str

    def set_status(conf):
        return (
            "Tinggi" if conf >= 80 else ("Sedang" if conf >= 30 else "Rendah")
        )

    df_raw["status"] = df_raw["confidence"].apply(set_status)
    gdf = overlay_kawasan_multi(df_raw)
    gdf["cluster_id"] = match_or_create_clusters_multi(conn, gdf)

    for _, row in gdf.iterrows():
        c.execute(
            """
            INSERT OR REPLACE INTO hotspot_daily_log (cluster_id, log_date, satellite_source, status, confidence)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                row.cluster_id,
                target_date_str,
                row.satellite_source,
                row.status,
                row.confidence,
            ),
        )

    c.execute(
        "SELECT cluster_id, consecutive_padam_days FROM hotspot_clusters WHERE"
        " is_active = 1"
    )
    all_active = c.fetchall()
    active_today = set(gdf["cluster_id"])

    for cid, padam_days in all_active:
        if cid not in active_today:
            new_p = padam_days + 1
            for sat in ["NOAA-20", "SNPP", "Terra/Aqua", "HK-PJKarhutla"]:
                c.execute(
                    """
                    INSERT OR REPLACE INTO hotspot_daily_log (cluster_id, log_date, satellite_source, status, confidence)
                    VALUES (?, ?, ?, 'Padam', 0.0)
                """,
                    (cid, target_date_str, sat),
                )

            if new_p >= 10:
                c.execute(
                    "UPDATE hotspot_clusters SET consecutive_padam_days = ?,"
                    " is_active = 0 WHERE cluster_id = ?",
                    (new_p, cid),
                )
            else:
                c.execute(
                    "UPDATE hotspot_clusters SET consecutive_padam_days = ?"
                    " WHERE cluster_id = ?",
                    (new_p, cid),
                )

    conn.commit()
    conn.close()


# ==========================================
# 5. SEED DATA DENGAN SISTEM AUTO-MIGRATION
# ==========================================
def seed_historical_multi():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Periksa apakah Kalimantan Tengah sudah ada
    c.execute(
        "SELECT COUNT(*) FROM hotspot_clusters WHERE provinsi = 'KALTENG'"
    )
    has_kalteng = c.fetchone()[0]

    # Jika Kalteng belum ada, bersihkan database lama agar migrasi ke 5 provinsi bersih sempurna
    if has_kalteng == 0:
        c.execute("DELETE FROM hotspot_clusters")
        c.execute("DELETE FROM hotspot_daily_log")
        c.execute("DELETE FROM fire_risk_forecast")
        conn.commit()

        clusters = [
            # Jambi
            {
                "id": "HS-JAMBI-01",
                "prov": "JAMBI",
                "lat": -1.3520,
                "lon": 104.1020,
                "kab": "Tanjung Jabung Timur",
                "f_kws": "KPA (Taman Nasional)",
                "n_kws": "TN Berbak (TNB)",
                "gambut": 1,
                "first": "2026-10-10",
                "last": "2026-10-17",
                "padam_cnt": 0,
                "logs": {
                    "2026-10-15": ("Tinggi", 92, "HK-PJKarhutla"),
                    "2026-10-16": ("Tinggi", 94, "HK-PJKarhutla"),
                    "2026-10-17": ("Tinggi", 93, "HK-PJKarhutla"),
                },
            },
            {
                "id": "HS-JAMBI-02",
                "prov": "JAMBI",
                "lat": -1.4200,
                "lon": 103.5500,
                "kab": "Muaro Jambi",
                "f_kws": "PBPH (Hutan Tanaman)",
                "n_kws": "PBPH PT WKS",
                "gambut": 1,
                "first": "2026-10-08",
                "last": "2026-10-11",
                "padam_cnt": 6,
                "logs": {
                    "2026-10-09": ("Sedang", 60, "Terra/Aqua"),
                    "2026-10-10": ("Tinggi", 82, "NOAA-20"),
                    "2026-10-11": ("Sedang", 55, "Terra/Aqua"),
                    "2026-10-12": ("Padam", 0, "Terra/Aqua"),
                    "2026-10-13": ("Padam", 0, "Terra/Aqua"),
                    "2026-10-14": ("Padam", 0, "Terra/Aqua"),
                    "2026-10-15": ("Padam", 0, "Terra/Aqua"),
                    "2026-10-16": ("Padam", 0, "Terra/Aqua"),
                    "2026-10-17": ("Padam", 0, "Terra/Aqua"),
                },
            },
            # Sumsel
            {
                "id": "HS-SUMSEL-01",
                "prov": "SUMSEL",
                "lat": -3.3500,
                "lon": 105.4200,
                "kab": "Ogan Komering Ilir",
                "f_kws": "PBPH (Hutan Tanaman)",
                "n_kws": "PBPH HTI Gambut OKI",
                "gambut": 1,
                "first": "2026-10-11",
                "last": "2026-10-17",
                "padam_cnt": 0,
                "logs": {
                    "2026-10-15": ("Tinggi", 96, "NOAA-20"),
                    "2026-10-16": ("Tinggi", 91, "NOAA-20"),
                    "2026-10-17": ("Tinggi", 93, "NOAA-20"),
                },
            },
            {
                "id": "HS-SUMSEL-02",
                "prov": "SUMSEL",
                "lat": -2.9500,
                "lon": 105.0500,
                "kab": "Ogan Komering Ilir",
                "f_kws": "KSA (Suaka Margasatwa)",
                "n_kws": "SM Padang Sugihan",
                "gambut": 1,
                "first": "2026-10-12",
                "last": "2026-10-17",
                "padam_cnt": 0,
                "logs": {
                    "2026-10-15": ("Sedang", 72, "Terra/Aqua"),
                    "2026-10-16": ("Tinggi", 84, "HK-PJKarhutla"),
                    "2026-10-17": ("Tinggi", 88, "HK-PJKarhutla"),
                },
            },
            # Jatim
            {
                "id": "HS-JATIM-01",
                "prov": "JATIM",
                "lat": -7.9400,
                "lon": 112.9500,
                "kab": "Malang / Probolinggo",
                "f_kws": "KPA (Taman Nasional)",
                "n_kws": "TN Bromo Tengger Semeru",
                "gambut": 0,
                "first": "2026-10-13",
                "last": "2026-10-17",
                "padam_cnt": 0,
                "logs": {
                    "2026-10-15": ("Tinggi", 86, "HK-PJKarhutla"),
                    "2026-10-16": ("Tinggi", 88, "HK-PJKarhutla"),
                    "2026-10-17": ("Tinggi", 91, "HK-PJKarhutla"),
                },
            },
            # Kalteng
            {
                "id": "HS-KALTENG-01",
                "prov": "KALTENG",
                "lat": -2.3500,
                "lon": 113.8500,
                "kab": "Palangka Raya / Pulang Pisau",
                "f_kws": "KPA (Taman Nasional)",
                "n_kws": "TN Sebangau (Kubah Gambut)",
                "gambut": 1,
                "first": "2026-10-12",
                "last": "2026-10-17",
                "padam_cnt": 0,
                "logs": {
                    "2026-10-15": ("Tinggi", 89, "NOAA-20"),
                    "2026-10-16": ("Tinggi", 95, "NOAA-20"),
                    "2026-10-17": ("Tinggi", 94, "NOAA-20"),
                },
            },
            {
                "id": "HS-KALTENG-02",
                "prov": "KALTENG",
                "lat": -2.7500,
                "lon": 114.2500,
                "kab": "Pulang Pisau",
                "f_kws": "Kawasan Hidrologis Gambut (KHG)",
                "n_kws": "Kawasan Eks-PLG Sejuta Hektar",
                "gambut": 1,
                "first": "2026-10-10",
                "last": "2026-10-17",
                "padam_cnt": 0,
                "logs": {
                    "2026-10-15": ("Tinggi", 87, "Terra/Aqua"),
                    "2026-10-16": ("Tinggi", 92, "Terra/Aqua"),
                    "2026-10-17": ("Tinggi", 96, "Terra/Aqua"),
                },
            },
            # Kaltim
            {
                "id": "HS-KALTIM-01",
                "prov": "KALTIM",
                "lat": -0.9800,
                "lon": 117.0200,
                "kab": "Kutai Kartanegara / PPU",
                "f_kws": "KPA (Tahura) / Kawasan IKN",
                "n_kws": "Tahura Bukit Soeharto & Lanskap IKN",
                "gambut": 0,
                "first": "2026-10-14",
                "last": "2026-10-17",
                "padam_cnt": 0,
                "logs": {
                    "2026-10-15": ("Rendah", 28, "Terra/Aqua"),
                    "2026-10-16": ("Sedang", 65, "Terra/Aqua"),
                    "2026-10-17": ("Tinggi", 84, "Terra/Aqua"),
                },
            },
        ]

        for item in clusters:
            c.execute(
                """
                INSERT OR REPLACE INTO hotspot_clusters 
                (cluster_id, provinsi, latitude, longitude, kabupaten, fungsi_kawasan, nama_kawasan, is_gambut, first_detected, last_detected, consecutive_padam_days, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
                (
                    item["id"],
                    item["prov"],
                    item["lat"],
                    item["lon"],
                    item["kab"],
                    item["f_kws"],
                    item["n_kws"],
                    item["gambut"],
                    item["first"],
                    item["last"],
                    item["padam_cnt"],
                ),
            )

            base_date = date(2026, 10, 8)
            for day_offset in range(10):
                d_str = (base_date + timedelta(days=day_offset)).strftime(
                    "%Y-%m-%d"
                )
                if d_str in item["logs"]:
                    status, conf, sat = item["logs"][d_str]
                else:
                    status, conf, sat = "Padam", 0.0, "Terra/Aqua"
                c.execute(
                    """
                    INSERT OR REPLACE INTO hotspot_daily_log (cluster_id, log_date, satellite_source, status, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (item["id"], d_str, sat, status, conf),
                )

        conn.commit()
    conn.close()


# ==========================================
# 6. LOGIKA PIVOT, REKAP & MAP DATA
# ==========================================
def get_effective_date_and_session():
    now = datetime.now()
    hour = now.hour
    if hour < 7:
        effective_date = now.date() - timedelta(days=1)
        session_text = "Sesi Sore 18.00 WIB (Kemarin)"
    elif 7 <= hour < 18:
        effective_date = now.date()
        session_text = "Sesi Pagi 07.00 WIB (Hari Ini)"
    else:
        effective_date = now.date()
        session_text = "Sesi Sore 18.00 WIB (Hari Ini)"
    return effective_date, session_text


def get_provincial_potential_summary():
    conn = sqlite3.connect(DB_NAME)
    summary = []
    prov_list = ["JAMBI", "SUMSEL", "JATIM", "KALTENG", "KALTIM"]

    for p in prov_list:
        df_p = pd.read_sql_query(
            f"SELECT risk_score, risk_category, is_gambut FROM fire_risk_forecast WHERE provinsi = '{p}'",
            conn,
        )
        df_hs = pd.read_sql_query(
            f"SELECT COUNT(*) as hs_cnt FROM hotspot_clusters WHERE provinsi = '{p}' AND is_active = 1",
            conn,
        )

        ekstrim = (
            int((df_p["risk_category"] == "Ekstrim").sum())
            if not df_p.empty
            else 0
        )
        tinggi = (
            int((df_p["risk_category"] == "Tinggi").sum())
            if not df_p.empty
            else 0
        )
        gambut = (
            int(
                (
                    (df_p["risk_score"] >= 60) & (df_p["is_gambut"] == 1)
                ).sum()
            )
            if not df_p.empty
            else 0
        )
        hs_count = int(df_hs.iloc[0]["hs_cnt"]) if not df_hs.empty else 0

        if ekstrim >= 2:
            status_label = "Siaga 1 (Kritis Ekstrim)"
            badge_class = "badge-ekstrim"
        elif ekstrim >= 1 or tinggi >= 2:
            status_label = "Siaga 2 (Tinggi)"
            badge_class = "badge-tinggi"
        else:
            status_label = "Waspada"
            badge_class = "badge-sedang"

        summary.append({
            "provinsi": p,
            "status_label": status_label,
            "badge_class": badge_class,
            "ekstrim_cnt": ekstrim,
            "tinggi_cnt": tinggi,
            "gambut_cnt": gambut,
            "hs_count": hs_count,
        })

    conn.close()
    return summary


def get_forecast_summary_multi(filter_prov="ALL"):
    conn = sqlite3.connect(DB_NAME)
    prov_where = (
        "" if filter_prov == "ALL" else f"WHERE provinsi = '{filter_prov}'"
    )
    query = f"""
        SELECT provinsi, nama_kawasan, latitude, longitude, forecast_date, temp_max, rh_min, precip_sum, wind_max, is_gambut, risk_score, risk_category
        FROM fire_risk_forecast
        {prov_where}
        ORDER BY forecast_date ASC, risk_score DESC
    """
    df_fc = pd.read_sql_query(query, conn)
    conn.close()
    return df_fc.to_dict(orient="records") if not df_fc.empty else []


def generate_table_multi(
    ref_date,
    filter_prov="ALL",
    filter_sat="ALL",
    filter_status="ALL",
    filter_source="ALL_DATA",
):
    date_list = [
        (ref_date - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(9, -1, -1)
    ]
    min_date = date_list[0]

    conn = sqlite3.connect(DB_NAME)
    prov_where = (
        "" if filter_prov == "ALL" else f"AND provinsi = '{filter_prov}'"
    )
    df_clusters = pd.read_sql_query(
        f"SELECT cluster_id, provinsi, latitude, longitude, kabupaten, fungsi_kawasan, nama_kawasan, is_gambut FROM hotspot_clusters WHERE is_active = 1 {prov_where}",
        conn,
    )

    summary_stats = {
        "total": 0,
        "tinggi": 0,
        "sedang": 0,
        "rendah": 0,
        "padam": 0,
        "aktif": 0,
        "konservasi": 0,
        "gambut": 0,
        "ekstrim_fc": 0,
    }

    if df_clusters.empty:
        conn.close()
        return pd.DataFrame(), date_list, summary_stats

    sat_cond = ""
    if filter_sat == "NOAA":
        sat_cond = "AND satellite_source LIKE '%NOAA%'"
    elif filter_sat == "TERRA":
        sat_cond = "AND satellite_source LIKE '%Terra%'"

    if filter_source == "GITHUB_HK":
        df_clusters = df_clusters[
            df_clusters["fungsi_kawasan"].str.contains(
                "KPA|KSA|Tahura", na=False
            )
        ]

    query_logs = f"""
        SELECT cluster_id, log_date, status, confidence, satellite_source
        FROM hotspot_daily_log 
        WHERE log_date >= '{min_date}' AND log_date <= '{date_list[-1]}' {sat_cond}
    """
    df_logs = pd.read_sql_query(query_logs, conn)

    fc_prov_where = (
        "" if filter_prov == "ALL" else f"AND provinsi = '{filter_prov}'"
    )
    df_fc_count = pd.read_sql_query(
        f"SELECT COUNT(*) as cnt FROM fire_risk_forecast WHERE risk_score >= 80 {fc_prov_where}",
        conn,
    )
    summary_stats["ekstrim_fc"] = (
        int(df_fc_count.iloc[0]["cnt"]) if not df_fc_count.empty else 0
    )

    conn.close()

    if not df_logs.empty:
        rank_map = {"Tinggi": 4, "Sedang": 3, "Rendah": 2, "Padam": 1}
        df_logs["rank"] = df_logs["status"].map(rank_map)
        agg_logs = (
            df_logs.sort_values("rank", ascending=False)
            .groupby(["cluster_id", "log_date"])
            .first()
            .reset_index()
        )
        pivot = agg_logs.pivot(
            index="cluster_id", columns="log_date", values="status"
        ).fillna("Padam")

        sat_summary = (
            df_logs[df_logs["status"] != "Padam"]
            .groupby("cluster_id")["satellite_source"]
            .unique()
            .apply(lambda x: ", ".join(x))
            .to_dict()
        )
    else:
        pivot = pd.DataFrame(index=df_clusters["cluster_id"])
        sat_summary = {}

    for d in date_list:
        if d not in pivot.columns:
            pivot[d] = "Padam"
    pivot = pivot[date_list]

    merged = pd.merge(df_clusters, pivot, on="cluster_id", how="inner")
    merged["is_gambut_label"] = merged["is_gambut"].map({1: "Ya", 0: "Tidak"})
    merged["sensor_terdeteksi"] = merged["cluster_id"].map(
        lambda cid: sat_summary.get(cid, "-")
    )

    if filter_sat != "ALL":
        merged = merged[merged["sensor_terdeteksi"] != "-"]

    latest_col = date_list[-1]
    merged["status_terakhir"] = merged[latest_col]

    if filter_status == "TINGGI":
        merged = merged[merged["status_terakhir"] == "Tinggi"]
    elif filter_status == "SEDANG_RENDAH":
        merged = merged[merged["status_terakhir"].isin(["Sedang", "Rendah"])]
    elif filter_status == "AKTIF":
        merged = merged[merged["status_terakhir"] != "Padam"]

    if not merged.empty:
        st_counts = merged["status_terakhir"].value_counts()
        summary_stats["tinggi"] = int(st_counts.get("Tinggi", 0))
        summary_stats["sedang"] = int(st_counts.get("Sedang", 0))
        summary_stats["rendah"] = int(st_counts.get("Rendah", 0))
        summary_stats["padam"] = int(st_counts.get("Padam", 0))
        summary_stats["aktif"] = (
            summary_stats["tinggi"]
            + summary_stats["sedang"]
            + summary_stats["rendah"]
        )
        summary_stats["total"] = len(merged)

        aktif_rows = merged[merged["status_terakhir"] != "Padam"]
        summary_stats["konservasi"] = int(
            aktif_rows["fungsi_kawasan"]
            .str.contains("KPA|KSA|Tahura", na=False)
            .sum()
        )
        summary_stats["gambut"] = int((aktif_rows["is_gambut"] == 1).sum())

    merged.insert(0, "No", range(1, len(merged) + 1))
    return merged, date_list, summary_stats


def export_to_excel_multi(
    ref_date_str,
    filter_prov="ALL",
    filter_sat="ALL",
    filter_source="ALL_DATA",
):
    ref_d = datetime.strptime(ref_date_str, "%Y-%m-%d").date()
    df, _, _ = generate_table_multi(
        ref_d,
        filter_prov=filter_prov,
        filter_sat=filter_sat,
        filter_source=filter_source,
    )
    file_name = f"Laporan_Hotspot_{filter_prov}.xlsx"
    if not df.empty:
        export_df = df.drop(
            columns=["is_gambut", "status_terakhir"], errors="ignore"
        )
        export_df = export_df.rename(columns={"is_gambut_label": "is_gambut"})
        export_df.to_excel(file_name, index=False)
    return file_name


# ==========================================
# 7. PENJADWAL OTOMATIS (07:00 & 18:00)
# ==========================================
def scheduled_sync_multi():
    today_str = date.today().strftime("%Y-%m-%d")
    sync_hotspots_multi(today_str)
    update_fire_risk_forecast_multi()


def run_scheduler():
    schedule.every().day.at("07:00").do(scheduled_sync_multi)
    schedule.every().day.at("18:00").do(scheduled_sync_multi)
    while True:
        schedule.run_pending()
        time.sleep(30)


# ==========================================
# 8. DASBOR WEB FLASK + LEAFLET.JS LENGKAP
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Sistem Monitoring Hotspot & Peringatan Dini Karhutla</title>
    <!-- Leaflet.js CDN -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 24px; background: #f8fafc; color: #1e293b; }
        .header { display: flex; justify-content: space-between; align-items: center; background: #fff; padding: 20px 28px; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }
        h1 { margin: 0; font-size: 22px; color: #0f172a; }
        .subtitle { font-size: 13px; color: #64748b; margin-top: 4px; }
        .schedule-badge { background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 6px; font-weight: 700; }
        
        .controls { display: flex; gap: 12px; align-items: center; background: white; padding: 14px 24px; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; flex-wrap: wrap; }
        select { padding: 9px 12px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 13px; font-weight: 600; background: white; color: #0f172a; }
        .select-prov { background: #f0fdf4; border-color: #86efac; color: #166534; font-size: 14px; }
        
        button, a.btn { padding: 9px 16px; border-radius: 8px; font-weight: 600; text-decoration: none; cursor: pointer; border: none; font-size: 13px; display: inline-flex; align-items: center; gap: 6px; }
        .btn-refresh { background: #0284c7; color: white; }
        .btn-excel { background: #16a34a; color: white; }
        
        .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow-x: auto; margin-bottom: 24px; }
        .card-header-title { font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }
        
        #map { height: 490px; width: 100%; border-radius: 10px; border: 1px solid #e2e8f0; }
        
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #f1f5f9; white-space: nowrap; }
        th { background: #f8fafc; color: #475569; font-weight: 700; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
        
        .badge { padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; }
        .badge-tinggi, .badge-ekstrim { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
        .badge-sedang { background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }
        .badge-rendah { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
        .badge-padam { background: #f1f5f9; color: #94a3b8; border: 1px solid #e2e8f0; }
        
        .prov-tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
        .prov-jambi { background: #e0f2fe; color: #0284c7; }
        .prov-sumsel { background: #fef3c7; color: #d97706; }
        .prov-jatim { background: #dcfce7; color: #15803d; }
        .prov-kalteng { background: #ffedd5; color: #ea580c; }
        .prov-kaltim { background: #f3e8ff; color: #7e22ce; }
        
        .summary-container { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 24px; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; }
        .stat-card { padding: 16px; border-radius: 10px; border: 1px solid #e2e8f0; background: #fafafa; display: flex; flex-direction: column; justify-content: space-between; }
        .stat-val { font-size: 26px; font-weight: 800; margin-top: 4px; }
        .stat-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; color: #64748b; }
        .stat-ekstrim { background: #fff1f2; border-color: #fda4af; }
        .stat-ekstrim .stat-val { color: #e11d48; }

        .legend-box { background: white; padding: 10px 14px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.2); font-size: 12px; line-height: 20px; }
        .legend-dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>Sistem Monitoring Hotspot & Peringatan Dini Karhutla</h1>
            <div class="subtitle">
                Cakupan: <strong>Jambi, Sumatera Selatan, Jawa Timur, Kalimantan Tengah, Kalimantan Timur</strong> | 
                Sinkronisasi: <span class="schedule-badge">07:00 & 18:00 WIB</span> | 
                Sesi: <strong>{{ session_name }} ({{ ref_date }})</strong>
            </div>
        </div>
        <div>
            <a href="/download-excel?prov={{ current_prov }}&sat={{ current_sat }}&source={{ current_source }}" class="btn btn-excel">📥 Unduh Excel ({{ current_prov }})</a>
        </div>
    </div>

    <div class="controls">
        <label><strong>Pilih Wilayah:</strong></label>
        <select id="provSelect" class="select-prov" onchange="applyFilters()">
            <option value="ALL" {% if current_prov == 'ALL' %}selected{% endif %}>🇮🇩 Seluruh 5 Wilayah Provinsi</option>
            <option value="JAMBI" {% if current_prov == 'JAMBI' %}selected{% endif %}>📍 Provinsi Jambi</option>
            <option value="SUMSEL" {% if current_prov == 'SUMSEL' %}selected{% endif %}>📍 Provinsi Sumatera Selatan</option>
            <option value="JATIM" {% if current_prov == 'JATIM' %}selected{% endif %}>📍 Provinsi Jawa Timur</option>
            <option value="KALTENG" {% if current_prov == 'KALTENG' %}selected{% endif %}>📍 Provinsi Kalimantan Tengah</option>
            <option value="KALTIM" {% if current_prov == 'KALTIM' %}selected{% endif %}>📍 Provinsi Kalimantan Timur</option>
        </select>

        <label><strong>Sumber Feed:</strong></label>
        <select id="sourceSelect" onchange="applyFilters()">
            <option value="ALL_DATA" {% if current_source == 'ALL_DATA' %}selected{% endif %}>🌐 Seluruh Lanskap (HK, HL, PBPH, APL)</option>
            <option value="GITHUB_HK" {% if current_source == 'GITHUB_HK' %}selected{% endif %}>🌲 Khusus Kawasan Konservasi (TN/CA/SM/Tahura)</option>
        </select>

        <label><strong>Sensor:</strong></label>
        <select id="satSelect" onchange="applyFilters()">
            <option value="ALL" {% if current_sat == 'ALL' %}selected{% endif %}>Semua Sensor (NOAA/SNPP/Terra/Aqua)</option>
            <option value="NOAA" {% if current_sat == 'NOAA' %}selected{% endif %}>🛰️ NOAA-20 / SNPP (375m)</option>
            <option value="TERRA" {% if current_sat == 'TERRA' %}selected{% endif %}>🛰️ Terra / Aqua (1km)</option>
        </select>

        <button class="btn btn-refresh" onclick="refreshData()">🔄 Sinkronisasi Hotspot & Cuaca</button>
    </div>

    <!-- KARTU STATISTIK UTAMA -->
    <div class="summary-container">
        <div class="summary-grid">
            <div class="stat-card">
                <span class="stat-lbl">Total Titik Hotspot Aktif</span>
                <span class="stat-val" style="color: #0284c7;">{{ stats['aktif'] }}</span>
            </div>
            <div class="stat-card" style="background:#fef2f2; border-color:#fecaca;">
                <span class="stat-lbl">Hotspot Risiko Tinggi (≥80%)</span>
                <span class="stat-val" style="color:#dc2626;">{{ stats['tinggi'] }}</span>
            </div>
            <div class="stat-card" style="background:#fffbeb; border-color:#fde68a;">
                <span class="stat-lbl">Hotspot Sedang (30-79%)</span>
                <span class="stat-val" style="color:#d97706;">{{ stats['sedang'] }}</span>
            </div>
            <div class="stat-card">
                <span class="stat-lbl">Di Hutan Konservasi (TN/SM)</span>
                <span class="stat-val" style="color:#e11d48;">{{ stats['konservasi'] }}</span>
            </div>
            <div class="stat-card">
                <span class="stat-lbl">Di Lahan Rawa Gambut</span>
                <span class="stat-val" style="color:#ea580c;">{{ stats['gambut'] }}</span>
            </div>
            <div class="stat-card stat-ekstrim">
                <span class="stat-lbl">Peringatan Ekstrim H-5</span>
                <span class="stat-val">{{ stats['ekstrim_fc'] }}</span>
            </div>
        </div>
    </div>

    <!-- PETA SPASIAL LEAFLET.JS -->
    <div class="card">
        <div class="card-header-title">
            <span>🗺️ Visualisasi Spasial Sebaran Hotspot & Sentinel Kerawanan H-5</span>
            <span style="font-size:12px; color:#64748b; font-weight:normal;">Peta otomatis mengarahkan ke wilayah yang dipilih</span>
        </div>
        <div id="map"></div>
    </div>

    <!-- REKAP POTENSI KARHUTLA ANTAR-PROVINSI -->
    <div class="card" style="border-top: 4px solid #0284c7;">
        <div class="card-header-title">
            📊 Rekapitulasi Matriks Status Potensi Karhutla Antar-Provinsi (5 Provinsi)
        </div>
        <table>
            <thead>
                <tr>
                    <th>Provinsi</th>
                    <th>Status Siaga Utama</th>
                    <th>Titik Sentinel Ekstrim (H-5)</th>
                    <th>Titik Sentinel Siaga Tinggi</th>
                    <th>Titik Rawan Gambut</th>
                    <th>Jumlah Hotspot Terdeteksi</th>
                    <th>Rekomendasi Operasional Lapangan</th>
                </tr>
            </thead>
            <tbody>
                {% for p in prov_summary %}
                <tr>
                    <td><span class="prov-tag prov-{{ p['provinsi']|lower }}">{{ p['provinsi'] }}</span></td>
                    <td><span class="badge {{ p['badge_class'] }}">{{ p['status_label'] }}</span></td>
                    <td><b>{{ p['ekstrim_cnt'] }} Titik</b></td>
                    <td>{{ p['tinggi_cnt'] }} Titik</td>
                    <td>{% if p['gambut_cnt'] > 0 %}<span style="color:#ea580c; font-weight:700;">{{ p['gambut_cnt'] }} Titik</span>{% else %}-{% endif %}</td>
                    <td><b>{{ p['hs_count'] }} Titik</b></td>
                    <td style="font-size:12px; color:#475569;">
                        {% if p['provinsi'] == 'KALTENG' %}
                            Operasi TMC, pembasahan kubah gambut Sebangau & Eks-PLG, siaga water bombing.
                        {% elif p['provinsi'] == 'SUMSEL' %}
                            Patroli darat konsesi HTI OKI, monitoring tinggi muka air gambut Padang Sugihan.
                        {% elif p['provinsi'] == 'JAMBI' %}
                            Aktivasi sekat kanal TN Berbak, pengawasan kawasan trans-boundary Hutan Harapan.
                        {% elif p['provinsi'] == 'KALTIM' %}
                            Pembersihan sekat bakar batas IKN, monitoring areal mineral terbuka TN Kutai.
                        {% else %}
                            Pembersihan serasah savana TN Baluran & pencegahan kebakaran dedaunan KPH Jatim.
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <!-- TABEL DETAIL PREDIKSI H-5 & ANALISIS 3 ELEMEN -->
    <div class="card" style="border-left: 5px solid #e11d48;">
        <div class="card-header-title">
            ⚠️ Detail Peringatan Dini Cuaca & Analisis 3 Elemen Pembakar 5 Hari ke Depan (Open-Meteo Ensemble)
        </div>
        <table>
            <thead>
                <tr>
                    <th>Provinsi</th>
                    <th>Titik Sentinel / Kawasan Kritis</th>
                    <th>Koordinat</th>
                    <th>Tanggal Proyeksi</th>
                    <th>1. Suhu Puncak</th>
                    <th>2. Bahan Bakar / Hujan</th>
                    <th>3. Desikasi (RH / Angin)</th>
                    <th>Tipe Lahan</th>
                    <th>Indeks Risiko</th>
                    <th>Tingkat Ancaman</th>
                </tr>
            </thead>
            <tbody>
                {% for f in forecasts %}
                <tr>
                    <td><span class="prov-tag prov-{{ f['provinsi']|lower }}">{{ f['provinsi'] }}</span></td>
                    <td><b>{{ f['nama_kawasan'] }}</b></td>
                    <td>{{ "%.3f"|format(f['latitude']) }}, {{ "%.3f"|format(f['longitude']) }}</td>
                    <td>{{ f['forecast_date'] }}</td>
                    <td><b>{{ f['temp_max'] }}°C</b></td>
                    <td>
                        {{ f['precip_sum'] }} mm 
                        {% if f['precip_sum'] < 1.0 %}
                            <span style="color:#dc2626; font-size:11px; font-weight:bold;">(Kering)</span>
                        {% else %}
                            <span style="color:#16a34a; font-size:11px;">(Lembap)</span>
                        {% endif %}
                    </td>
                    <td>RH {{ f['rh_min'] }}% | {{ f['wind_max'] }} km/j</td>
                    <td>{% if f['is_gambut'] == 1 %}<span style="color:#ea580c; font-weight:700;">Gambut</span>{% else %}Mineral{% endif %}</td>
                    <td><b>{{ f['risk_score'] }}</b></td>
                    <td>
                        {% if f['risk_category'] == 'Ekstrim' %}
                            <span class="badge badge-ekstrim">🔥 EKSTRIM</span>
                        {% elif f['risk_category'] == 'Tinggi' %}
                            <span class="badge badge-tinggi">SIAGA TINGGI</span>
                        {% elif f['risk_category'] == 'Sedang' %}
                            <span class="badge badge-sedang">WASPADA</span>
                        {% else %}
                            <span class="badge badge-rendah">AMAN</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <!-- TABEL RIWAYAT HOTSPOT 10 HARI -->
    <div class="card">
        <div class="card-header-title">
            🛰️ Matriks Riwayat Titik Hotspot 10 Hari Terakhir (Multi-Satelit)
        </div>
        <table>
            <thead>
                <tr>
                    <th>No</th>
                    <th>Provinsi</th>
                    <th>Koordinat</th>
                    <th>Kabupaten</th>
                    <th>Fungsi Kawasan</th>
                    <th>Nama Kawasan / PBPH / Unit</th>
                    <th>Gambut</th>
                    <th>Sensor</th>
                    {% for d in date_columns %}
                        <th>{{ d }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in rows %}
                <tr>
                    <td>{{ row['No'] }}</td>
                    <td><span class="prov-tag prov-{{ row['provinsi']|lower }}">{{ row['provinsi'] }}</span></td>
                    <td><b>{{ "%.4f"|format(row['latitude']) }}, {{ "%.4f"|format(row['longitude']) }}</b></td>
                    <td>{{ row['kabupaten'] }}</td>
                    <td>{{ row['fungsi_kawasan'] }}</td>
                    <td><b>{{ row['nama_kawasan'] }}</b></td>
                    <td>{% if row['is_gambut_label'] == 'Ya' %}<span style="color:#ea580c; font-weight:700;">Gambut</span>{% else %}Mineral{% endif %}</td>
                    <td>{{ row['sensor_terdeteksi'] }}</td>
                    {% for d in date_columns %}
                        <td>
                            {% set st = row[d] %}
                            {% if st == 'Tinggi' %}
                                <span class="badge badge-tinggi">Tinggi</span>
                            {% elif st == 'Sedang' %}
                                <span class="badge badge-sedang">Sedang</span>
                            {% elif st == 'Rendah' %}
                                <span class="badge badge-rendah">Rendah</span>
                            {% else %}
                                <span class="badge badge-padam">Padam</span>
                            {% endif %}
                        </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <!-- SCRIPT LEAFLET & INTERAKSI LENGKAP -->
    <script>
        function applyFilters() {
            const prov = document.getElementById('provSelect').value;
            const source = document.getElementById('sourceSelect').value;
            const sat = document.getElementById('satSelect').value;
            window.location.href = '/?prov=' + prov + '&source=' + source + '&sat=' + sat;
        }

        function refreshData() {
            const btn = document.querySelector('.btn-refresh');
            btn.innerHTML = '⏳ Menarik Data Cuaca & Satelit...';
            btn.disabled = true;
            fetch('/api/sync')
                .then(r => r.json())
                .then(res => {
                    alert('Data hotspot dan prediksi cuaca 5 provinsi berhasil disinkronkan!');
                    window.location.reload();
                })
                .catch(e => {
                    alert('Gagal menyinkronkan data.');
                    btn.innerHTML = '🔄 Sinkronisasi Hotspot & Cuaca';
                    btn.disabled = false;
                });
        }

        // Inisialisasi Peta Leaflet.js
        const mapCenter = {{ map_center|tojson }};
        const map = L.map('map').setView([mapCenter[0], mapCenter[1]], mapCenter[2]);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 18,
            attribution: '© OpenStreetMap contributors | Karhutla Early Warning'
        }).addTo(map);

        const hotspotData = {{ rows|tojson }};
        const forecastData = {{ forecasts|tojson }};

        // Render Titik Hotspot (Circle Markers)
        hotspotData.forEach(pt => {
            const isPadam = pt.status_terakhir === 'Padam';
            const color = isPadam ? '#94a3b8' : (pt.status_terakhir === 'Tinggi' ? '#dc2626' : (pt.status_terakhir === 'Sedang' ? '#d97706' : '#16a34a'));
            const circle = L.circleMarker([pt.latitude, pt.longitude], {
                radius: isPadam ? 5 : 8,
                fillColor: color,
                color: '#ffffff',
                weight: 1.5,
                opacity: 1,
                fillOpacity: 0.85
            }).addTo(map);

            circle.bindPopup(`
                <div style="font-size:12px; min-width:200px;">
                    <strong style="color:${color}; font-size:13px;">Status: Hotspot ${pt.status_terakhir}</strong><br>
                    <b>Kawasan:</b> ${pt.nama_kawasan}<br>
                    <b>Kabupaten:</b> ${pt.kabupaten} (${pt.provinsi})<br>
                    <b>Fungsi:</b> ${pt.fungsi_kawasan}<br>
                    <b>Tipe Lahan:</b> ${pt.is_gambut_label === 'Ya' ? 'Rawa Gambut' : 'Tanah Mineral'}<br>
                    <b>Sensor Satelit:</b> ${pt.sensor_terdeteksi}
                </div>
            `);
        });

        // Render Titik Sentinel Prediksi H-5 (Popup 3 Elemen Pembakar)
        const sentinelRendered = new Set();
        forecastData.forEach(fc => {
            const key = fc.latitude + '_' + fc.longitude;
            if (!sentinelRendered.has(key)) {
                sentinelRendered.add(key);
                const isEkstrim = fc.risk_category === 'Ekstrim';
                const pulseColor = isEkstrim ? '#dc2626' : '#ea580c';
                
                const marker = L.circleMarker([fc.latitude, fc.longitude], {
                    radius: 12,
                    fillColor: pulseColor,
                    color: '#ffffff',
                    weight: 2,
                    dashArray: '3, 3',
                    fillOpacity: 0.4
                }).addTo(map);

                marker.bindPopup(`
                    <div style="font-size:12px; min-width:230px;">
                        <span style="background:${pulseColor}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;">
                            🔥 Ancaman H-5: ${fc.risk_category} (${fc.risk_score}/100)
                        </span><br>
                        <b style="font-size:13px; display:inline-block; margin-top:6px;">${fc.nama_kawasan}</b><br>
                        <span style="color:#64748b;">${fc.provinsi} | ${fc.is_gambut === 1 ? 'Rawa Gambut' : 'Tanah Mineral'}</span>
                        <hr style="border:0; border-top:1px solid #e2e8f0; margin:6px 0;">
                        <b>Analisis 3 Elemen Pembakar:</b><br>
                        🔥 <b>1. Suhu Puncak:</b> ${fc.temp_max}°C (${fc.temp_max >= 34 ? '<span style="color:red; font-weight:bold;">Kritis</span>' : 'Normal'})<br>
                        🍂 <b>2. Bahan Bakar / Hujan:</b> ${fc.precip_sum} mm (${fc.precip_sum < 1.0 ? '<span style="color:red; font-weight:bold;">Kering Ekstrim</span>' : 'Lembap'})<br>
                        💨 <b>3. Desikasi (RH & Angin):</b> RH ${fc.rh_min}% | Angin ${fc.wind_max} km/j
                    </div>
                `);
            }
        });

        // Legenda Spasial Peta
        const legend = L.control({position: 'bottomright'});
        legend.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'legend-box');
            div.innerHTML = `
                <b>Legenda Spasial</b><br>
                <span class="legend-dot" style="background:#dc2626;"></span> Hotspot Risiko Tinggi<br>
                <span class="legend-dot" style="background:#d97706;"></span> Hotspot Risiko Sedang<br>
                <span class="legend-dot" style="background:#94a3b8;"></span> Hotspot Padam (Riwayat)<br>
                <span class="legend-dot" style="background:#ea580c; border:1px dashed #ffffff;"></span> Titik Sentinel Prediksi H-5
            `;
            return div;
        };
        legend.addTo(map);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    prov = request.args.get("prov", "ALL")
    source = request.args.get("source", "ALL_DATA")
    sat = request.args.get("sat", "ALL")
    status = request.args.get("status", "ALL")

    ref_d, session_name = get_effective_date_and_session()
    ref_str = ref_d.strftime("%Y-%m-%d")

    df, date_list, stats = generate_table_multi(
        ref_d,
        filter_prov=prov,
        filter_sat=sat,
        filter_status=status,
        filter_source=source,
    )
    rows = df.to_dict(orient="records") if not df.empty else []
    forecasts = get_forecast_summary_multi(filter_prov=prov)
    prov_summary = get_provincial_potential_summary()
    map_center = PROV_CENTERS.get(prov, PROV_CENTERS["ALL"])

    return render_template_string(
        HTML_TEMPLATE,
        rows=rows,
        forecasts=forecasts,
        prov_summary=prov_summary,
        map_center=map_center,
        date_columns=date_list,
        ref_date=ref_str,
        session_name=session_name,
        current_prov=prov,
        current_source=source,
        current_sat=sat,
        stats=stats,
    )


@app.route("/api/sync")
def api_sync():
    ref_d, session_name = get_effective_date_and_session()
    ref_str = ref_d.strftime("%Y-%m-%d")
    sync_hotspots_multi(ref_str)
    update_fire_risk_forecast_multi()
    return jsonify(
        {"status": "ok", "effective_date": ref_str, "session": session_name}
    )


@app.route("/download-excel")
def download_excel():
    prov = request.args.get("prov", "ALL")
    sat = request.args.get("sat", "ALL")
    source = request.args.get("source", "ALL_DATA")
    ref_d, _ = get_effective_date_and_session()
    ref_str = ref_d.strftime("%Y-%m-%d")
    out_file = export_to_excel_multi(
        ref_str, filter_prov=prov, filter_sat=sat, filter_source=source
    )
    if os.path.exists(out_file):
        return send_file(out_file, as_attachment=True, download_name=out_file)
    return "File Excel belum tersedia", 404


# ==========================================
# 9. INISIALISASI OTOMATIS (LOKAL & SERVER CLOUD)
# ==========================================
init_db()
seed_historical_multi()
update_fire_risk_forecast_multi()

t = threading.Thread(target=run_scheduler, daemon=True)
t.start()

try:
    initial_date, _ = get_effective_date_and_session()
    export_to_excel_multi(initial_date.strftime("%Y-%m-%d"))
except Exception:
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
