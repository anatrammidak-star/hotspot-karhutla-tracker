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
# KONFIGURASI SISTEM MULTI-PROVINSI
# ==========================================
DB_NAME = "hotspot_multi.db"
DEFAULT_NASA_KEY = "DEMO_KEY"

# Bounding Box Resmi (min_lon, min_lat, max_lon, max_lat)
PROVINSI_BBOX = {
    "JAMBI": "101.15,-2.50,104.55,-0.75",
    "SUMSEL": "102.05,-4.95,106.15,-1.65",
    "JATIM": "110.85,-8.80,114.65,-6.75",
}

PJKARHUTLA_BASE_URL = "https://pjkarhutla-del.github.io/Monitor-Hotspot-HK"


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
# 2. OVERLAY KAWASAN SPESIFIK 3 PROVINSI
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

        # ---------------- JAMBI ----------------
        if prov == "JAMBI":
            if lon <= 101.90 and lat <= -1.50:
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
            elif lon >= 104.00 and -1.55 <= lat <= -1.15:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Berbak (TNB)",
                    1,
                    "Tanjung Jabung Timur",
                ])
            elif 103.00 <= lon <= 103.35 and -1.25 <= lat <= -0.90:
                return pd.Series([
                    "Hutan Lindung (HL)",
                    "HL Gambut Bram Itam",
                    1,
                    "Tanjung Jabung Barat",
                ])
            elif 103.15 <= lon <= 103.55 and -2.40 <= lat <= -2.00:
                return pd.Series([
                    "PBPH (Restorasi Ekosistem)",
                    "PBPH PT REKI (Hutan Harapan)",
                    0,
                    "Batanghari / Sarolangun",
                ])
            elif 103.35 <= lon <= 103.75 and -1.60 <= lat <= -1.20:
                return pd.Series(
                    ["PBPH (Hutan Tanaman)", "PBPH PT WKS", 1, "Muaro Jambi"]
                )
            else:
                is_g = 1 if (lon >= 103.50 and lat >= -1.70) else 0
                g_str = "Gambut" if is_g == 1 else "Mineral"
                kab = (
                    "Batanghari"
                    if lon < 103.3
                    else (
                        "Muaro Jambi" if lat < -1.5 else "Tanjung Jabung Barat"
                    )
                )
                return pd.Series([
                    f"APL (Lahan {g_str})",
                    "Areal Perkebunan / Lahan Masyarakat",
                    is_g,
                    kab,
                ])

        # ------------ SUMATERA SELATAN ------------
        elif prov == "SUMSEL":
            # TN Sembilang / Lanskap Pesisir Banyuasin
            if 104.40 <= lon <= 105.10 and -2.40 <= lat <= -1.70:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Sembilang",
                    1,
                    "Banyuasin",
                ])
            # SM Padang Sugihan (Gambut OKI / Banyuasin)
            elif 104.80 <= lon <= 105.35 and -3.20 <= lat <= -2.60:
                return pd.Series([
                    "KSA (Suaka Margasatwa)",
                    "SM Padang Sugihan",
                    1,
                    "Ogan Komering Ilir",
                ])
            # PBPH Gambut OKI (Kawasan Hutan Produksi Tanaman Industri)
            elif lon >= 105.20 and -3.60 <= lat <= -2.80:
                return pd.Series([
                    "PBPH (Hutan Tanaman)",
                    "PBPH HTI Gambut OKI",
                    1,
                    "Ogan Komering Ilir",
                ])
            # PBPH Musi Banyuasin (MUBA)
            elif 103.40 <= lon <= 104.30 and -2.90 <= lat <= -2.10:
                return pd.Series([
                    "PBPH (Hutan Produksi)",
                    "PBPH Wilayah Musi Banyuasin",
                    0,
                    "Musi Banyuasin",
                ])
            # Hutan Lindung Bukit Balai Rejang / Lahat / Muara Enim
            elif lon <= 103.60 and lat <= -3.60:
                return pd.Series([
                    "Hutan Lindung (HL)",
                    "HL Bukit Balai Rejang",
                    0,
                    "Lahat / Muara Enim",
                ])
            else:
                is_g = 1 if lon >= 104.70 else 0
                g_str = "Gambut" if is_g == 1 else "Mineral"
                kab = (
                    "Ogan Komering Ilir"
                    if lon >= 104.90
                    else ("Banyuasin" if lat >= -2.8 else "Ogan Ilir")
                )
                return pd.Series([
                    f"APL (Lahan {g_str})",
                    "Areal Perkebunan Sawit/Karet",
                    is_g,
                    kab,
                ])

        # -------------- JAWA TIMUR --------------
        elif prov == "JATIM":
            # TN Bromo Tengger Semeru (TNBTS)
            if 112.80 <= lon <= 113.15 and -8.15 <= lat <= -7.85:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Bromo Tengger Semeru (TNBTS)",
                    0,
                    "Malang / Probolinggo / Pasuruan",
                ])
            # TN Baluran (Savana & Hutan Musim Situbondo)
            elif 114.25 <= lon <= 114.55 and -7.90 <= lat <= -7.70:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Baluran",
                    0,
                    "Situbondo",
                ])
            # TN Meru Betiri (Jember / Banyuwangi)
            elif 113.60 <= lon <= 114.00 and -8.60 <= lat <= -8.30:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Meru Betiri",
                    0,
                    "Jember / Banyuwangi",
                ])
            # TN Alas Purwo (Ujung Timur Banyuwangi)
            elif lon >= 114.30 and -8.80 <= lat <= -8.50:
                return pd.Series([
                    "KPA (Taman Nasional)",
                    "TN Alas Purwo",
                    0,
                    "Banyuwangi",
                ])
            # Tahura Raden Soerjo (Mojokerto / Batu / Pasuruan)
            elif 112.45 <= lon <= 112.70 and -7.80 <= lat <= -7.65:
                return pd.Series([
                    "KPA (Tahura)",
                    "Tahura Raden Soerjo",
                    0,
                    "Mojokerto / Batu",
                ])
            # KPH Perhutani (Hutan Produksi Jati / Pinus Jawa Timur)
            elif (
                (111.40 <= lon <= 112.20 and -7.60 <= lat <= -7.10)
                or (111.80 <= lon <= 112.50 and -8.20 <= lat <= -7.80)
            ):
                kab = (
                    "Bojonegoro / Ngawi" if lat > -7.5 else "Ponorogo / Trenggalek"
                )
                return pd.Series(
                    ["KPH Perhutani", "Hutan Produksi Perhutani Divre Jatim", 0, kab]
                )
            else:
                kab = (
                    "Madiun / Magetan"
                    if lon < 111.8
                    else (
                        "Jombang / Mojokerto" if lon < 112.6 else "Banyuwangi"
                    )
                )
                return pd.Series([
                    "APL (Lahan Mineral)",
                    "Tegalan / Pertanian / Pemukiman",
                    0,
                    kab,
                ])

        return pd.Series(["Lainnya", "Area Luar Wilayah", 0, "Kabupaten"])

    gdf[["fungsi_kawasan", "nama_kawasan", "is_gambut", "kabupaten"]] = (
        gdf.apply(classify_row, axis=1)
    )
    return gdf


# ==========================================
# 3. SPATIAL CLUSTERING (~500 meter per Provinsi)
# ==========================================
def match_or_create_clusters_multi(conn, new_gdf, threshold_km=0.5):
    c = conn.cursor()
    matched_ids = []

    for prov in ["JAMBI", "SUMSEL", "JATIM"]:
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
        deg_to_km = 111.0
        tree = cKDTree(np.array(ex_coords) * deg_to_km)

        for idx, row in prov_gdf.iterrows():
            pt = np.array([row.latitude, row.longitude]) * deg_to_km
            dist, index = tree.query(pt, distance_upper_bound=threshold_km)

            if dist < threshold_km:
                cid = ex_ids[index]
                c.execute(
                    """
                    UPDATE hotspot_clusters 
                    SET last_detected = ?, consecutive_padam_days = 0 
                    WHERE cluster_id = ?
                """,
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
    # Petakan id kembali sesuai urutan indeks dataframe
    id_dict = {orig_idx: cid for cid, orig_idx in matched_ids}
    return [id_dict.get(idx, f"HS-GEN-{idx}") for idx in new_gdf.index]


# ==========================================
# 4. KONEKTOR PENARIKAN DATA NASA & GITHUB HK
# ==========================================
def fetch_satellite_data_multi(target_date_str, target_prov="ALL"):
    map_key = get_nasa_key()
    prov_keys = (
        ["JAMBI", "SUMSEL", "JATIM"]
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
                        if c_raw == "h":
                            c_val = 90.0
                        elif c_raw == "n":
                            c_val = 55.0
                        elif c_raw == "l":
                            c_val = 20.0
                        else:
                            try:
                                c_val = float(c_raw)
                            except ValueError:
                                c_val = 50.0

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
        if conf >= 80:
            return "Tinggi"
        if conf >= 30:
            return "Sedang"
        return "Rendah"

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

    # Identifikasi cluster yang padam
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
# 5. SEED DATA HISTORIS 3 PROVINSI (14 - 23 SEPT 2026)
# ==========================================
def seed_historical_multi():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM hotspot_clusters")
    count = c.fetchone()[0]

    if count == 0:
        clusters = [
            # --- JAMBI ---
            {
                "id": "HS-JAMBI-01",
                "prov": "JAMBI",
                "lat": -1.3520,
                "lon": 104.1020,
                "kab": "Tanjung Jabung Timur",
                "f_kws": "KPA (Taman Nasional)",
                "n_kws": "TN Berbak (TNB)",
                "gambut": 1,
                "first": "2026-09-16",
                "last": "2026-09-23",
                "padam_cnt": 0,
                "logs": {
                    "2026-09-16": ("Sedang", 65, "Terra/Aqua"),
                    "2026-09-17": ("Sedang", 70, "Terra/Aqua"),
                    "2026-09-18": ("Tinggi", 85, "Terra/Aqua"),
                    "2026-09-19": ("Tinggi", 88, "HK-PJKarhutla"),
                    "2026-09-20": ("Tinggi", 90, "HK-PJKarhutla"),
                    "2026-09-21": ("Tinggi", 92, "HK-PJKarhutla"),
                    "2026-09-22": ("Tinggi", 94, "HK-PJKarhutla"),
                    "2026-09-23": ("Tinggi", 93, "HK-PJKarhutla"),
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
                "first": "2026-09-14",
                "last": "2026-09-16",
                "padam_cnt": 7,
                "logs": {
                    "2026-09-14": ("Sedang", 60, "Terra/Aqua"),
                    "2026-09-15": ("Tinggi", 82, "NOAA-20"),
                    "2026-09-16": ("Sedang", 55, "Terra/Aqua"),
                    "2026-09-17": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-18": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-19": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-20": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-21": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-22": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-23": ("Padam", 0, "Terra/Aqua"),
                },
            },
            {
                "id": "HS-JAMBI-03",
                "prov": "JAMBI",
                "lat": -1.8900,
                "lon": 102.6100,
                "kab": "Sarolangun / Batanghari",
                "f_kws": "KPA (Taman Nasional)",
                "n_kws": "TN Bukit Duabelas (TNBD)",
                "gambut": 0,
                "first": "2026-09-18",
                "last": "2026-09-23",
                "padam_cnt": 0,
                "logs": {
                    "2026-09-18": ("Sedang", 55, "Terra/Aqua"),
                    "2026-09-19": ("Sedang", 72, "Terra/Aqua"),
                    "2026-09-20": ("Tinggi", 86, "HK-PJKarhutla"),
                    "2026-09-21": ("Tinggi", 91, "HK-PJKarhutla"),
                    "2026-09-22": ("Tinggi", 92, "HK-PJKarhutla"),
                    "2026-09-23": ("Tinggi", 94, "HK-PJKarhutla"),
                },
            },
            # --- SUMATERA SELATAN ---
            {
                "id": "HS-SUMSEL-01",
                "prov": "SUMSEL",
                "lat": -3.3500,
                "lon": 105.4200,
                "kab": "Ogan Komering Ilir",
                "f_kws": "PBPH (Hutan Tanaman)",
                "n_kws": "PBPH HTI Gambut OKI",
                "gambut": 1,
                "first": "2026-09-17",
                "last": "2026-09-23",
                "padam_cnt": 0,
                "logs": {
                    "2026-09-17": ("Sedang", 65, "Terra/Aqua"),
                    "2026-09-18": ("Tinggi", 88, "NOAA-20"),
                    "2026-09-19": ("Tinggi", 92, "NOAA-20"),
                    "2026-09-20": ("Tinggi", 95, "NOAA-20"),
                    "2026-09-21": ("Tinggi", 96, "NOAA-20"),
                    "2026-09-22": ("Tinggi", 91, "NOAA-20"),
                    "2026-09-23": ("Tinggi", 93, "NOAA-20"),
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
                "first": "2026-09-15",
                "last": "2026-09-18",
                "padam_cnt": 5,
                "logs": {
                    "2026-09-15": ("Sedang", 55, "Terra/Aqua"),
                    "2026-09-16": ("Tinggi", 84, "HK-PJKarhutla"),
                    "2026-09-17": ("Sedang", 50, "HK-PJKarhutla"),
                    "2026-09-18": ("Rendah", 25, "Terra/Aqua"),
                    "2026-09-19": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-20": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-21": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-22": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-23": ("Padam", 0, "Terra/Aqua"),
                },
            },
            {
                "id": "HS-SUMSEL-03",
                "prov": "SUMSEL",
                "lat": -2.1500,
                "lon": 104.7500,
                "kab": "Banyuasin",
                "f_kws": "KPA (Taman Nasional)",
                "n_kws": "TN Sembilang",
                "gambut": 1,
                "first": "2026-09-21",
                "last": "2026-09-23",
                "padam_cnt": 0,
                "logs": {
                    "2026-09-21": ("Sedang", 68, "HK-PJKarhutla"),
                    "2026-09-22": ("Tinggi", 85, "HK-PJKarhutla"),
                    "2026-09-23": ("Tinggi", 89, "HK-PJKarhutla"),
                },
            },
            # --- JAWA TIMUR ---
            {
                "id": "HS-JATIM-01",
                "prov": "JATIM",
                "lat": -7.9400,
                "lon": 112.9500,
                "kab": "Malang / Probolinggo / Pasuruan",
                "f_kws": "KPA (Taman Nasional)",
                "n_kws": "TN Bromo Tengger Semeru (TNBTS)",
                "gambut": 0,
                "first": "2026-09-19",
                "last": "2026-09-23",
                "padam_cnt": 0,
                "logs": {
                    "2026-09-19": ("Sedang", 55, "Terra/Aqua"),
                    "2026-09-20": ("Sedang", 72, "Terra/Aqua"),
                    "2026-09-21": ("Tinggi", 86, "HK-PJKarhutla"),
                    "2026-09-22": ("Tinggi", 88, "HK-PJKarhutla"),
                    "2026-09-23": ("Tinggi", 91, "HK-PJKarhutla"),
                },
            },
            {
                "id": "HS-JATIM-02",
                "prov": "JATIM",
                "lat": -7.8200,
                "lon": 114.4200,
                "kab": "Situbondo",
                "f_kws": "KPA (Taman Nasional)",
                "n_kws": "TN Baluran",
                "gambut": 0,
                "first": "2026-09-16",
                "last": "2026-09-19",
                "padam_cnt": 4,
                "logs": {
                    "2026-09-16": ("Sedang", 60, "Terra/Aqua"),
                    "2026-09-17": ("Tinggi", 84, "NOAA-20"),
                    "2026-09-18": ("Sedang", 50, "Terra/Aqua"),
                    "2026-09-19": ("Rendah", 24, "Terra/Aqua"),
                    "2026-09-20": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-21": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-22": ("Padam", 0, "Terra/Aqua"),
                    "2026-09-23": ("Padam", 0, "Terra/Aqua"),
                },
            },
            {
                "id": "HS-JATIM-03",
                "prov": "JATIM",
                "lat": -7.3800,
                "lon": 111.7500,
                "kab": "Bojonegoro / Ngawi",
                "f_kws": "KPH Perhutani",
                "n_kws": "Hutan Produksi Perhutani Divre Jatim",
                "gambut": 0,
                "first": "2026-09-20",
                "last": "2026-09-23",
                "padam_cnt": 0,
                "logs": {
                    "2026-09-20": ("Rendah", 28, "Terra/Aqua"),
                    "2026-09-21": ("Sedang", 55, "Terra/Aqua"),
                    "2026-09-22": ("Sedang", 62, "Terra/Aqua"),
                    "2026-09-23": ("Sedang", 68, "Terra/Aqua"),
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

            base_date = date(2026, 9, 14)
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
# 6. LOGIKA CUT-OFF & PIVOT 10 HARI
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
    }

    if df_clusters.empty:
        conn.close()
        return pd.DataFrame(), date_list, summary_stats

    sat_condition = ""
    if filter_sat == "NOAA":
        sat_condition = "AND satellite_source LIKE '%NOAA%'"
    elif filter_sat == "TERRA":
        sat_condition = "AND satellite_source LIKE '%Terra%'"

    if filter_source == "GITHUB_HK":
        df_clusters = df_clusters[
            df_clusters["fungsi_kawasan"].str.contains(
                "KPA|KSA|Tahura", na=False
            )
        ]

    query_logs = f"""
        SELECT cluster_id, log_date, status, confidence, satellite_source
        FROM hotspot_daily_log 
        WHERE log_date >= '{min_date}' AND log_date <= '{date_list[-1]}' {sat_condition}
    """
    df_logs = pd.read_sql_query(query_logs, conn)
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
    file_name = (
        f"Laporan_Hotspot_{filter_prov}.xlsx"
        if filter_prov != "ALL"
        else "Laporan_Hotspot_MultiProvinsi.xlsx"
    )
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


def run_scheduler():
    schedule.every().day.at("07:00").do(scheduled_sync_multi)
    schedule.every().day.at("18:00").do(scheduled_sync_multi)
    while True:
        schedule.run_pending()
        time.sleep(30)


# ==========================================
# 8. DASBOR WEB MULTI-PROVINSI (FLASK)
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Sistem Pemantauan Hotspot Multi-Provinsi (Jambi, Sumsel, Jatim)</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 24px; background: #f8fafc; color: #1e293b; }
        .header { display: flex; justify-content: space-between; align-items: center; background: #fff; padding: 20px 28px; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }
        h1 { margin: 0; font-size: 22px; color: #0f172a; }
        .subtitle { font-size: 13px; color: #64748b; margin-top: 4px; }
        .schedule-badge { background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 6px; font-weight: 700; }
        
        .controls { display: flex; gap: 12px; align-items: center; background: white; padding: 14px 24px; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; flex-wrap: wrap; }
        select { padding: 9px 12px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 13px; font-weight: 600; background: white; color: #0f172a; }
        .select-prov { background: #f0fdf4; border-color: #86efac; color: #166534; }
        
        button, a.btn { padding: 9px 16px; border-radius: 8px; font-weight: 600; text-decoration: none; cursor: pointer; border: none; font-size: 13px; display: inline-flex; align-items: center; gap: 6px; }
        .btn-refresh { background: #0284c7; color: white; }
        .btn-excel { background: #16a34a; color: white; }
        .btn-key { background: #64748b; color: white; }
        
        .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow-x: auto; margin-bottom: 24px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #f1f5f9; white-space: nowrap; }
        th { background: #f8fafc; color: #475569; font-weight: 700; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
        
        .badge { padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; text-align: center; }
        .badge-tinggi { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
        .badge-sedang { background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }
        .badge-rendah { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
        .badge-padam { background: #f1f5f9; color: #94a3b8; border: 1px solid #e2e8f0; }
        
        .prov-tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
        .prov-jambi { background: #e0f2fe; color: #0284c7; }
        .prov-sumsel { background: #fef3c7; color: #d97706; }
        .prov-jatim { background: #dcfce7; color: #15803d; }
        
        .kpa { color: #dc2626; font-weight: bold; }
        .hl { color: #0284c7; font-weight: bold; }
        .pbph { color: #059669; font-weight: bold; }
        .gambut-ya { background: #ffedd5; color: #c2410c; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }
        .sensor-tag { background: #ede9fe; color: #6d28d9; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }
        
        .summary-container { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
        .summary-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; }
        .summary-title { font-size: 16px; font-weight: 700; color: #0f172a; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; }
        .stat-card { padding: 16px; border-radius: 10px; border: 1px solid #e2e8f0; background: #fafafa; display: flex; flex-direction: column; justify-content: space-between; }
        .stat-val { font-size: 26px; font-weight: 800; margin-top: 4px; }
        .stat-lbl { font-size: 12px; font-weight: 600; text-transform: uppercase; color: #64748b; }
        
        .stat-tinggi { background: #fef2f2; border-color: #fecaca; }
        .stat-tinggi .stat-val { color: #dc2626; }
        .stat-sedang { background: #fffbeb; border-color: #fde68a; }
        .stat-sedang .stat-val { color: #d97706; }
        .stat-rendah { background: #f0fdf4; border-color: #bbf7d0; }
        .stat-rendah .stat-val { color: #16a34a; }
        .stat-padam { background: #f8fafc; border-color: #e2e8f0; }
        .stat-padam .stat-val { color: #64748b; }
        .stat-konservasi { background: #fff1f2; border-color: #ffe4e6; }
        .stat-konservasi .stat-val { color: #e11d48; }
        .stat-gambut { background: #fff7ed; border-color: #ffedd5; }
        .stat-gambut .stat-val { color: #ea580c; }
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>Sistem Pemantauan Hotspot Multi-Provinsi</h1>
            <div class="subtitle">
                Cakupan Wilayah: <strong>Jambi, Sumatera Selatan, Jawa Timur</strong> | 
                Jadwal: <span class="schedule-badge">07:00 & 18:00 WIB</span> | 
                Status: <strong>{{ session_name }} ({{ ref_date }})</strong>
            </div>
        </div>
        <div>
            <a href="/download-excel?prov={{ current_prov }}&sat={{ current_sat }}&source={{ current_source }}" class="btn btn-excel">📥 Unduh Excel ({{ current_prov }})</a>
        </div>
    </div>

    <div class="controls">
        <label><strong>Pilih Provinsi:</strong></label>
        <select id="provSelect" class="select-prov" onchange="applyFilters()">
            <option value="ALL" {% if current_prov == 'ALL' %}selected{% endif %}>🇮🇩 Semua Provinsi (Rekap Gabungan)</option>
            <option value="JAMBI" {% if current_prov == 'JAMBI' %}selected{% endif %}>📍 Provinsi Jambi</option>
            <option value="SUMSEL" {% if current_prov == 'SUMSEL' %}selected{% endif %}>📍 Provinsi Sumatera Selatan</option>
            <option value="JATIM" {% if current_prov == 'JATIM' %}selected{% endif %}>📍 Provinsi Jawa Timur</option>
        </select>

        <label><strong>Sumber Feed:</strong></label>
        <select id="sourceSelect" onchange="applyFilters()">
            <option value="ALL_DATA" {% if current_source == 'ALL_DATA' %}selected{% endif %}>🌐 Seluruh Lanskap (HK, HL, HP/PBPH, APL)</option>
            <option value="GITHUB_HK" {% if current_source == 'GITHUB_HK' %}selected{% endif %}>🌲 Khusus Hutan Konservasi (TN/CA/SM/Tahura)</option>
        </select>

        <label><strong>Satelit:</strong></label>
        <select id="satSelect" onchange="applyFilters()">
            <option value="ALL" {% if current_sat == 'ALL' %}selected{% endif %}>🌐 Semua Satelit (Gabungan)</option>
            <option value="NOAA" {% if current_sat == 'NOAA' %}selected{% endif %}>🛰️ NOAA-20 / SNPP (375m)</option>
            <option value="TERRA" {% if current_sat == 'TERRA' %}selected{% endif %}>🛰️ Terra / Aqua (1km)</option>
        </select>

        <label><strong>Status:</strong></label>
        <select id="statusSelect" onchange="applyFilters()">
            <option value="ALL" {% if current_status == 'ALL' %}selected{% endif %}>Semua Status</option>
            <option value="AKTIF" {% if current_status == 'AKTIF' %}selected{% endif %}>🔥 Titik Aktif Saja</option>
            <option value="TINGGI" {% if current_status == 'TINGGI' %}selected{% endif %}>🔴 Hanya Risiko Tinggi</option>
            <option value="SEDANG_RENDAH" {% if current_status == 'SEDANG_RENDAH' %}selected{% endif %}>🟡/🟢 Sedang & Rendah</option>
        </select>

        <button class="btn btn-refresh" onclick="refreshData()">🔄 Sinkronisasi Sekarang</button>
        <button class="btn btn-key" onclick="updateApiKey()">🔑 NASA Key</button>
    </div>

    <div class="card">
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
                    <td>
                        <span class="prov-tag prov-{{ row['provinsi']|lower }}">{{ row['provinsi'] }}</span>
                    </td>
                    <td><b>{{ "%.4f"|format(row['latitude']) }}, {{ "%.4f"|format(row['longitude']) }}</b></td>
                    <td>{{ row['kabupaten'] }}</td>
                    <td>
                        <span class="{% if 'KPA' in row['fungsi_kawasan'] or 'KSA' in row['fungsi_kawasan'] %}kpa{% elif 'Hutan Lindung' in row['fungsi_kawasan'] %}hl{% elif 'PBPH' in row['fungsi_kawasan'] or 'Perhutani' in row['fungsi_kawasan'] %}pbph{% endif %}">
                            {{ row['fungsi_kawasan'] }}
                        </span>
                    </td>
                    <td>{{ row['nama_kawasan'] }}</td>
                    <td>
                        {% if row['is_gambut_label'] == 'Ya' %}
                            <span class="gambut-ya">Gambut</span>
                        {% else %}
                            <span style="color:#94a3b8;">Mineral</span>
                        {% endif %}
                    </td>
                    <td><span class="sensor-tag">{{ row['sensor_terdeteksi'] }}</span></td>
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
                {% if rows|length == 0 %}
                <tr>
                    <td colspan="{{ 8 + date_columns|length }}" style="text-align:center; padding: 40px; color: #94a3b8;">
                        Tidak ada titik hotspot yang terpantau pada wilayah dan filter yang dipilih.
                    </td>
                </tr>
                {% endif %}
            </tbody>
        </table>
    </div>

    <div class="summary-container">
        <div class="summary-header">
            <div class="summary-title">
                📊 Ringkasan Status Hotspot 
                <span style="font-weight:normal; font-size:13px; color:#64748b;">
                    (Wilayah: <b>{{ current_prov }}</b> | Rujukan: 
                    {% if current_sat == 'ALL' %}<b>Semua Satelit</b>
                    {% elif current_sat == 'NOAA' %}<b>NOAA-20/SNPP (375m)</b>
                    {% else %}<b>Terra/Aqua (1km)</b>{% endif %})
                </span>
            </div>
        </div>

        <div class="summary-grid">
            <div class="stat-card">
                <span class="stat-lbl">Total Titik Aktif</span>
                <span class="stat-val" style="color: #0284c7;">{{ stats['aktif'] }}</span>
            </div>
            <div class="stat-card stat-tinggi">
                <span class="stat-lbl">Tinggi / High (≥80%)</span>
                <span class="stat-val">{{ stats['tinggi'] }}</span>
            </div>
            <div class="stat-card stat-sedang">
                <span class="stat-lbl">Sedang / Medium (30-79%)</span>
                <span class="stat-val">{{ stats['sedang'] }}</span>
            </div>
            <div class="stat-card stat-rendah">
                <span class="stat-lbl">Rendah / Low (&lt;30%)</span>
                <span class="stat-val">{{ stats['rendah'] }}</span>
            </div>
            <div class="stat-card stat-padam">
                <span class="stat-lbl">Padam (Hari Ini)</span>
                <span class="stat-val">{{ stats['padam'] }}</span>
            </div>
            <div class="stat-card stat-konservasi">
                <span class="stat-lbl">Di Kawasan Konservasi</span>
                <span class="stat-val">{{ stats['konservasi'] }}</span>
            </div>
            <div class="stat-card stat-gambut">
                <span class="stat-lbl">Di Lahan Gambut</span>
                <span class="stat-val">{{ stats['gambut'] }}</span>
            </div>
        </div>
    </div>

    <script>
        function applyFilters() {
            const prov = document.getElementById('provSelect').value;
            const source = document.getElementById('sourceSelect').value;
            const sat = document.getElementById('satSelect').value;
            const status = document.getElementById('statusSelect').value;
            window.location.href = '/?prov=' + prov + '&source=' + source + '&sat=' + sat + '&status=' + status;
        }

        function refreshData() {
            const btn = document.querySelector('.btn-refresh');
            btn.innerHTML = '⏳ Menarik Data...';
            btn.disabled = true;
            fetch('/api/sync')
                .then(r => r.json())
                .then(res => {
                    alert('Data multi-provinsi berhasil disinkronkan (' + res.session + ')!');
                    window.location.reload();
                })
                .catch(e => {
                    alert('Gagal menyinkronkan data.');
                    btn.innerHTML = '🔄 Sinkronisasi Sekarang';
                    btn.disabled = false;
                });
        }

        function updateApiKey() {
            const current = prompt("Masukkan NASA FIRMS MAP_KEY Anda:");
            if (current) {
                fetch('/api/set-key', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({nasa_key: current.trim()})
                })
                .then(r => r.json())
                .then(res => {
                    alert("NASA Map Key berhasil diperbarui!");
                    refreshData();
                });
            }
        }
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
    return render_template_string(
        HTML_TEMPLATE,
        rows=rows,
        date_columns=date_list,
        ref_date=ref_str,
        session_name=session_name,
        current_prov=prov,
        current_source=source,
        current_sat=sat,
        current_status=status,
        stats=stats,
    )


@app.route("/api/sync")
def api_sync():
    ref_d, session_name = get_effective_date_and_session()
    ref_str = ref_d.strftime("%Y-%m-%d")
    sync_hotspots_multi(ref_str)
    return jsonify(
        {"status": "ok", "effective_date": ref_str, "session": session_name}
    )


@app.route("/api/set-key", methods=["POST"])
def api_set_key():
    data = request.get_json() or {}
    key = data.get("nasa_key")
    if key:
        set_nasa_key(key)
        return jsonify({"status": "updated"})
    return jsonify({"status": "error"}), 400


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
# Jalankan inisialisasi database dan seed data secara langsung agar Gunicorn mengeksekusinya
init_db()
seed_historical_multi()

# Jalankan background scheduler untuk penarikan 07:00 & 18:00 WIB
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

