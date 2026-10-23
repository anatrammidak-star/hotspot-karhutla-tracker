import io
import json
import math
import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from flask import Flask, jsonify, render_template_string, request, send_file
import pandas as pd
import requests

# ==========================================
# KONFIGURASI SISTEM (5 PROVINSI)
# ==========================================
DB_NAME = "hotspot_multi.db"
DEFAULT_NASA_KEY = "cf5d1449942e6c851eba10358106ea6f"

# Bounding Box: min_lon, min_lat, max_lon, max_lat
PROVINSI_BBOX = {
    "JAMBI": "101.15,-2.50,104.55,-0.75",
    "SUMSEL": "102.05,-4.95,106.15,-1.65",
    "JATIM": "110.85,-8.80,114.65,-6.75",
    "KALTENG": "110.75,-3.55,115.85,0.75",
    "KALTIM": "113.80,-2.45,119.05,2.85",
}

SENTINEL_POINTS = [
    # Jambi
    {"prov": "JAMBI", "kawasan": "TN Berbak (Pesisir Gambut)", "lat": -1.352, "lon": 104.102, "gambut": 1},
    {"prov": "JAMBI", "kawasan": "HL Gambut Bram Itam", "lat": -1.120, "lon": 103.180, "gambut": 1},
    {"prov": "JAMBI", "kawasan": "PBPH PT REKI (Hutan Harapan)", "lat": -2.210, "lon": 103.350, "gambut": 0},
    # Sumatera Selatan
    {"prov": "SUMSEL", "kawasan": "PBPH Gambut OKI (Air Sugihan)", "lat": -3.350, "lon": 105.420, "gambut": 1},
    {"prov": "SUMSEL", "kawasan": "SM Padang Sugihan", "lat": -2.950, "lon": 105.050, "gambut": 1},
    {"prov": "SUMSEL", "kawasan": "TN Sembilang Pesisir", "lat": -2.150, "lon": 104.750, "gambut": 1},
    # Jawa Timur
    {"prov": "JATIM", "kawasan": "TN Bromo Tengger Semeru", "lat": -7.940, "lon": 112.950, "gambut": 0},
    {"prov": "JATIM", "kawasan": "TN Baluran (Savana Bekol)", "lat": -7.820, "lon": 114.420, "gambut": 0},
    {"prov": "JATIM", "kawasan": "KPH Perhutani Bojonegoro", "lat": -7.280, "lon": 111.750, "gambut": 0},
    # Kalimantan Tengah
    {"prov": "KALTENG", "kawasan": "TN Sebangau (Kubah Gambut)", "lat": -2.350, "lon": 113.850, "gambut": 1},
    {"prov": "KALTENG", "kawasan": "Eks-PLG Blok A Pulang Pisau", "lat": -2.750, "lon": 114.250, "gambut": 1},
    {"prov": "KALTENG", "kawasan": "TN Tanjung Puting", "lat": -2.950, "lon": 111.950, "gambut": 1},
    # Kalimantan Timur
    {"prov": "KALTIM", "kawasan": "Tahura Bukit Soeharto / IKN", "lat": -0.980, "lon": 117.020, "gambut": 0},
    {"prov": "KALTIM", "kawasan": "TN Kutai (Sangkima)", "lat": 0.250, "lon": 117.350, "gambut": 0},
    {"prov": "KALTIM", "kawasan": "Konsesi Tambang Kutai Kartanegara", "lat": -0.450, "lon": 116.850, "gambut": 0},
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

# ==========================================
# 2. OVERLAY KAWASAN RINGAN (RAM < 5MB)
# ==========================================
def classify_point_fast(prov, lat, lon):
    if prov == "JAMBI":
        if lon >= 104.00 and -1.60 <= lat <= -1.15:
            return "KPA (Taman Nasional)", "TN Berbak (TNB)", 1, "Tanjung Jabung Timur"
        elif 103.00 <= lon <= 103.45 and -1.25 <= lat <= -0.90:
            return "Hutan Lindung (HL)", "HL Gambut Bram Itam", 1, "Tanjung Jabung Barat"
        elif 103.15 <= lon <= 103.55 and -2.40 <= lat <= -2.00:
            return "PBPH (Restorasi)", "PBPH PT REKI (Hutan Harapan)", 0, "Batanghari"
        else:
            is_g = 1 if (lon >= 103.50 and lat >= -1.70) else 0
            return f"APL ({'Gambut' if is_g else 'Mineral'})", "Areal Perkebunan / Lahan Terbuka", is_g, "Muaro Jambi"

    elif prov == "SUMSEL":
        if 104.80 <= lon <= 105.35 and -3.20 <= lat <= -2.60:
            return "KSA (Suaka Margasatwa)", "SM Padang Sugihan", 1, "OKI"
        elif lon >= 105.15 and -3.75 <= lat <= -2.75:
            return "PBPH (Hutan Tanaman)", "PBPH HTI Gambut OKI", 1, "OKI"
        else:
            is_g = 1 if lon >= 104.70 else 0
            return f"APL ({'Gambut' if is_g else 'Mineral'})", "Perkebunan / Lahan Terbuka", is_g, "OKI"

    elif prov == "JATIM":
        if 112.80 <= lon <= 113.15 and -8.20 <= lat <= -7.85:
            return "KPA (Taman Nasional)", "TN Bromo Tengger Semeru", 0, "Malang / Probolinggo"
        elif 114.25 <= lon <= 114.55 and -7.95 <= lat <= -7.70:
            return "KPA (Taman Nasional)", "TN Baluran", 0, "Situbondo"
        else:
            return "APL (Mineral)", "Tegalan / Sawah / Pemukiman", 0, "Jatim Terbuka"

    elif prov == "KALTENG":
        if 113.25 <= lon <= 114.15 and -3.25 <= lat <= -2.00:
            return "KPA (Taman Nasional)", "TN Sebangau (Kubah Gambut)", 1, "Palangka Raya / Pulang Pisau"
        elif 111.70 <= lon <= 112.35 and -3.35 <= lat <= -2.55:
            return "KPA (Taman Nasional)", "TN Tanjung Puting", 1, "Kotawaringin Barat"
        elif 114.00 <= lon <= 114.85 and -3.10 <= lat <= -2.30:
            return "Kawasan Gambut (KHG)", "Kawasan Eks-PLG", 1, "Pulang Pisau / Kapuas"
        else:
            is_g = 1 if lat <= -1.80 else 0
            return f"APL ({'Gambut' if is_g else 'Mineral'})", "Lahan Budidaya / Terbuka Kalteng", is_g, "Kotim / Pulang Pisau"

    elif prov == "KALTIM":
        if 116.70 <= lon <= 117.20 and -1.25 <= lat <= -0.75:
            return "KPA (Tahura) / IKN", "Tahura Bukit Soeharto & Lanskap IKN", 0, "Kukar / PPU"
        elif 117.10 <= lon <= 117.60 and 0.10 <= lat <= 0.65:
            return "KPA (Taman Nasional)", "TN Kutai", 0, "Kutai Timur"
        else:
            return "APL (Mineral)", "Perkebunan / Lahan Terbuka Kaltim", 0, "Kukar / Paser"

    return "Lainnya", "Area Terbuka", 0, "Kabupaten"

# ==========================================
# 3. PENARIKAN NASA FIRMS & OPEN-METEO
# ==========================================
def fetch_satellite_data_multi(target_date_str, target_prov="ALL"):
    map_key = get_nasa_key()
    prov_keys = ["JAMBI", "SUMSEL", "JATIM", "KALTENG", "KALTIM"] if target_prov == "ALL" else [target_prov]
    combined = []

    sensors = [
        ("VIIRS_SNPP_NRT", "SNPP"),
        ("VIIRS_NOAA20_NRT", "NOAA-20"),
        ("MODIS_NRT", "Terra/Aqua"),
    ]

    for p in prov_keys:
        bbox = PROVINSI_BBOX[p]
        for api_code, sat_lbl in sensors:
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{api_code}/{bbox}/2"
            try:
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200 and "latitude" in resp.text:
                    df = pd.read_csv(io.StringIO(resp.text))
                    if df.empty or "latitude" not in df.columns:
                        continue

                    for _, r in df.iterrows():
                        lat = float(r["latitude"])
                        lon = float(r["longitude"])

                        c_raw = str(r.get("confidence", "n")).strip().lower()
                        if c_raw in ["h", "high"]:
                            conf_val = 90.0
                        elif c_raw in ["l", "low"]:
                            conf_val = 25.0
                        else:
                            try:
                                conf_val = float(c_raw)
                            except ValueError:
                                conf_val = 60.0

                        acq_date = str(r.get("acq_date", target_date_str))

                        combined.append({
                            "provinsi": p,
                            "latitude": lat,
                            "longitude": lon,
                            "confidence": conf_val,
                            "satellite_source": sat_lbl,
                            "acq_date": acq_date,
                        })
            except Exception:
                pass

    return combined

def calculate_risk_and_save(conn, item, fc_date, tm, rm, pr, wn, dry_count):
    temp_score = min(max((tm - 28.0) * 8.0, 0), 40)
    rh_score = min(max((75.0 - rm) * 1.2, 0), 30)
    dry_score = 25 if pr < 1.0 else (10 if pr < 5.0 else 0)
    peat_multiplier = 1.25 if item["gambut"] == 1 else 1.05

    raw_score = (temp_score + rh_score + dry_score) * peat_multiplier * (1.0 + (wn / 60.0))
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
        cat = "Tinggi"
    elif score >= 35:
        cat = "Sedang"
    else:
        cat = "Rendah"

    fid = f"FC-{item['prov']}-{fc_date}-{int(item['lat']*100)}-{int(item['lon']*100)}"
    c = conn.cursor()
    c.execute(
        """
        INSERT OR REPLACE INTO fire_risk_forecast
        (forecast_id, provinsi, nama_kawasan, latitude, longitude, forecast_date, temp_max, rh_min, precip_sum, wind_max, is_gambut, risk_score, risk_category, confidence_pct, confidence_label, dry_days_projected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            fid,
            item["prov"],
            item["kawasan"],
            item["lat"],
            item["lon"],
            fc_date,
            tm,
            rm,
            pr,
            wn,
            item["gambut"],
            score,
            cat,
            conf_pct,
            conf_label,
            dry_count,
        ),
    )

def update_fire_risk_forecast_multi():
    conn = sqlite3.connect(DB_NAME)
    today = date.today()

    for item in SENTINEL_POINTS:
        lat, lon = item["lat"], item["lon"]
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,relative_humidity_2m_min,precipitation_sum,wind_speed_10m_max"
            "&timezone=Asia%2FJakarta&forecast_days=5"
        )
        fetched_successfully = False
        try:
            r = requests.get(url, timeout=5)
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
                    rm = rh_min[i] if i < len(rh_min) else 50.0
                    pr = precip[i] if i < len(precip) else 0.0
                    wn = wind[i] if i < len(wind) else 12.0
                    dry_count = dry_count + 1 if pr < 1.0 else 0
                    calculate_risk_and_save(conn, item, fc_date, tm, rm, pr, wn, dry_count)

                fetched_successfully = True
        except Exception:
            fetched_successfully = False

        if not fetched_successfully:
            for day_offset in range(5):
                fc_date = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                tm = 35.5 if item["gambut"] == 1 else 33.5
                rm = 42.0 if item["gambut"] == 1 else 55.0
                pr = 0.0 if day_offset < 3 else 2.5
                wn = 14.0
                dry_count = day_offset + 1 if pr < 1.0 else 0
                calculate_risk_and_save(conn, item, fc_date, tm, rm, pr, wn, dry_count)

    conn.commit()
    conn.close()

def sync_hotspots_multi(target_date_str):
    init_db()
    raw_points = fetch_satellite_data_multi(target_date_str, target_prov="ALL")

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    if raw_points:
        for pt in raw_points:
            prov = pt["provinsi"]
            lat = pt["latitude"]
            lon = pt["longitude"]
            conf = pt["confidence"]
            sat = pt["satellite_source"]
            dt = pt["acq_date"]

            status = "Tinggi" if conf >= 80 else ("Sedang" if conf >= 30 else "Rendah")
            cid = f"HS-{prov}-{dt}-{int(abs(lat)*1000)}-{int(abs(lon)*1000)}"
            fkws, nkws, is_g, kab = classify_point_fast(prov, lat, lon)

            c.execute("""
                INSERT OR REPLACE INTO hotspot_clusters 
                (cluster_id, provinsi, latitude, longitude, kabupaten, fungsi_kawasan, nama_kawasan, is_gambut, first_detected, last_detected, consecutive_padam_days, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
            """, (cid, prov, lat, lon, kab, fkws, nkws, is_g, dt, dt))

            c.execute("""
                INSERT OR REPLACE INTO hotspot_daily_log 
                (cluster_id, log_date, satellite_source, status, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (cid, dt, sat, status, conf))

    conn.commit()
    conn.close()

# ==========================================
# 4. INITIAL SEED (BASELINE 10 HARI CEPAT)
# ==========================================
def seed_historical_multi():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM hotspot_clusters")
    total_clusters = c.fetchone()[0]

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    if total_clusters == 0:
        sample_clusters = [
            ("HS-KALTENG-S01", "KALTENG", -2.350, 113.850, "Pulang Pisau", "KPA (Taman Nasional)", "TN Sebangau", 1),
            ("HS-KALTENG-S02", "KALTENG", -2.750, 114.250, "Pulang Pisau", "Kawasan Gambut (KHG)", "Eks-PLG Blok A", 1),
            ("HS-KALTENG-S03", "KALTENG", -2.410, 113.910, "Palangka Raya", "APL (Gambut)", "Area Gambut Terbuka", 1),
            ("HS-KALTENG-S04", "KALTENG", -2.120, 112.950, "Kotim", "PBPH / Perkebunan", "Konsesi Sawit Kalteng", 1),
            ("HS-SUMSEL-S01", "SUMSEL", -3.350, 105.420, "OKI", "PBPH (Hutan Tanaman)", "PBPH HTI Gambut OKI", 1),
            ("HS-SUMSEL-S02", "SUMSEL", -2.950, 105.050, "OKI", "KSA (Suaka Margasatwa)", "SM Padang Sugihan", 1),
            ("HS-JAMBI-S01", "JAMBI", -1.352, 104.102, "Tanjabtim", "KPA (Taman Nasional)", "TN Berbak", 1),
            ("HS-JAMBI-S02", "JAMBI", -1.420, 103.550, "Muaro Jambi", "PBPH (Hutan Tanaman)", "PBPH PT WKS", 1),
            ("HS-JATIM-S01", "JATIM", -7.940, 112.950, "Probolinggo", "KPA (Taman Nasional)", "TN Bromo Tengger Semeru", 0),
            ("HS-KALTIM-S01", "KALTIM", -0.980, 117.020, "PPU / IKN", "KPA (Tahura) / IKN", "Tahura Bukit Soeharto", 0),
        ]

        for cid, prov, lat, lon, kab, fkws, nkws, is_g in sample_clusters:
            c.execute("""
                INSERT OR REPLACE INTO hotspot_clusters 
                (cluster_id, provinsi, latitude, longitude, kabupaten, fungsi_kawasan, nama_kawasan, is_gambut, first_detected, last_detected, consecutive_padam_days, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
            """, (cid, prov, lat, lon, kab, fkws, nkws, is_g, (today - timedelta(days=6)).strftime("%Y-%m-%d"), today_str))

            for day_offset in range(10):
                d_str = (today - timedelta(days=(9 - day_offset))).strftime("%Y-%m-%d")
                status = "Tinggi" if day_offset >= 5 else "Sedang"
                conf = 92.0 if status == "Tinggi" else 65.0
                c.execute("""
                    INSERT OR REPLACE INTO hotspot_daily_log 
                    (cluster_id, log_date, satellite_source, status, confidence)
                    VALUES (?, ?, 'NOAA-20', ?, ?)
                """, (cid, d_str, status, conf))

        conn.commit()

    conn.close()

# ==========================================
# 5. PIVOT & PERHITUNGAN RINGKASAN
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
            f"SELECT nama_kawasan, risk_score, risk_category, is_gambut, confidence_pct FROM fire_risk_forecast WHERE provinsi = '{p}'",
            conn,
        )
        df_hs = pd.read_sql_query(
            f"SELECT COUNT(*) as hs_cnt FROM hotspot_clusters WHERE provinsi = '{p}' AND is_active = 1",
            conn,
        )

        if not df_p.empty:
            max_conf = round(float(df_p["confidence_pct"].max()), 1)
            ekstrim_sites = df_p[df_p["risk_category"] == "Ekstrim"]["nama_kawasan"].nunique()
            tinggi_sites = df_p[df_p["risk_category"] == "Tinggi"]["nama_kawasan"].nunique()
            gambut_sites = df_p[(df_p["risk_score"] >= 60) & (df_p["is_gambut"] == 1)]["nama_kawasan"].nunique()
        else:
            max_conf = 0.0
            ekstrim_sites = 0
            tinggi_sites = 0
            gambut_sites = 0

        hs_count = int(df_hs.iloc[0]["hs_cnt"]) if not df_hs.empty else 0

        if ekstrim_sites >= 2 or max_conf >= 90.0:
            status_label = "Siaga 1 (Kritis Ekstrim)"
            badge_class = "badge-ekstrim"
        elif ekstrim_sites >= 1 or tinggi_sites >= 2:
            status_label = "Siaga 2 (Tinggi)"
            badge_class = "badge-tinggi"
        else:
            status_label = "Waspada"
            badge_class = "badge-sedang"

        summary.append({
            "provinsi": p,
            "status_label": status_label,
            "badge_class": badge_class,
            "ekstrim_cnt": ekstrim_sites,
            "tinggi_cnt": tinggi_sites,
            "gambut_cnt": gambut_sites,
            "max_confidence": max_conf,
            "hs_count": hs_count,
        })

    conn.close()
    return summary

def get_forecast_summary_multi(filter_prov="ALL"):
    conn = sqlite3.connect(DB_NAME)
    prov_where = "" if filter_prov == "ALL" else f"WHERE provinsi = '{filter_prov}'"
    query = f"""
        SELECT provinsi, nama_kawasan, latitude, longitude, forecast_date, temp_max, rh_min, precip_sum, wind_max, is_gambut, risk_score, risk_category, confidence_pct, confidence_label, dry_days_projected
        FROM fire_risk_forecast
        {prov_where}
        ORDER BY forecast_date ASC, risk_score DESC
    """
    df_fc = pd.read_sql_query(query, conn)
    conn.close()

    records = df_fc.to_dict(orient="records") if not df_fc.empty else []
    for idx, item in enumerate(records, start=1):
        item["no"] = idx
    return records

def generate_table_multi(
    ref_date,
    filter_prov="ALL",
    filter_sat="ALL",
    filter_status="ALL",
    filter_source="ALL_DATA",
):
    date_list = [(ref_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(9, -1, -1)]
    min_date = date_list[0]

    conn = sqlite3.connect(DB_NAME)
    prov_where = "" if filter_prov == "ALL" else f"AND provinsi = '{filter_prov}'"
    df_clusters = pd.read_sql_query(
        f"SELECT cluster_id, provinsi, latitude, longitude, kabupaten, fungsi_kawasan, nama_kawasan, is_gambut FROM hotspot_clusters WHERE is_active = 1 {prov_where}",
        conn,
    )

    summary_stats = {
        "total": 0, "tinggi": 0, "sedang": 0, "rendah": 0, "padam": 0, "aktif": 0,
        "konservasi": 0, "gambut": 0, "ekstrim_fc": 0, "max_conf": 0,
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
        df_clusters = df_clusters[df_clusters["fungsi_kawasan"].str.contains("KPA|KSA|Tahura", na=False)]

    query_logs = f"""
        SELECT cluster_id, log_date, status, confidence, satellite_source
        FROM hotspot_daily_log 
        WHERE log_date >= '{min_date}' AND log_date <= '{date_list[-1]}' {sat_cond}
    """
    df_logs = pd.read_sql_query(query_logs, conn)

    fc_prov_where = "" if filter_prov == "ALL" else f"AND provinsi = '{filter_prov}'"
    df_fc_stats = pd.read_sql_query(
        f"SELECT COUNT(DISTINCT nama_kawasan) as cnt, MAX(confidence_pct) as max_c FROM fire_risk_forecast WHERE risk_score >= 80 {fc_prov_where}",
        conn,
    )
    if not df_fc_stats.empty:
        summary_stats["ekstrim_fc"] = int(df_fc_stats.iloc[0]["cnt"] or 0)
        summary_stats["max_conf"] = round(float(df_fc_stats.iloc[0]["max_c"] or 0.0), 1)

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
        pivot = agg_logs.pivot(index="cluster_id", columns="log_date", values="status").fillna("Padam")

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
    merged["sensor_terdeteksi"] = merged["cluster_id"].map(lambda cid: sat_summary.get(cid, "-"))

    if filter_sat != "ALL":
        merged = merged[merged["sensor_terdeteksi"] != "-"]

    def get_latest_active_status(row):
        for dt in reversed(date_list):
            val = row.get(dt, "Padam")
            if val != "Padam":
                return val
        return "Padam"

    merged["status_terakhir"] = merged.apply(get_latest_active_status, axis=1)

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
        summary_stats["aktif"] = summary_stats["tinggi"] + summary_stats["sedang"] + summary_stats["rendah"]
        summary_stats["total"] = len(merged)

        aktif_rows = merged[merged["status_terakhir"] != "Padam"]
        summary_stats["konservasi"] = int(aktif_rows["fungsi_kawasan"].str.contains("KPA|KSA|Tahura", na=False).sum())
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
        export_df = df.drop(columns=["is_gambut", "status_terakhir"], errors="ignore")
        export_df = export_df.rename(columns={"is_gambut_label": "is_gambut"})
        export_df.to_excel(file_name, index=False)
    return file_name

# ==========================================
# 6. PENJADWAL OTOMATIS
# ==========================================
def scheduled_sync_multi():
    today_str = date.today().strftime("%Y-%m-%d")
    sync_hotspots_multi(today_str)
    update_fire_risk_forecast_multi()

def run_scheduler():
    while True:
        now_str = datetime.now().strftime("%H:%M")
        if now_str in ["07:00", "18:00"]:
            scheduled_sync_multi()
            time.sleep(60)
        time.sleep(25)

# ==========================================
# 7. FLASK APP & HTML TEMPLATE
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Sistem Pemantauan Hotspot & Peringatan Dini Karhutla Multi-Provinsi</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 24px; background: #f8fafc; color: #1e293b; }
        .header { display: flex; justify-content: space-between; align-items: center; background: #fff; padding: 20px 28px; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }
        h1 { margin: 0; font-size: 22px; color: #0f172a; }
        .subtitle { font-size: 13px; color: #64748b; margin-top: 4px; }
        .schedule-badge { background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 6px; font-weight: 700; }
        
        .controls { display: flex; gap: 12px; align-items: center; background: white; padding: 14px 24px; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; flex-wrap: wrap; }
        select { padding: 9px 12px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 13px; font-weight: 600; background: white; color: #0f172a; }
        .select-prov { background: #f0fdf4; border-color: #86efac; color: #166534; font-size: 14px; }
        .select-mode { background: #eff6ff; border-color: #93c5fd; color: #1e40af; font-size: 14px; font-weight: 700; }
        
        button, a.btn { padding: 9px 16px; border-radius: 8px; font-weight: 600; text-decoration: none; cursor: pointer; border: none; font-size: 13px; display: inline-flex; align-items: center; gap: 6px; }
        .btn-refresh { background: #0284c7; color: white; }
        .btn-excel { background: #16a34a; color: white; }
        
        .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow-x: auto; margin-bottom: 24px; }
        .card-header-title { font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }
        
        #map { height: 500px; width: 100%; border-radius: 10px; border: 1px solid #e2e8f0; }
        
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
            <h1>Sistem Pemantauan Hotspot & Peringatan Dini Karhutla</h1>
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

        <label><strong>Fokus Tampilan:</strong></label>
        <select id="viewModeSelect" class="select-mode" onchange="applyFilters()">
            <option value="ALL_LAYERS" {% if current_view == 'ALL_LAYERS' %}selected{% endif %}>🔥 Tampilkan Keduanya (Hotspot & Potensi H-5)</option>
            <option value="HOTSPOT_ONLY" {% if current_view == 'HOTSPOT_ONLY' %}selected{% endif %}>🛰️ Hanya Hotspot Eksisting (10 Hari)</option>
            <option value="FORECAST_ONLY" {% if current_view == 'FORECAST_ONLY' %}selected{% endif %}>⚠️ Hanya Potensi Karhutla (Prediksi H-5)</option>
        </select>

        <label><strong>Sumber Feed:</strong></label>
        <select id="sourceSelect" onchange="applyFilters()">
            <option value="ALL_DATA" {% if current_source == 'ALL_DATA' %}selected{% endif %}>🌐 Seluruh Lanskap (HK, HL, PBPH, APL)</option>
            <option value="GITHUB_HK" {% if current_source == 'GITHUB_HK' %}selected{% endif %}>🌲 Khusus Kawasan Konservasi (TN/CA/SM/Tahura)</option>
        </select>

        <button class="btn btn-refresh" onclick="refreshData()">🔄 Sinkronisasi Hotspot & Cuaca</button>
    </div>

    <!-- KARTU STATISTIK UTAMA -->
    <div class="summary-container">
        <div class="summary-grid">
            <div class="stat-card">
                <span class="stat-lbl">Total Hotspot Aktif</span>
                <span class="stat-val" style="color: #0284c7;">{{ stats['aktif'] }}</span>
            </div>
            <div class="stat-card" style="background:#fef2f2; border-color:#fecaca;">
                <span class="stat-lbl">Hotspot Risiko Tinggi</span>
                <span class="stat-val" style="color:#dc2626;">{{ stats['tinggi'] }}</span>
            </div>
            <div class="stat-card" style="background:#fffbeb; border-color:#fde68a;">
                <span class="stat-lbl">Hotspot Sedang</span>
                <span class="stat-val" style="color:#d97706;">{{ stats['sedang'] }}</span>
            </div>
            <div class="stat-card">
                <span class="stat-lbl">Di Hutan Konservasi</span>
                <span class="stat-val" style="color:#e11d48;">{{ stats['konservasi'] }}</span>
            </div>
            <div class="stat-card">
                <span class="stat-lbl">Di Lahan Gambut</span>
                <span class="stat-val" style="color:#ea580c;">{{ stats['gambut'] }}</span>
            </div>
            <div class="stat-card stat-ekstrim">
                <span class="stat-lbl">Puncak Keyakinan H-5</span>
                <span class="stat-val">{{ stats['max_conf'] }}%</span>
            </div>
        </div>
    </div>

    <!-- PETA SPASIAL LEAFLET.JS -->
    <div class="card">
        <div class="card-header-title">
            <span>🗺️ Visualisasi Spasial Sebaran Hotspot Eksisting & Titik Potensi Karhutla H-5</span>
            <span style="font-size:12px; color:#64748b; font-weight:normal;">Kontrol layer di pojok kanan atas peta untuk memilih tampilan</span>
        </div>
        <div id="map"></div>
    </div>

    <!-- REKAP POTENSI ANTAR PROVINSI -->
    {% if current_view != 'HOTSPOT_ONLY' %}
    <div class="card" style="border-top: 4px solid #0284c7;">
        <div class="card-header-title">
            📊 Rekapitulasi Matriks Status Potensi Karhutla Antar-Provinsi (Agregasi Data Prediksi H-5)
        </div>
        <table>
            <thead>
                <tr>
                    <th>Provinsi</th>
                    <th>Status Siaga Utama</th>
                    <th>Keyakinan Hotspot Maksimum (Detail H-5)</th>
                    <th>Titik Sentinel Ekstrim (H-5)</th>
                    <th>Titik Sentinel Siaga Tinggi</th>
                    <th>Titik Rawan Gambut</th>
                    <th>Hotspot Eksisting</th>
                    <th>Rekomendasi Operasional Lapangan</th>
                </tr>
            </thead>
            <tbody>
                {% for p in prov_summary %}
                <tr>
                    <td><span class="prov-tag prov-{{ p['provinsi']|lower }}">{{ p['provinsi'] }}</span></td>
                    <td><span class="badge {{ p['badge_class'] }}">{{ p['status_label'] }}</span></td>
                    <td><b style="color:#dc2626; font-size:14px;">{{ p['max_confidence'] }}%</b></td>
                    <td><b>{{ p['ekstrim_cnt'] }} Titik Sentinel</b></td>
                    <td>{{ p['tinggi_cnt'] }} Titik Sentinel</td>
                    <td>{% if p['gambut_cnt'] > 0 %}<span style="color:#ea580c; font-weight:700;">{{ p['gambut_cnt'] }} Titik Sentinel</span>{% else %}-{% endif %}</td>
                    <td><b>{{ p['hs_count'] }} Titik</b></td>
                    <td style="font-size:12px; color:#475569;">
                        {% if p['provinsi'] == 'KALTENG' %}
                            Operasi TMC pembasahan kubah gambut Sebangau & Eks-PLG, siaga helikopter water bombing.
                        {% elif p['provinsi'] == 'SUMSEL' %}
                            Patroli darat koridor HTI OKI, penutupan sekat kanal Padang Sugihan.
                        {% elif p['provinsi'] == 'JAMBI' %}
                            Aktivasi pintu air sekat kanal TN Berbak, pengawasan trans-boundary Hutan Harapan.
                        {% elif p['provinsi'] == 'KALTIM' %}
                            Pembersihan sekat bakar batas inti IKN, monitoring areal mineral TN Kutai.
                        {% else %}
                            Pembersihan serasah savana TN Baluran & pencegahan kebakaran daun jati KPH Jatim.
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <!-- TABEL PREDIKSI H-5 & KOORDINAT ACUAN -->
    <div class="card" style="border-left: 5px solid #e11d48;">
        <div class="card-header-title">
            ⚠️ Detail Peringatan Dini Cuaca, 3 Elemen Pembakar & Keyakinan Hotspot 5 Hari ke Depan
        </div>
        <table>
            <thead>
                <tr>
                    <th>No</th>
                    <th>Provinsi</th>
                    <th>Titik Sentinel / Kawasan Kritis</th>
                    <th>Koordinat Acuan</th>
                    <th>Tanggal Proyeksi</th>
                    <th>Keyakinan Timbul Hotspot (%)</th>
                    <th>Hari Tanpa Hujan (HTH)</th>
                    <th>1. Suhu Puncak</th>
                    <th>2. Ketersediaan Bahan Bakar</th>
                    <th>3. Desikasi (RH / Angin)</th>
                    <th>Tipe Lahan</th>
                    <th>Tingkat Ancaman</th>
                </tr>
            </thead>
            <tbody>
                {% for f in forecasts %}
                <tr>
                    <td><b>{{ f['no'] }}</b></td>
                    <td><span class="prov-tag prov-{{ f['provinsi']|lower }}">{{ f['provinsi'] }}</span></td>
                    <td><b>{{ f['nama_kawasan'] }}</b></td>
                    <td><code style="background:#f1f5f9; padding:2px 6px; border-radius:4px; font-weight:bold;">{{ "%.4f"|format(f['latitude']) }}, {{ "%.4f"|format(f['longitude']) }}</code></td>
                    <td>{{ f['forecast_date'] }}</td>
                    <td>
                        <span style="font-size:14px; font-weight:800; color:{% if f['confidence_pct'] >= 80 %}#dc2626{% elif f['confidence_pct'] >= 60 %}#d97706{% else %}#16a34a{% endif %};">
                            {{ f['confidence_pct'] }}%
                        </span>
                        <div style="font-size:11px; color:#64748b;">{{ f['confidence_label'] }}</div>
                    </td>
                    <td><b>{{ f['dry_days_projected'] }} Hari Kering</b></td>
                    <td><b>{{ f['temp_max'] }}°C</b></td>
                    <td>
                        {{ f['precip_sum'] }} mm 
                        {% if f['precip_sum'] < 1.0 %}
                            <span style="color:#dc2626; font-size:11px; font-weight:bold;">(Kering Ekstrim)</span>
                        {% else %}
                            <span style="color:#16a34a; font-size:11px;">(Lembap)</span>
                        {% endif %}
                    </td>
                    <td>RH {{ f['rh_min'] }}% | {{ f['wind_max'] }} km/j</td>
                    <td>{% if f['is_gambut'] == 1 %}<span style="color:#ea580c; font-weight:700;">Gambut</span>{% else %}Mineral{% endif %}</td>
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
    {% endif %}

    <!-- TABEL RIWAYAT HOTSPOT 10 HARI -->
    {% if current_view != 'FORECAST_ONLY' %}
    <div class="card">
        <div class="card-header-title">
            🛰️ Matriks Riwayat Titik Hotspot Eksisting 10 Hari Terakhir (Multi-Satelit)
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
    {% endif %}

    <script>
        function applyFilters() {
            const prov = document.getElementById('provSelect').value;
            const view = document.getElementById('viewModeSelect').value;
            const source = document.getElementById('sourceSelect').value;
            window.location.href = '/?prov=' + prov + '&view=' + view + '&source=' + source;
        }

        function refreshData() {
            const btn = document.querySelector('.btn-refresh');
            btn.innerHTML = '⏳ Menjalankan Sinkronisasi Latar Belakang...';
            btn.disabled = true;
            fetch('/api/sync')
                .then(r => r.json())
                .then(res => {
                    btn.innerHTML = '⏳ Menarik Data Satelit & Cuaca (8 dtk)...';
                    setTimeout(() => {
                        alert('Data hotspot dan prediksi cuaca 5 provinsi berhasil disinkronkan!');
                        window.location.reload();
                    }, 8000);
                })
                .catch(e => {
                    alert('Gagal menyinkronkan data.');
                    btn.innerHTML = '🔄 Sinkronisasi Hotspot & Cuaca';
                    btn.disabled = false;
                });
        }

        const mapCenter = {{ map_center|tojson }};
        const map = L.map('map').setView([mapCenter[0], mapCenter[1]], mapCenter[2]);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 18,
            attribution: '© OpenStreetMap contributors | Pemantauan Karhutla'
        }).addTo(map);

        const hotspotLayer = L.layerGroup();
        const forecastLayer = L.layerGroup();

        const hotspotData = {{ rows|tojson }};
        const forecastData = {{ forecasts|tojson }};
        const currentView = "{{ current_view }}";

        hotspotData.forEach(pt => {
            const st = pt.status_terakhir || 'Padam';
            const isPadam = st === 'Padam';
            const color = isPadam ? '#94a3b8' : (st === 'Tinggi' ? '#dc2626' : (st === 'Sedang' ? '#d97706' : '#16a34a'));
            
            const circle = L.circleMarker([pt.latitude, pt.longitude], {
                radius: isPadam ? 5 : 8,
                fillColor: color,
                color: '#ffffff',
                weight: 1.5,
                opacity: 1,
                fillOpacity: 0.9
            });

            circle.bindPopup(`
                <div style="font-size:12px; min-width:210px;">
                    <strong style="color:${color}; font-size:14px;">Hotspot Eksisting: ${st}</strong><br>
                    <b>Kawasan:</b> ${pt.nama_kawasan}<br>
                    <b>Koordinat:</b> ${Number(pt.latitude).toFixed(4)}, ${Number(pt.longitude).toFixed(4)}<br>
                    <b>Kabupaten:</b> ${pt.kabupaten} (${pt.provinsi})<br>
                    <b>Fungsi:</b> ${pt.fungsi_kawasan}<br>
                    <b>Tipe Lahan:</b> ${pt.is_gambut_label === 'Ya' ? 'Rawa Gambut' : 'Tanah Mineral'}<br>
                    <b>Sensor Satelit:</b> ${pt.sensor_terdeteksi}
                </div>
            `);
            hotspotLayer.addLayer(circle);
        });

        const sentinelRendered = new Set();
        forecastData.forEach(fc => {
            const key = fc.latitude + '_' + fc.longitude;
            if (!sentinelRendered.has(key)) {
                sentinelRendered.add(key);
                const isEkstrim = fc.risk_category === 'Ekstrim';
                const pulseColor = isEkstrim ? '#dc2626' : '#ea580c';
                
                const marker = L.circleMarker([fc.latitude, fc.longitude], {
                    radius: 13,
                    fillColor: pulseColor,
                    color: '#ffffff',
                    weight: 2,
                    dashArray: '4, 4',
                    fillOpacity: 0.5
                });

                marker.bindPopup(`
                    <div style="font-size:12px; min-width:245px;">
                        <span style="background:${pulseColor}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;">
                            🔥 Potensi H-5: ${fc.risk_category}
                        </span><br>
                        <div style="margin: 6px 0 2px 0;">
                            <b>Keyakinan Hotspot:</b> <span style="font-size:14px; font-weight:bold; color:${pulseColor};">${fc.confidence_pct}%</span>
                            <div style="font-size:11px; color:#64748b;">${fc.confidence_label}</div>
                        </div>
                        <b style="font-size:13px; display:inline-block; margin-top:4px;">${fc.nama_kawasan}</b><br>
                        <span style="color:#0f172a; font-weight:bold; background:#f1f5f9; padding:1px 4px; border-radius:3px;">
                            📍 ${Number(fc.latitude).toFixed(4)}, ${Number(fc.longitude).toFixed(4)}
                        </span><br>
                        <span style="color:#64748b;">${fc.provinsi} | ${fc.is_gambut === 1 ? 'Rawa Gambut' : 'Tanah Mineral'}</span>
                        <hr style="border:0; border-top:1px solid #e2e8f0; margin:6px 0;">
                        <b>Analisis 3 Elemen Pembakar:</b><br>
                        🔥 <b>1. Suhu Puncak:</b> ${fc.temp_max}°C (${fc.temp_max >= 34 ? '<span style="color:red; font-weight:bold;">Kritis</span>' : 'Normal'})<br>
                        🍂 <b>2. Bahan Bakar / HTH:</b> ${fc.dry_days_projected} Hari Kering (${fc.precip_sum} mm)<br>
                        💨 <b>3. Desikasi (RH & Angin):</b> RH ${fc.rh_min}% | Angin ${fc.wind_max} km/j
                    </div>
                `);
                forecastLayer.addLayer(marker);
            }
        });

        if (currentView === 'ALL_LAYERS') {
            hotspotLayer.addTo(map);
            forecastLayer.addTo(map);
        } else if (currentView === 'HOTSPOT_ONLY') {
            hotspotLayer.addTo(map);
        } else if (currentView === 'FORE
