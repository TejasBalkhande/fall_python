import pandas as pd
import streamlit as st
import requests
import pickle
import joblib
import time
import math
from datetime import datetime
import os


# ——— Load your trained model and scaler ———
with open('knn_model_all.pkl', 'rb') as f:
    model = pickle.load(f)
scaler = joblib.load('data_scaler_all.pkl')

# ——— Define feature names and column order ———
feature_names = ["ax", "ay", "az", "droll", "dpitch", "dyaw", "w", "x", "y", "z"]
columns_in_order = ['w', 'x', 'y', 'z', 'droll', 'dpitch', 'dyaw', 'ax', 'ay', 'az']

# ——— ESP32 HTTP endpoint ———
ESP32_IP = "192.168.213.84"  # Replace with your ESP32’s IP
URL = f"http://{ESP32_IP}/data"

# ——— Output CSV file ———
csv_file = "fall_records.csv"
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["timestamp"] + columns_in_order).to_csv(csv_file, index=False)

# ——— Madgwick Filter Implementation ———
class MadgwickFilter:
    def __init__(self, sample_freq=50.0, beta=0.1):
        self.q0, self.q1, self.q2, self.q3 = 1.0, 0.0, 0.0, 0.0
        self.beta = beta
        self.sample_freq = sample_freq

    def update(self, gx, gy, gz, ax, ay, az):
        gx, gy, gz = map(math.radians, (gx, gy, gz))
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm == 0:
            return
        ax, ay, az = ax/norm, ay/norm, az/norm

        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3
        _2q0, _2q1, _2q2, _2q3 = 2*q0, 2*q1, 2*q2, 2*q3
        _4q0, _4q1, _4q2 = 4*q0, 4*q1, 4*q2
        _8q1, _8q2 = 8*q1, 8*q2
        q0q0, q1q1, q2q2, q3q3 = q0*q0, q1*q1, q2*q2, q3*q3

        s0 = _4q0*q2q2 + _2q2*ax + _4q0*q1q1 - _2q1*ay
        s1 = _4q1*q3q3 - _2q3*ax + 4*q0q0*q1 - _2q0*ay - _4q1 + _8q1*q1q1 + _8q1*q2q2 + _4q1*az
        s2 = 4*q0q0*q2 + _2q0*ax + _4q2*q3q3 - _2q3*ay - _4q2 + _8q2*q1q1 + _8q2*q2q2 + _4q2*az
        s3 = 4*q1q1*q3 - _2q1*ax + 4*q2q2*q3 - _2q2*ay

        norm_s = math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        if norm_s == 0:
            return
        s0, s1, s2, s3 = s0/norm_s, s1/norm_s, s2/norm_s, s3/norm_s

        qDot0 = 0.5 * (-q1*gx - q2*gy - q3*gz) - self.beta * s0
        qDot1 = 0.5 * ( q0*gx + q2*gz - q3*gy) - self.beta * s1
        qDot2 = 0.5 * ( q0*gy - q1*gz + q3*gx) - self.beta * s2
        qDot3 = 0.5 * ( q0*gz + q1*gy - q2*gx) - self.beta * s3

        q0 += qDot0 / self.sample_freq
        q1 += qDot1 / self.sample_freq
        q2 += qDot2 / self.sample_freq
        q3 += qDot3 / self.sample_freq
        norm_q = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        if norm_q == 0:
            return
        self.q0, self.q1, self.q2, self.q3 = q0/norm_q, q1/norm_q, q2/norm_q, q3/norm_q

# ——— Initialize Madgwick filter and fall history ———
madgwick = MadgwickFilter(sample_freq=50.0, beta=0.1)
fall_times = []

# ——— Streamlit setup ———
st.set_page_config(page_title="ESP32 Fall Detection", layout="wide")
st.title("🔍 Real-Time Fall Detection")
placeholder = st.empty()
fall_log_container = st.container()

# ——— Main loop ———
while True:
    try:
        resp = requests.get(URL, timeout=1.0)
        data = resp.json()

        if "a" in data and "g" in data:
            ax, ay, az = data["a"]
            gx, gy, gz = data["g"]

            # orientation
            madgwick.update(gx, gy, gz, ax, ay, az)
            w, x, y, z = madgwick.q0, madgwick.q1, madgwick.q2, madgwick.q3

            # deltas = gyro rates
            droll, dpitch, dyaw = gx, gy, gz

            # build feature dictionary
            value_map = {
                "ax": ax, "ay": ay, "az": az,
                "droll": droll, "dpitch": dpitch, "dyaw": dyaw,
                "w": w, "x": x, "y": y, "z": z
            }

            # Create DataFrame with all features
            test_df = pd.DataFrame([value_map])
            test_df.columns = feature_names

            # Reorder columns
            X_test = test_df[columns_in_order]

            # Scale and predict
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
            pred = predictions[0]
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            if pred.startswith('fall'):
                label = f"🚨 Fall Detected! {pred}"
                fall_times.append(ts)

                # Save to CSV
                fall_data = [ts] + [value_map[col] for col in columns_in_order]
                pd.DataFrame([fall_data], columns=["timestamp"] + columns_in_order).to_csv(csv_file, mode='a', header=False, index=False)

                # Show recent fall timestamps
                with fall_log_container:
                    st.subheader("📅 Fall Timestamps")
                    st.markdown("\n".join([f"- {t}" for t in reversed(fall_times[-10:])]))  # Last 10 falls
                # break
            else:
                label = f"🧍 Normal Activity - {pred}"

            # UI display
            placeholder.markdown(f"""
                **Time:** {ts}  
                **Accel:** X={ax:.3f} g  Y={ay:.3f} g  Z={az:.3f} g  
                **Gyro:** Roll={gx:.2f}°/s  Pitch={gy:.2f}°/s  Yaw={gz:.2f}°/s  
                **Quat:** W={w:.4f}  X={x:.4f}  Y={y:.4f}  Z={z:.4f}  
                ---  
                ## Prediction: **{label}**
            """)
        else:
            placeholder.warning("⚠️ Empty or malformed data received.")
    except Exception as e:
        placeholder.error(f"Error fetching or processing data: {e}")
    time.sleep(0.5)

def trigger():
    print('Exited with code 1')
    print(fall_times)