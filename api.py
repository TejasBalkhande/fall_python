from flask import Flask, request, jsonify
import pandas as pd
import pickle
import joblib
import time
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load trained model and scaler
logger.info("Loading model and scaler...")
try:
    with open('knn_model_all.pkl', 'rb') as f:
        model = pickle.load(f)
    scaler = joblib.load('data_scaler_all.pkl')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise

# Define feature names and column order
feature_names = ["ax", "ay", "az", "droll", "dpitch", "dyaw", "w", "x", "y", "z"]
columns_in_order = ['w', 'x', 'y', 'z', 'droll', 'dpitch', 'dyaw', 'ax', 'ay', 'az']

# CSV file for storing fall events
csv_file = "fall_records.csv"
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["timestamp"] + columns_in_order + ["predicted_activity"]).to_csv(csv_file, index=False)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict_activity():
    """Predict human activity from IMU sensor data"""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate input data
        missing_features = [feature for feature in feature_names if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        # Create DataFrame with all features
        value_map = {feature: data[feature] for feature in feature_names}
        test_df = pd.DataFrame([value_map])
        
        # Reorder columns according to the model's expected order
        X_test = test_df[columns_in_order]
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make prediction
        prediction = model.predict(X_test_scaled)[0]
        
        # Log timestamp for the prediction
        timestamp = datetime.now().isoformat()
        
        # Log fall events
        if prediction.startswith('fall'):
            # Save to CSV with all features and prediction
            fall_data = [timestamp] + [value_map[col] for col in columns_in_order] + [prediction]
            columns = ["timestamp"] + columns_in_order + ["predicted_activity"]
            pd.DataFrame([fall_data], columns=columns).to_csv(csv_file, mode='a', header=False, index=False)
            logger.warning(f"Fall detected: {prediction} at {timestamp}")
        
        # Return result
        return jsonify({
            "activity": prediction,
            "timestamp": timestamp,
            "confidence": 1.0  # Placeholder, KNN doesn't provide confidence by default
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/data', methods=['GET'])
def get_mock_data():
    """Endpoint to provide mock sensor data for testing"""
    # This endpoint mimics the ESP32 data endpoint for testing
    return jsonify({
        "a": [0.01, -0.95, 9.78],  # [ax, ay, az]
        "g": [-1.5, 0.3, 0.4]       # [gx, gy, gz]
    })

# Madgwick Filter implementation for reference
class MadgwickFilter:
    def __init__(self, sample_freq=50.0, beta=0.1):
        self.q0, self.q1, self.q2, self.q3 = 1.0, 0.0, 0.0, 0.0
        self.beta = beta
        self.sample_freq = sample_freq

    def update(self, gx, gy, gz, ax, ay, az):
        import math
        # Convert gyro from degrees to radians
        gx, gy, gz = map(math.radians, (gx, gy, gz))
        
        # Normalize accelerometer measurement
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm == 0:
            return
        ax, ay, az = ax/norm, ay/norm, az/norm

        # Current quaternion values
        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3
        
        # Auxiliary variables
        _2q0, _2q1, _2q2, _2q3 = 2*q0, 2*q1, 2*q2, 2*q3
        _4q0, _4q1, _4q2 = 4*q0, 4*q1, 4*q2
        _8q1, _8q2 = 8*q1, 8*q2
        q0q0, q1q1, q2q2, q3q3 = q0*q0, q1*q1, q2*q2, q3*q3

        # Gradient descent algorithm corrective step
        s0 = _4q0*q2q2 + _2q2*ax + _4q0*q1q1 - _2q1*ay
        s1 = _4q1*q3q3 - _2q3*ax + 4*q0q0*q1 - _2q0*ay - _4q1 + _8q1*q1q1 + _8q1*q2q2 + _4q1*az
        s2 = 4*q0q0*q2 + _2q0*ax + _4q2*q3q3 - _2q3*ay - _4q2 + _8q2*q1q1 + _8q2*q2q2 + _4q2*az
        s3 = 4*q1q1*q3 - _2q1*ax + 4*q2q2*q3 - _2q2*ay

        # Normalize step magnitude
        norm_s = math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        if norm_s == 0:
            return
        s0, s1, s2, s3 = s0/norm_s, s1/norm_s, s2/norm_s, s3/norm_s

        # Rate of change of quaternion from gyroscope
        qDot0 = 0.5 * (-q1*gx - q2*gy - q3*gz) - self.beta * s0
        qDot1 = 0.5 * (q0*gx + q2*gz - q3*gy) - self.beta * s1
        qDot2 = 0.5 * (q0*gy - q1*gz + q3*gx) - self.beta * s2
        qDot3 = 0.5 * (q0*gz + q1*gy - q2*gx) - self.beta * s3

        # Integrate to yield quaternion
        q0 += qDot0 / self.sample_freq
        q1 += qDot1 / self.sample_freq
        q2 += qDot2 / self.sample_freq
        q3 += qDot3 / self.sample_freq
        
        # Normalize quaternion
        norm_q = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        if norm_q == 0:
            return
        self.q0, self.q1, self.q2, self.q3 = q0/norm_q, q1/norm_q, q2/norm_q, q3/norm_q

@app.route('/analyze', methods=['POST'])
def analyze_fall_pattern():
    """Analyze historical fall patterns"""
    try:
        # Check if fall records exist
        if not os.path.exists(csv_file):
            return jsonify({"error": "No fall records found"}), 404
        
        # Load fall records
        falls_df = pd.read_csv(csv_file)
        
        # Basic statistics
        total_falls = len(falls_df)
        fall_types = falls_df['predicted_activity'].value_counts().to_dict()
        
        # Acceleration statistics
        acc_stats = {
            'mean_ax': falls_df['ax'].mean(),
            'mean_ay': falls_df['ay'].mean(),
            'mean_az': falls_df['az'].mean(),
            'max_ax': falls_df['ax'].max(),
            'max_ay': falls_df['ay'].max(),
            'max_az': falls_df['az'].max(),
        }
        
        # Calculate SMV for each fall
        falls_df['smv'] = (falls_df['ax']**2 + falls_df['ay']**2 + falls_df['az']**2).apply(lambda x: x**0.5)
        smv_stats = {
            'mean_smv': falls_df['smv'].mean(),
            'max_smv': falls_df['smv'].max(),
            'min_smv': falls_df['smv'].min()
        }
        
        return jsonify({
            'total_falls': total_falls,
            'fall_types': fall_types,
            'acceleration_stats': acc_stats,
            'smv_stats': smv_stats,
            'latest_fall': falls_df.iloc[-1]['timestamp'] if not falls_df.empty else None
        })
    
    except Exception as e:
        logger.error(f"Error analyzing fall patterns: {e}")
        return jsonify({"error": f"Error analyzing fall patterns: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)