"""
simulates icm42688 Inertial Measurement Unit
mirrors icm42688.cpp driver
"""

import numpy as np
from .sensor_base import SensorBase

class IMU(SensorBase):
    """Simulates ICM-42688-P 6-DOF IMU"""
    
    # Sensor specifications (from datasheet)
    ACCEL_SCALE_2G = 2.0 / 32768.0      
    GYRO_SCALE_250DPS = 250.0 / 32768.0 
    
    def __init__(self, sample_rate=1000):
        super().__init__("ICM-42688-P", sample_rate)
        
        # Noise characteristics (from datasheet)
        self.accel_noise_density = 0.01  # m/s^2/sqrt(Hz)
        self.gyro_noise_density = 0.001  # rad/s/sqrt(Hz)
        
        # Bias (will be calibrated)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        
        # Calibration data
        self.is_calibrated = False
        
        # True state 
        self.true_accel = np.array([0.0, 0.0, 9.81])  # Gravity
        self.true_gyro = np.array([0.0, 0.0, 0.0])
        
    def initialize(self) -> bool:
        """Initialize sensor (like I2C setup in C++)"""
        print(f"[{self.name}] Initializing...")

        self.is_initialized = True
        return True
    
    def set_true_motion(self, accel: np.ndarray, gyro: np.ndarray):
        """Set true motion (for simulation only, not in C++)"""
        self.true_accel = accel
        self.true_gyro = gyro
    
    def read_raw(self) -> dict:
        """Read raw sensor values (16-bit ADC counts)"""

        
        # Add noise
        accel_noise = np.random.normal(0, self.accel_noise_density * np.sqrt(self.sample_rate), 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_density * np.sqrt(self.sample_rate), 3)
        
        # Simulate measurements
        accel_meas = self.true_accel + accel_noise
        gyro_meas = self.true_gyro + self.gyro_bias + gyro_noise
        
        # Convert to 16-bit values (simulate ADC)
        accel_raw = (accel_meas / (self.ACCEL_SCALE_2G * 9.81)).astype(np.int16)
        gyro_raw = (np.rad2deg(gyro_meas) / self.GYRO_SCALE_250DPS).astype(np.int16)
        
        return {
            'accel_raw': accel_raw,
            'gyro_raw': gyro_raw,
            'timestamp': self.last_update_time
        }
    
    def read_scaled(self) -> dict:
        """Read calibrated sensor data in physical units"""
        raw = self.read_raw()
        
        accel_g = raw['accel_raw'] * self.ACCEL_SCALE_2G
        accel_ms2 = accel_g * 9.81  # Convert to m/s^2
        
        gyro_dps = raw['gyro_raw'] * self.GYRO_SCALE_250DPS
        gyro_rads = np.deg2rad(gyro_dps)  # Convert to rad/s
        
        if self.is_calibrated:
            gyro_rads -= self.gyro_bias
        
        return {
            'accel': accel_ms2,      # m/s^2 
            'gyro': gyro_rads,       # rad/s 
            'timestamp': raw['timestamp']
        }
    
    def calibrate(self, num_samples: int = 1000):
        """Calibrate gyro bias (assumes stationary)"""
        print(f"[{self.name}] Calibrating with {num_samples} samples...")
        
        gyro_sum = np.zeros(3)
        
        for i in range(num_samples):
            data = self.read_scaled()
            gyro_sum += data['gyro']
        
        self.gyro_bias = gyro_sum / num_samples
        self.is_calibrated = True
        
        print(f"[{self.name}] Gyro bias: {self.gyro_bias}")
        return self.gyro_bias