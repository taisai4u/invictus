"""
Simulates MMC5983MA Magnetometer
mirrors mmc5983.cpp driver
"""

import numpy as np
from .sensor_base import SensorBase

class Magnetometer(SensorBase):
    """Simulates MMC5983MA 3-axis magnetometer"""
    
    def __init__(self, sample_rate=100):
        super().__init__("MMC5983MA", sample_rate)
        
        # Earth's magnetic field (Charlottesville, VA)
        self.mag_field_strength = 50.0  # μT
        self.inclination = np.deg2rad(65.0)  # Dip angle
        self.declination = np.deg2rad(-10.0)  # From true north
        
        # Noise
        self.noise_std = 0.5  # μT
        
        # World frame magnetic field 
        self.mag_world = np.array([
            self.mag_field_strength * np.cos(self.inclination) * np.cos(self.declination),
            self.mag_field_strength * np.cos(self.inclination) * np.sin(self.declination),
            self.mag_field_strength * np.sin(self.inclination)
        ])
        
    def initialize(self) -> bool:
        print(f"[{self.name}] Initializing")
        self.is_initialized = True
        return True
    
    def set_orientation(self, quaternion):
        """Set current orientation (for body frame transformation)"""
        self.current_quat = quaternion
    
    def read_raw(self) -> np.ndarray:
        """Read raw magnetometer (ADC counts)"""

        noise = np.random.normal(0, self.noise_std, 3)
        return self.mag_world + noise
    
    def read_scaled(self) -> dict:
        """Read magnetometer in microT"""
        mag = self.read_raw()
        
        return {
            'mag': mag,  # microT (3D vector)
            'timestamp': self.last_update_time
        }
    
    def calibrate(self, num_samples: int = 1000):
        """Calibrate hard/soft iron"""
        print(f"[{self.name}] Calibration not implemented yet")
        pass