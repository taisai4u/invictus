"""
barometer.py
Simulates DPS368 barometer
"""

import numpy as np
from .sensor_base import SensorBase

class Barometer(SensorBase):
    """Simulates DPS368 barometer"""
    
    def __init__(self, sample_rate=50):
        super().__init__("DPS368", sample_rate)
        
        # Reference conditions
        self.sea_level_pressure = 101325.0  # Pa
        self.temperature = 15.0  # Celsius
        
        # Noise
        self.pressure_noise_std = 5.0  # Pa
        
        # Current altitude
        self.true_altitude = 0.0  # meters
        
    def initialize(self) -> bool:
        print(f"[{self.name}] Initializing")
        self.is_initialized = True
        return True
    
    def set_altitude(self, altitude: float):
        """Set true altitude"""
        self.true_altitude = altitude
    
    def read_raw(self) -> int:
        """Read raw pressure (ADC)"""
        pressure = self._altitude_to_pressure(self.true_altitude)
        noise = np.random.normal(0, self.pressure_noise_std)
        return int(pressure + noise)
    
    def read_scaled(self) -> dict:
        """Read pressure and calculated altitude"""
        pressure = self.read_raw()
        altitude = self._pressure_to_altitude(pressure)
        
        return {
            'pressure': float(pressure),  # Pa
            'altitude': altitude,          # meters
            'temperature': self.temperature,  # Celsius
            'timestamp': self.last_update_time
        }
    
    def _altitude_to_pressure(self, altitude: float) -> float:
        """Barometric formula: altitude -> pressure"""
        return self.sea_level_pressure * (1 - 2.25577e-5 * altitude) ** 5.25588
    
    def _pressure_to_altitude(self, pressure: float) -> float:
        """Inverse barometric formula: pressure -> altitude"""
        return 44330.0 * (1.0 - (pressure / self.sea_level_pressure) ** 0.1903)
    
    def calibrate(self, num_samples: int = 100):
        """Calibrate sea level pressure"""
        print(f"[{self.name}] Calibrating sea level pressure...")
        
        pressure_sum = 0.0
        for _ in range(num_samples):
            pressure_sum += self.read_raw()
        
        self.sea_level_pressure = pressure_sum / num_samples
        print(f"[{self.name}] Sea level pressure: {self.sea_level_pressure:.2f} Pa")