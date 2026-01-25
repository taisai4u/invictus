"""
Base class for all sensors
Mirrors C++ sensor_base.h
"""

from abc import ABC, abstractmethod
import numpy as np

class SensorBase(ABC):
    """Abstract base class for all sensors"""
    
    def __init__(self, name: str, sample_rate: float):
        self.name = name
        self.sample_rate = sample_rate  # Hz
        self.dt = 1.0 / sample_rate
        self.last_update_time = 0.0
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the sensor (like init() in C++)"""
        pass
    
    @abstractmethod
    def read_raw(self) -> np.ndarray:
        """Read raw sensor data"""
        pass
    
    @abstractmethod
    def read_scaled(self) -> np.ndarray:
        """Read calibrated/scaled sensor data"""
        pass
    
    @abstractmethod
    def calibrate(self, num_samples: int = 1000):
        """Perform sensor calibration"""
        pass
    
    def update(self, current_time: float):
        """Update internal state"""
        self.last_update_time = current_time