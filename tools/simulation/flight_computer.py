"""
Main flight computer logic
"""

import numpy as np
from sensors.imu import IMU
from sensors.magnetometer import Magnetometer
from sensors.barometer import Barometer

class FlightComputer:
    """
    Main flight computer class
    Manages sensors, estimation, and control
    """
    
    def __init__(self, loop_rate=500):
        """
        Initialize flight computer
        
        Args:
            loop_rate: Main loop frequency in Hz
        """
        self.loop_rate = loop_rate
        self.dt = 1.0 / loop_rate
        
        # Sensors
        self.imu = IMU(sample_rate=1000)
        self.mag = Magnetometer(sample_rate=100)
        self.baro = Barometer(sample_rate=50)
        
        # State estimation 
        self.estimator = None
        
        # Control 
        self.controller = None
        
        # Flight state
        self.time = 0.0
        self.is_armed = False
        self.flight_phase = "IDLE"
        
        # Sensor data storage
        self.sensor_data = {
            'accel': np.zeros(3),
            'gyro': np.zeros(3),
            'mag': np.zeros(3),
            'pressure': 0.0,
            'altitude': 0.0
        }
        
    def initialize(self):
        """Initialize all subsystems"""
        print("=" * 50)
        print("INVICTUS FLIGHT COMPUTER INITIALIZATION")
        print("=" * 50)
        
        # Initialize sensors
        if not self.imu.initialize():
            print("[ERROR] IMU initialization failed")
            return False
            
        if not self.mag.initialize():
            print("[ERROR] Magnetometer initialization failed")
            return False
            
        if not self.baro.initialize():
            print("[ERROR] Barometer initialization failed")
            return False
        
        # Calibrate sensors
        print("\nCalibrating sensors...")
        self.imu.calibrate(num_samples=1000)
        self.baro.calibrate(num_samples=100)
        
        print("\n Flight computer initialized!")
        print("=" * 50)
        return True
    
    def read_sensors(self):
        """Read all sensors (called every loop)"""
        # Read IMU at high rate
        imu_data = self.imu.read_scaled()
        self.sensor_data['accel'] = imu_data['accel']
        self.sensor_data['gyro'] = imu_data['gyro']
        
        # Read mag (slower rate)
        mag_data = self.mag.read_scaled()
        self.sensor_data['mag'] = mag_data['mag']
        
        # Read baro (slower rate)
        baro_data = self.baro.read_scaled()
        self.sensor_data['pressure'] = baro_data['pressure']
        self.sensor_data['altitude'] = baro_data['altitude']
    
    def update_estimator(self):
        """Run state estimation (ESKF will go here)"""
        # TODO: Implement ESKF
        # For now, just pass
        pass
    
    def update_controller(self):
        """Run control algorithm"""
        # TODO: Implement PID/LQR
        pass
    
    def update(self):
        """Main update loop (called at loop_rate Hz)"""
        # 1. Read sensors
        self.read_sensors()
        
        # 2. Update state estimator
        self.update_estimator()
        
        # 3. Update controller
        self.update_controller()
        
        # 4. Update time
        self.time += self.dt
    
    def get_state(self):
        """Get current state estimate"""
        return {
            'time': self.time,
            'sensor_data': self.sensor_data,
            'flight_phase': self.flight_phase
        }