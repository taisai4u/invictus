"""
main.py
Test the flight computer in simulation
"""

import time
from flight_computer import FlightComputer

def main():
    # Create flight computer
    fc = FlightComputer(loop_rate=500)
    
    # Initialize
    if not fc.initialize():
        print("Initialization failed!")
        return
    
    # Simulate bench test (stationary)
    fc.imu.set_true_motion(
        accel=np.array([0, 0, 9.81]),  # Just gravity
        gyro=np.array([0, 0, 0])        # No rotation
    )
    
    # Run for 5 seconds
    print("\nRunning simulation...")
    for i in range(2500):  # 5 seconds at 500 Hz
        fc.update()
        
        # Print every 100 loops (0.2 seconds)
        if i % 100 == 0:
            state = fc.get_state()
            print(f"Time: {state['time']:.2f}s | "
                  f"Accel: {state['sensor_data']['accel']} | "
                  f"Alt: {state['sensor_data']['altitude']:.2f}m")
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    import numpy as np
    main()