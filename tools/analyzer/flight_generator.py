import numpy as np
import pandas as pd

# Flight parameters
dt = 0.001  # 1ms timestep (1000 Hz)
duration = 15.0  # 15 second flight
time = np.arange(0, duration, dt)
num_samples = len(time)

# Rocket parameters
mass = 2.5  # kg
thrust = 500  # N
burn_time = 1.0  # seconds
drag_coefficient = 0.3

# Earth constants
g = 9.81  # m/s^2
mag_field_strength = 50.0  # μT (Charlottesville, VA)
mag_inclination = np.deg2rad(65)
mag_declination = np.deg2rad(-10)
sea_level_pressure = 101325  # Pa

# Initialize arrays
accel_x = np.zeros(num_samples)
accel_y = np.zeros(num_samples)
accel_z = np.zeros(num_samples)
gyro_x = np.zeros(num_samples)
gyro_y = np.zeros(num_samples)
gyro_z = np.zeros(num_samples)
mag_x = np.zeros(num_samples)
mag_y = np.zeros(num_samples)
mag_z = np.zeros(num_samples)
pressure = np.zeros(num_samples)
temperature = np.zeros(num_samples)

# State variables for simulation
pos_x, pos_y, pos_z = 0.0, 0.0, 0.0
vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
pitch = 0.0  # Pitch angle (rotation about y-axis)
pitch_rate = 0.0

# Earth's magnetic field in world frame (NED)
mag_world = np.array([
    mag_field_strength * np.cos(mag_inclination) * np.cos(mag_declination),
    mag_field_strength * np.cos(mag_inclination) * np.sin(mag_declination),
    mag_field_strength * np.sin(mag_inclination)
])

print("Generating flight data...")

for i, t in enumerate(time):
    # Phase 1: Boost (0-1s) - Straight up with 500N thrust
    if t < burn_time:
        thrust_accel = thrust / mass
        accel_z_body = thrust_accel - g  # Net upward acceleration
        accel_x_body = 0.0
        accel_y_body = 0.0
    
    # Phase 2: Coast with gradual turn (1s-6s)
    elif t < 6.0:
        # Start pitching over gradually at t=1s
        target_pitch = np.deg2rad(45)  # 45 degree pitch over
        pitch_time = t - burn_time
        pitch = target_pitch * np.tanh(pitch_time / 2.0)  # Smooth transition
        pitch_rate = target_pitch / 2.0 * (1 - np.tanh(pitch_time / 2.0)**2)
        
        # Drag force (simplified)
        velocity = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        drag = 0.5 * drag_coefficient * velocity**2 / mass if velocity > 0 else 0
        
        # Acceleration in world frame
        accel_world_x = -drag * vel_x / (velocity + 0.001)
        accel_world_y = 0.0
        accel_world_z = -g - drag * vel_z / (velocity + 0.001)
        
        # Transform to body frame (simple rotation about y-axis)
        accel_x_body = accel_world_x * np.cos(pitch) + accel_world_z * np.sin(pitch)
        accel_y_body = accel_world_y
        accel_z_body = -accel_world_x * np.sin(pitch) + accel_world_z * np.cos(pitch)
    
    # Phase 3: Descent (6s-15s)
    else:
        pitch_rate *= 0.99  # Slowly stop rotating
        velocity = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        drag = 0.5 * drag_coefficient * velocity**2 / mass if velocity > 0 else 0
        
        accel_world_x = -drag * vel_x / (velocity + 0.001)
        accel_world_z = -g - drag * vel_z / (velocity + 0.001)
        
        accel_x_body = accel_world_x * np.cos(pitch) + accel_world_z * np.sin(pitch)
        accel_y_body = 0.0
        accel_z_body = -accel_world_x * np.sin(pitch) + accel_world_z * np.cos(pitch)
    
    # Update velocities and positions (world frame)
    accel_world_x_update = accel_x_body * np.cos(pitch) - accel_z_body * np.sin(pitch)
    accel_world_z_update = accel_x_body * np.sin(pitch) + accel_z_body * np.cos(pitch)
    
    vel_x += accel_world_x_update * dt
    vel_z += accel_world_z_update * dt
    
    pos_x += vel_x * dt
    pos_z += vel_z * dt
    
    # Ground collision
    if pos_z < 0:
        pos_z = 0
        vel_x = 0
        vel_z = 0
        accel_x_body = 0
        accel_z_body = 0
        pitch_rate = 0
    
    # Add noise to measurements
    noise_accel = 0.05
    noise_gyro = 0.002
    noise_mag = 0.3
    
    accel_x[i] = accel_x_body + np.random.normal(0, noise_accel)
    accel_y[i] = accel_y_body + np.random.normal(0, noise_accel)
    accel_z[i] = accel_z_body + g + np.random.normal(0, noise_accel)  # Add gravity back for IMU reading
    
    gyro_x[i] = 0.0 + np.random.normal(0, noise_gyro)  # No roll
    gyro_y[i] = pitch_rate + np.random.normal(0, noise_gyro)  # Pitch rate
    gyro_z[i] = 0.0 + np.random.normal(0, noise_gyro)  # No yaw
    
    # Magnetometer (rotated by pitch angle)
    mag_body_x = mag_world[0] * np.cos(pitch) + mag_world[2] * np.sin(pitch)
    mag_body_y = mag_world[1]
    mag_body_z = -mag_world[0] * np.sin(pitch) + mag_world[2] * np.cos(pitch)
    
    mag_x[i] = mag_body_x + np.random.normal(0, noise_mag)
    mag_y[i] = mag_body_y + np.random.normal(0, noise_mag)
    mag_z[i] = mag_body_z + np.random.normal(0, noise_mag)
    
    # Barometric pressure (altitude-dependent)
    altitude = pos_z
    pressure[i] = sea_level_pressure * (1 - 2.25577e-5 * altitude) ** 5.25588
    pressure[i] += np.random.normal(0, 5)  # Noise
    
    temperature[i] = 15.0 - 0.0065 * altitude + np.random.normal(0, 0.1)

    # Progress indicator
    if i % 1000 == 0:
        print(f"Progress: {100*i/num_samples:.1f}% (t={t:.1f}s, alt={pos_z:.1f}m)")

# Create DataFrame
df = pd.DataFrame({
    'timestamp': time,
    'accel_x': accel_x,
    'accel_y': accel_y,
    'accel_z': accel_z,
    'gyro_x': gyro_x,
    'gyro_y': gyro_y,
    'gyro_z': gyro_z,
    'mag_x': mag_x,
    'mag_y': mag_y,
    'mag_z': mag_z,
    'pressure': pressure,
    'temperature': temperature
})

# Create output directory if it doesn't exist
import os
os.makedirs('assets/sample_data', exist_ok=True)

# Save to CSV
output_file = 'assets/sample_data/simulated_flight.csv'
df.to_csv(output_file, index=False)

print(f"\nFlight simulation complete!")
print(f"Saved to: {output_file}")
print(f"Total samples: {num_samples:,}")
print(f"Duration: {duration:.1f} seconds")
print(f"Max altitude: {pos_z:.1f} meters ({pos_z*3.28084:.1f} feet)")
print(f"Max downrange: {pos_x:.1f} meters")