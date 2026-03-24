import pandas as pd
import numpy as np

class DataHandler:
    
    @staticmethod
    def load_csv(file_path):
        """Load and validate CSV"""
        df = pd.read_csv(file_path)
        
        required = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 
                   'gyro_x', 'gyro_y', 'gyro_z']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        
        return df
    
    @staticmethod
    def downsample_for_display(df, target_points=5000):
        """Downsample data for smooth plotting"""
        if len(df) <= target_points:
            return df
        
        # Take every Nth point
        step = len(df) // target_points
        return df.iloc[::step].copy()
    
    @staticmethod
    def calculate_stats(df):
        """Calculate flight statistics"""
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        sample_rate = len(df) / duration if duration > 0 else 0
        
        max_alt = None
        if 'pressure' in df.columns:
            sea_level = df['pressure'].iloc[0]
            min_pressure = df['pressure'].min()
            max_alt = 44330 * (1 - (min_pressure / sea_level) ** 0.1903)
        
        max_accel = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2).max()
        max_g = max_accel / 9.81
        
        return {
            'num_rows': len(df),
            'duration': duration,
            'sample_rate': sample_rate,
            'max_altitude': max_alt,
            'max_accel': max_accel,
            'max_g': max_g
        }
    
    @staticmethod
    def format_info_text(file_path, stats, df):
        """Format flight summary text"""
        lines = [
            "FLIGHT SUMMARY",
            "-" * 60,
            "",
            f"File               {file_path.split('/')[-1].split(chr(92))[-1]}",
            f"Duration           {stats['duration']:.2f} s",
            f"Sample Rate        {stats['sample_rate']:.0f} Hz",
            f"Total Samples      {stats['num_rows']:,}",
            "",
            "PERFORMANCE",
            "-" * 60,
            ""
        ]
        
        if stats['max_altitude']:
            lines.append(f"Max Altitude       {stats['max_altitude']:.1f} m  "
                        f"({stats['max_altitude']*3.28084:.0f} ft)")
        
        lines.append(f"Max Acceleration   {stats['max_accel']:.1f} m/s^2  "
                    f"({stats['max_g']:.1f} G)")
        lines.extend(["", "SENSORS DETECTED", "-" * 60, "",
                     "Accelerometer      accel_x, accel_y, accel_z",
                     "Gyroscope          gyro_x, gyro_y, gyro_z"])
        
        if 'mag_x' in df.columns:
            lines.append("Magnetometer       mag_x, mag_y, mag_z")
        if 'pressure' in df.columns:
            line = "Barometer          pressure"
            if 'temperature' in df.columns:
                line += ", temperature"
            lines.append(line)
        
        return "\n".join(lines)