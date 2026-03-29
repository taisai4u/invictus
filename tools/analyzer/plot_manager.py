"""
Plot Management
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt

class PlotManager:
    
    def __init__(self, accel_plot, gyro_plot, altitude_plot):
        self.accel_plot = accel_plot
        self.gyro_plot = gyro_plot
        self.altitude_plot = altitude_plot
        
        self.time_line_accel = None
        self.time_line_gyro = None
        self.time_line_alt = None
        
        self.setup_plots()
    
    def setup_plots(self):
        """Configure plot appearance"""
        plots = [
            (self.accel_plot, 'Acceleration', 'm/s^2', 'Accelerometer'),
            (self.gyro_plot, 'Angular Velocity', 'rad/s', 'Gyroscope'),
            (self.altitude_plot, 'Altitude', 'm', 'Altitude')
        ]
        
        for plot, ylabel, units, title in plots:
            plot.setLabel('left', ylabel, units=units)
            plot.setLabel('bottom', 'Time', units='s')
            plot.setTitle(title)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setContentsMargins(0, 0, 0, 0)
            if plot != self.altitude_plot:
                plot.addLegend()
    
    def plot_data(self, df):
        """Plot sensor data"""
        time = df['timestamp'].values
        
        self.accel_plot.clear()
        self.gyro_plot.clear()
        self.altitude_plot.clear()
        
        self.setup_plots()  # Re-add legends
        
        # Accelerometer
        self.accel_plot.plot(time, df['accel_x'].values, 
                            pen=pg.mkPen(color='#ff4444', width=1.5), name='X')
        self.accel_plot.plot(time, df['accel_y'].values, 
                            pen=pg.mkPen(color='#44ff44', width=1.5), name='Y')
        self.accel_plot.plot(time, df['accel_z'].values, 
                            pen=pg.mkPen(color='#4444ff', width=1.5), name='Z')
        
        # Gyroscope
        self.gyro_plot.plot(time, df['gyro_x'].values, 
                           pen=pg.mkPen(color='#ff4444', width=1.5), name='X')
        self.gyro_plot.plot(time, df['gyro_y'].values, 
                           pen=pg.mkPen(color='#44ff44', width=1.5), name='Y')
        self.gyro_plot.plot(time, df['gyro_z'].values, 
                           pen=pg.mkPen(color='#4444ff', width=1.5), name='Z')
        
        # Altitude
        if 'pressure' in df.columns:
            sea_level = df['pressure'].iloc[0]
            altitude = 44330 * (1 - (df['pressure'] / sea_level) ** 0.1903)
            self.altitude_plot.plot(time, altitude.values, 
                                   pen=pg.mkPen(color='#00aaff', width=2))
    
    def update_time_indicator(self, current_time):
        """Update vertical time line"""
        # Remove old lines
        for line, plot in [(self.time_line_accel, self.accel_plot),
                           (self.time_line_gyro, self.gyro_plot),
                           (self.time_line_alt, self.altitude_plot)]:
            if line is not None:
                plot.removeItem(line)
        
        # Add new lines
        pen = pg.mkPen(color='#ff9500', width=2, style=Qt.DashLine)
        self.time_line_accel = self.accel_plot.addLine(x=current_time, pen=pen)
        self.time_line_gyro = self.gyro_plot.addLine(x=current_time, pen=pen)
        self.time_line_alt = self.altitude_plot.addLine(x=current_time, pen=pen)