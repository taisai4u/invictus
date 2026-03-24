import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
                             QFileDialog, QTextEdit, QMessageBox, QScrollArea, 
                             QSlider)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# Configure PyQtGraph
pg.setConfigOption('background', '#0d0d0d')
pg.setConfigOption('foreground', '#d0d0d0')
pg.setConfigOptions(antialias=True)

class InvictusAnalyzer(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.display_data = None  # Downsampled data for plotting
        self.current_time = 0.0
        self.duration = 0.0
        self.time_line_accel = None
        self.time_line_gyro = None
        self.time_line_alt = None
        self.is_playing = False
        self.playback_speed = 1.0
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_step)
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Invictus Flight Data Analyzer")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
            }
            QLabel {
                color: #e8e8e8;
            }
            QTextEdit {
                background-color: #0d0d0d;
                color: #d0d0d0;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 12px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.6;
            }
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
            QScrollBar:vertical {
                background-color: #1a1a1a;
                width: 12px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #3a3a3a;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4a4a4a;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Set application font
        app_font = QFont()
        if sys.platform == "darwin":
            app_font = QFont("SF Pro Text", 11)
        elif sys.platform == "win32":
            app_font = QFont("Segoe UI", 10)
        else:
            app_font = QFont("Ubuntu", 10)
        app_font.setLetterSpacing(QFont.PercentageSpacing, 102)
        QApplication.setFont(app_font)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: #1a1a1a;")
        header_main_layout = QVBoxLayout(header_widget)
        header_main_layout.setContentsMargins(20, 15, 20, 15)
        header_main_layout.setSpacing(12)
        
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        title = QLabel("Invictus")
        title_font = QFont()
        if sys.platform == "darwin":
            title_font = QFont("SF Pro Display", 20, QFont.Light)
        elif sys.platform == "win32":
            title_font = QFont("Segoe UI", 20, QFont.Light)
        else:
            title_font = QFont("Ubuntu", 20, QFont.Light)
        title.setFont(title_font)
        title.setStyleSheet("color: #ffffff; letter-spacing: 1px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.status_label = QLabel("No data loaded")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        header_layout.addWidget(self.status_label)
        
        load_button = QPushButton("Load CSV")
        load_button.setFixedWidth(90)
        load_button.setFixedHeight(28)
        load_button.setCursor(Qt.PointingHandCursor)
        load_button.setStyleSheet("""
            QPushButton {
                background-color: #0a84ff;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #0077ed;
            }
            QPushButton:pressed {
                background-color: #006edb;
            }
        """)
        load_button.clicked.connect(self.load_flight_data)
        header_layout.addWidget(load_button)
        
        header_main_layout.addLayout(header_layout)
        
        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #2a2a2a;")
        header_main_layout.addWidget(divider)
        
        main_layout.addWidget(header_widget)
        
        # Scrollable content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.setSpacing(20)
        
        info_label = QLabel("Flight Data")
        info_label.setStyleSheet("color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;")
        scroll_layout.addWidget(info_label)
        
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setPlaceholderText("Load a CSV file to view flight data...")
        self.info_display.setFixedHeight(180)
        scroll_layout.addWidget(self.info_display)
        
        plots_label = QLabel("Sensor Data")
        plots_label.setStyleSheet("color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px;")
        scroll_layout.addWidget(plots_label)
        
        # Create plots with fixed margins
        self.accel_plot = PlotWidget()
        self.accel_plot.setLabel('left', 'Acceleration', units='m/s^2')
        self.accel_plot.setLabel('bottom', 'Time', units='s')
        self.accel_plot.setTitle('Accelerometer')
        self.accel_plot.addLegend()
        self.accel_plot.showGrid(x=True, y=True, alpha=0.3)
        self.accel_plot.setMinimumHeight(300)
        self.accel_plot.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(self.accel_plot)
        
        self.gyro_plot = PlotWidget()
        self.gyro_plot.setLabel('left', 'Angular Velocity', units='rad/s')
        self.gyro_plot.setLabel('bottom', 'Time', units='s')
        self.gyro_plot.setTitle('Gyroscope')
        self.gyro_plot.addLegend()
        self.gyro_plot.showGrid(x=True, y=True, alpha=0.3)
        self.gyro_plot.setMinimumHeight(300)
        self.gyro_plot.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(self.gyro_plot)
        
        self.altitude_plot = PlotWidget()
        self.altitude_plot.setLabel('left', 'Altitude', units='m')
        self.altitude_plot.setLabel('bottom', 'Time', units='s')
        self.altitude_plot.setTitle('Altitude')
        self.altitude_plot.showGrid(x=True, y=True, alpha=0.3)
        self.altitude_plot.setMinimumHeight(300)
        self.altitude_plot.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(self.altitude_plot)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)
        
        # Timeline controls
        timeline_widget = QWidget()
        timeline_widget.setStyleSheet("background-color: #1a1a1a; border-top: 1px solid #2a2a2a; padding: 0px;")
        timeline_layout = QVBoxLayout(timeline_widget)
        timeline_layout.setContentsMargins(20, 10, 20, 10)
        timeline_layout.setSpacing(8)
        
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(60)
        self.play_button.setFixedHeight(28)
        self.play_button.setCursor(Qt.PointingHandCursor)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #0a84ff;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0077ed;
            }
            QPushButton:pressed {
                background-color: #006edb;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
        """)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        self.time_label = QLabel("0.00s / 0.00s")
        self.time_label.setStyleSheet("color: #888; font-size: 11px;")
        controls_layout.addWidget(self.time_label)
        
        controls_layout.addStretch()
        timeline_layout.addLayout(controls_layout)
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(False)
        self.time_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #2a2a2a;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0a84ff;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #0077ed;
            }
        """)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        timeline_layout.addWidget(self.time_slider)
        
        main_layout.addWidget(timeline_widget, 0)
    
    def downsample_data(self, df, target_points=5000):
        """Downsample data for smooth plotting"""
        if len(df) <= target_points:
            return df
        
        step = len(df) // target_points
        return df.iloc[::step].copy()
    
    def load_flight_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Flight Data CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.status_label.setText("Loading...")
            QApplication.processEvents()
            
            df = pd.read_csv(file_path)
            
            required_columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 
                              'gyro_x', 'gyro_y', 'gyro_z']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Store full data
            self.data = df
            
            # Downsample for display
            self.display_data = self.downsample_data(df, target_points=5000)
            
            num_rows = len(df)
            duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
            sample_rate = num_rows / duration if duration > 0 else 0
            
            if 'pressure' in df.columns:
                sea_level = df['pressure'].iloc[0]
                min_pressure = df['pressure'].min()
                max_alt = 44330 * (1 - (min_pressure / sea_level) ** 0.1903)
            else:
                max_alt = None
            
            max_accel = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2).max()
            max_g = max_accel / 9.81
            
            info_lines = []
            info_lines.append("FLIGHT SUMMARY")
            info_lines.append("-" * 60)
            info_lines.append("")
            info_lines.append(f"File               {file_path.split('/')[-1].split(chr(92))[-1]}")
            info_lines.append(f"Duration           {duration:.2f} s")
            info_lines.append(f"Sample Rate        {sample_rate:.0f} Hz")
            info_lines.append(f"Total Samples      {num_rows:,}")
            info_lines.append(f"Display Samples    {len(self.display_data):,} (downsampled)")
            info_lines.append("")
            info_lines.append("PERFORMANCE")
            info_lines.append("-" * 60)
            info_lines.append("")
            
            if max_alt is not None:
                info_lines.append(f"Max Altitude       {max_alt:.1f} m  ({max_alt*3.28084:.0f} ft)")
            
            info_lines.append(f"Max Acceleration   {max_accel:.1f} m/s^2  ({max_g:.1f} G)")
            info_lines.append("")
            info_lines.append("SENSORS DETECTED")
            info_lines.append("-" * 60)
            info_lines.append("")
            info_lines.append("Accelerometer      accel_x, accel_y, accel_z")
            info_lines.append("Gyroscope          gyro_x, gyro_y, gyro_z")
            
            if 'mag_x' in df.columns:
                info_lines.append("Magnetometer       mag_x, mag_y, mag_z")
            if 'pressure' in df.columns:
                mag_line = "Barometer          pressure"
                if 'temperature' in df.columns:
                    mag_line += ", temperature"
                info_lines.append(mag_line)
            
            info_text = "\n".join(info_lines)
            self.info_display.setPlainText(info_text)
            
            filename = file_path.split('/')[-1].split(chr(92))[-1]
            self.status_label.setText(f"{filename} - {duration:.1f}s - {num_rows:,} samples")
            
            self.duration = duration
            self.time_slider.setEnabled(True)
            self.time_slider.setValue(0)
            self.time_label.setText(f"0.00s / {self.duration:.2f}s")
            self.play_button.setEnabled(True)
            
            self.plot_sensor_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error: {str(e)}")
            self.status_label.setText("Error loading file")
    
    def plot_sensor_data(self):
        if self.display_data is None:
            return
        
        df = self.display_data
        time = df['timestamp'].values
        
        self.accel_plot.clear()
        self.gyro_plot.clear()
        self.altitude_plot.clear()
        
        # Re-add legends
        self.accel_plot.addLegend()
        self.gyro_plot.addLegend()
        
        # Plot with downsampled data
        self.accel_plot.plot(time, df['accel_x'].values, 
                            pen=pg.mkPen(color='#ff4444', width=1.5), name='X')
        self.accel_plot.plot(time, df['accel_y'].values, 
                            pen=pg.mkPen(color='#44ff44', width=1.5), name='Y')
        self.accel_plot.plot(time, df['accel_z'].values, 
                            pen=pg.mkPen(color='#4444ff', width=1.5), name='Z')
        
        self.gyro_plot.plot(time, df['gyro_x'].values, 
                           pen=pg.mkPen(color='#ff4444', width=1.5), name='X')
        self.gyro_plot.plot(time, df['gyro_y'].values, 
                           pen=pg.mkPen(color='#44ff44', width=1.5), name='Y')
        self.gyro_plot.plot(time, df['gyro_z'].values, 
                           pen=pg.mkPen(color='#4444ff', width=1.5), name='Z')
        
        if 'pressure' in df.columns:
            sea_level = df['pressure'].iloc[0]
            altitude = 44330 * (1 - (df['pressure'] / sea_level) ** 0.1903)
            self.altitude_plot.plot(time, altitude.values, 
                                   pen=pg.mkPen(color='#00aaff', width=2))
    
    def on_slider_changed(self, value):
        if self.data is None:
            return
        
        self.current_time = (value / 1000.0) * self.duration
        self.time_label.setText(f"{self.current_time:.2f}s / {self.duration:.2f}s")
        self.update_time_indicator()
    
    def update_time_indicator(self):
        """Only update vertical line - don't replot data"""
        if self.time_line_accel is not None:
            self.accel_plot.removeItem(self.time_line_accel)
        if self.time_line_gyro is not None:
            self.gyro_plot.removeItem(self.time_line_gyro)
        if self.time_line_alt is not None:
            self.altitude_plot.removeItem(self.time_line_alt)
        
        pen = pg.mkPen(color='#ff9500', width=2, style=Qt.DashLine)
        self.time_line_accel = self.accel_plot.addLine(x=self.current_time, pen=pen)
        self.time_line_gyro = self.gyro_plot.addLine(x=self.current_time, pen=pen)
        self.time_line_alt = self.altitude_plot.addLine(x=self.current_time, pen=pen)
    
    def toggle_playback(self):
        if self.is_playing:
            self.animation_timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False
        else:
            self.animation_timer.start(16)  # 60 FPS
            self.play_button.setText("Pause")
            self.is_playing = True
    
    def animate_step(self):
        """Animation loop - only updates time indicator"""
        if self.data is None:
            return
        
        # Increment time
        time_step = 0.016 * self.playback_speed
        self.current_time += time_step
        
        # Loop back at end
        if self.current_time >= self.duration:
            self.current_time = 0.0
        
        # Update slider without triggering signal
        slider_value = int((self.current_time / self.duration) * 1000)
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(slider_value)
        self.time_slider.blockSignals(False)
        
        # Update display
        self.time_label.setText(f"{self.current_time:.2f}s / {self.duration:.2f}s")
        self.update_time_indicator()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = InvictusAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()