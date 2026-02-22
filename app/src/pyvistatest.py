import sys
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle("PyVista in Qt Window Example")
        self.resize(800, 600)

        # Create a central widget and a layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Embed the PyVista QtInteractor
        # This acts as the plotting widget
        self.plotter = QtInteractor(self.central_widget)
        self.main_layout.addWidget(self.plotter)

        # Add a mesh to the plotter (e.g., a sphere)
        self.mesh = pv.Sphere()
        self.plotter.add_mesh(self.mesh, color="skyblue")
        self.plotter.add_axes()
        self.plotter.show_grid()  # Optional: add a grid for visual reference

        # Optional: Add other Qt widgets (e.g., a button)
        self.button = QtWidgets.QPushButton("Update Plot")
        self.button.clicked.connect(self.update_plot)
        self.main_layout.addWidget(self.button)

    def update_plot(self):
        # Example function to update the plot (e.g., change sphere resolution)
        print("Updating plot...")
        self.plotter.clear_meshes()  # Clear existing meshes
        new_mesh = pv.Sphere(phi=30, theta=30)  # Create a more detailed sphere
        self.plotter.add_mesh(new_mesh, color="lightcoral")
        self.plotter.reset_camera()  # Reset camera to fit the new mesh

    def closeEvent(self, event):
        """Handle the closing of the window properly."""
        self.plotter.close()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
