from __future__ import unicode_literals
import sys
import os
import random
import matplotlib
import numpy as np
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import MockDataGenerator as MDG

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)

        timer.timeout.connect(self.update_figure)

        timer.start(1000)

    def linkDataCache(self, xdata, ydata):
        self.xdata= xdata
        self.ydata= ydata

    def compute_initial_figure(self):
        pass

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)

        self.axes.cla()
        self.axes.plot(self.xdata, self.ydata, 'ro-' )
        self.draw()

class Experiment:
    def __init__(self, ndata=20):
        self.NDat= ndata
        self.xdata= np.zeros(self.NDat)
        self.ydata= np.zeros(self.NDat)
        self.mdg= MDG.MockDataGenerator()
        self.cur= 0

    def updateData(self):
        pair= self.mdg.generateNoisedLinearPair(self.cur % 100)
        self.xdata[self.cur]= pair[0]
        self.ydata[self.cur]= pair[1]
        self.cur= (self.cur+1) % ndata

    def updateCanvas(self, canvas):
        canvas.update_figure()

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.main_widget = QtWidgets.QWidget(self)

        self.dataCache= Experiment(50000)


        l = QtWidgets.QVBoxLayout(self.main_widget)
        dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        dc.linkDataCache(self.dataCache.xdata, self.dataCache.ydata)
        l.addWidget(dc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()


qApp = QtWidgets.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("Modified!")
aw.show()
sys.exit(qApp.exec_())