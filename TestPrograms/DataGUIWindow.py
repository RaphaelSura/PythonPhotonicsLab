'''
jl3562@cornell.edu
'''
import sys
# sys.path.append("Z:\\PythonPhotonicsLab\\")
import threading
import time
import numpy as np
import TestPrograms.MockDataGenerator as MDG

import os
import random
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MPLCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(121)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class Daq():
    def __init__(self, bufferSize=10):
        self.x = np.zeros(bufferSize)
        self.y = np.zeros(bufferSize)
        self.xa= []
        self.ya= []
        self.cur= 0
        self.bufferSize= bufferSize
        self.mockDataGen= MDG.MockDataGenerator()
        self.tWaitGen= MDG.MockDataGenerator()

        self.newDataAcquired= threading.Event()
        self.newDataAcquired.clear()
        pass

    def acquireNext(self):
        x0= self.cur % self.bufferSize
        self.x[x0]= x0
        newY= self.mockDataGen.generateNoisedLinearPair(x0, sigma=1)[1]
        self.y[x0]= newY

        self.ya.append(self.mockDataGen.generateNoisedLinearPair(self.cur, sigma=1)[1])
        self.xa.append(self.cur)

        self.cur= self.cur+1

    def acquirePerInterval(self, intervalSec=0.1):
        while True:
            print(f"Daq: acquiring {self.cur}-th number")
            self.acquireNext()
            time.sleep(intervalSec)

    def acquireAtRandomTimes(self, intervalMean= 0.5):
        while True:
            tWait= self.tWaitGen.generateNoisedLinearPair(intervalMean, sigma=2)[1]
            while tWait<0:
                tWait= self.tWaitGen.generateNoisedLinearPair(intervalMean, sigma=2)[1]
            print(f"Daq: acquiring {self.cur}-th number")
            self.acquireNext()
            self.newDataAcquired.set()
            time.sleep(tWait)
            print(f'waited for {tWait} seconds')

class DynamicMPLCanvas(MPLCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MPLCanvas.__init__(self, *args, **kwargs)

    def fetchData(self, daq):
        '''
        Assuming daq.x and daq.y are 1-D array-like object; would need to revise this method every time
        you change the daq module... something to think about
        '''
        self.x= daq.xa
        self.y= daq.ya

    # def fetchData(self, daqX, daqY):
    #     self.x= daqX
    #     self.y= daqY

    def updateFigure(self):
        self.axes.cla()
        self.axes.plot(self.x, self.y, 'ro-' )
        self.draw()

    def updateWhenNotified(self, daq, event):
        while True:
            event.wait()
            print(f"new data painted")
            self.fetchData(daq)
            self.updateFigure()
            event.clear()

class DataGUIWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Data plot")

        self.fileMenu = QtWidgets.QMenu('&File', self)
        self.fileMenu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.fileMenu)

        self.helpMenu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.helpMenu)

        self.helpMenu.addAction('&About', self.about)

        self.mainWidget = QtWidgets.QWidget(self)

        l = QtWidgets.QVBoxLayout(self.mainWidget)
        self.dc = DynamicMPLCanvas(self.mainWidget, width=5, height=4, dpi=100)
        l.addWidget(self.dc)


        self.mainWidget.setFocus()
        self.setCentralWidget(self.mainWidget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        self.close()

    def linkDaq(self, daq):
        '''
        Give a reference of the daq to this GUI
        '''
        self.daq= daq

    def startCanvasEventThread(self):
        canvasThread= threading.Thread(name='canvas',
                                       daemon= True,
                                       target= self.dc.updateWhenNotified,
                                       args= (self.daq, self.daq.newDataAcquired, ))
        canvasThread.daemon= True
        canvasThread.start()
        # canvasThread.join()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """Qt5 implementation for data acquisition real-time plotting"""
                                )

class TestMainApplication():
    '''
    only for testing this DataGUIWindow.. using a daqThread to "acquire" data from a mock data generator
    '''
    def __init__(self):
        bufferSize= 10
        self.daq= Daq(bufferSize)
        # self.printer= XYPrinter(bufferSize)

        self.daqThread= threading.Thread(name='daq', target=self.daq.acquireAtRandomTimes)

        self.qApp = QtWidgets.QApplication(sys.argv)

        self.aw = DataGUIWindow()
        self.aw.setWindowTitle("Meow")
        self.aw.linkDaq(self.daq)
        self.aw.show()

    def go(self):
        self.daqThread.daemon= True
        # self.printerThread.daemon= True
        self.daqThread.start()
        # self.printerThread.start()
        self.aw.startCanvasEventThread()
        sys.exit(self.qApp.exec())

        # self.daqThread.join()
        # self.printerThread.join()


if __name__ == "__main__":

    testProg= TestMainApplication()
    testProg.go()
