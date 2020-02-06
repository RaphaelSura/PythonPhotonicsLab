import sys
sys.path.append("Z:\\PythonPhotonicsLab\\")
import logging
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



class MainApplication():
    def __init__(self):
        bufferSize= 10
        self.daq= Daq(bufferSize)
        self.printer= XYPrinter(bufferSize)

        self.daqThread= threading.Thread(name='daq', target=self.daq.acquireAtRandomTimes)
        self.printerThread= threading.Thread(name='console message',target=self.printer.fetchDaqBufferContinuously, args=(self.daq,))

        self.qApp = QtWidgets.QApplication(sys.argv)

        self.aw = ApplicationWindow()
        self.aw.setWindowTitle("Meow")
        self.aw.linkDaq(self.daq)
        self.aw.show()

    def go(self):
        self.daqThread.daemon= True
        self.printerThread.daemon= True
        self.daqThread.start()
        self.printerThread.start()
        self.aw.startCanvasEventThread()
        sys.exit(self.qApp.exec())

        # self.daqThread.join()
        # self.printerThread.join()


class XYPrinter():
    def __init__(self, bufferSize=10):
        self.xCache=np.zeros(bufferSize)
        self.yCache=np.zeros(bufferSize)

    def fetchDaqBuffer(self, daq):
        self.xCache= daq.x
        self.yCache= daq.y
        self.xyprint()

    def fetchDaqBufferContinuously(self, daq, intervalSec=2):
        while True:
            self.fetchDaqBuffer(daq)
            time.sleep(intervalSec)

    def xyprint(self):
        print(self.xCache)
        print(self.yCache)


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

    def fetchData(self, daq):
        '''
        Assuming daq.x and daq.y are 1-D array-like object
        '''
        self.x= daq.xa
        self.y= daq.ya

    def updateFigure(self):
        self.axes.cla()
        self.axes.plot(self.x, self.y, 'ro-' )
        self.draw()

    def updateFigureContinuously(self, daq):
        while True:
            self.fetchData(daq)
            self.updateFigure()
            time.sleep(3)

    def updateWhenNotified(self, daq, event):
        while True:
            event.wait()
            print(f"new data painted")
            self.fetchData(daq)
            self.updateFigure()
            event.clear()


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

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QVBoxLayout(self.main_widget)
        self.dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(self.dc)


        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        self.close()

    def linkDaq(self, daq):
        self.daq= daq

    def startCanvasEventThread(self):
        canvasThread= threading.Thread(name='canvas', target= self.dc.updateWhenNotified, args= (self.daq, self.daq.newDataAcquired, ))
        canvasThread.daemon= True
        canvasThread.start()
        # canvasThread.join()

    def startCanvasThread(self):
        dynamicCanvasThread= threading.Thread(name='canvas', target= self.dc.updateFigureContinuously, args= (self.daq,))
        dynamicCanvasThread.daemon= True
        dynamicCanvasThread.start()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """embedding_in_qt5.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a simple example of a Qt5 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation.

This is modified from the embedding in qt4 example to show the difference
between qt4 and qt5"""
                                )


if __name__ == "__main__":

    testProg= MainApplication()
    testProg.go()
