'''
jl3562@cornell.edu
'''
import sys

sys.path.append("Z:\\PythonPhotonicsLab\\")
import threading
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


class DynamicMPLCanvas(MPLCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MPLCanvas.__init__(self, *args, **kwargs)
        self.xLabel = "init X label"
        self.yLabel = "init Y label"
        self.setXYLabels(self.xLabel, self.yLabel)

    def setXYLabels(self, xl, yl):
        self.axes.set_xlabel(xl)
        self.axes.set_ylabel(yl)

    def fetchData(self, daq):
        '''
        Assuming daq.x and daq.y are 1-D array-like object; would need to revise this method every time
        you change the daq module... something to think about
        '''
        self.x = daq.xa
        self.y = daq.ya

    # def fetchData(self, daqX, daqY):
    #     self.x= daqX
    #     self.y= daqY

    def updateFigure(self):
        self.axes.cla()
        self.axes.plot(self.x, self.y, 'ro-')
        self.setXYLabels(self.xLabel, self.yLabel)
        self.draw()

    #TODO: something is strange about having a function being called in a loop in a thread and when I try to call the same function from another place it crashes the program
    def updateWhenNotified(self, daq, canvasUpdateFlag):
        while True:
            canvasUpdateFlag.wait()
            print(f"new data painted")
            self.fetchData(daq)
            self.updateFigure()
            canvasUpdateFlag.clear()

    def updateAxesLabels(self, xlabel="default X", ylabel="default Y"):
        self.xLabel = xlabel
        self.yLabel = ylabel
        self.setXYLabels(xlabel, ylabel)
        self.draw()
        # print(self.xLabel)
        # self.updateFigure()


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

        mainWindowLayout = QtWidgets.QVBoxLayout(self.mainWidget)
        self.dc = DynamicMPLCanvas(self.mainWidget, width=5, height=4, dpi=100)
        mainWindowLayout.addWidget(self.dc)
        mainWindowLayout.addWidget(self.createGraphOptionControlGUI())

        self.mainWidget.setFocus()
        self.setCentralWidget(self.mainWidget)

        self.daq = None
        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        self.close()

    def createGraphOptionControlGUI(self):
        self.graphOptionBox = QtWidgets.QGroupBox("Graph options")
        self.graphOptionLayout = QtWidgets.QFormLayout()
        self.graphXlabel = QtWidgets.QLineEdit()
        self.graphYlabel = QtWidgets.QLineEdit()

        self.graphXlabel.setText("init X label")
        self.graphYlabel.setText("init Y label")

        self.graphXlabel.textChanged.connect(self.updateGraphLabels)
        self.graphYlabel.textChanged.connect(self.updateGraphLabels)
        self.graphOptionLayout.addRow("X axis label", self.graphXlabel)
        self.graphOptionLayout.addRow("Y axis label", self.graphYlabel)
        self.graphOptionBox.setLayout(self.graphOptionLayout)
        return self.graphOptionBox

    def updateGraphLabels(self):
        self.dc.updateAxesLabels(self.graphXlabel.text(), self.graphYlabel.text())
        # print(self.graphXlabel.text())

    def linkDaq(self, daq):
        """
        Give a reference of the daq to this GUI
        """
        self.daq = daq

    def startCanvasEventThread(self):
        canvasThread = threading.Thread(name='canvas',
                                        daemon=True,
                                        target=self.dc.updateWhenNotified,
                                        args=(self.daq, self.daq.newDataAcquired,))
        canvasThread.daemon = True
        canvasThread.start()
        # canvasThread.join()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """Qt5 implementation for data acquisition real-time plotting"""
                                    )
