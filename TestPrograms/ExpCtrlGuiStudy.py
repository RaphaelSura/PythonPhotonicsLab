'''
jl3562@cornell.edu
'''

import sys

sys.path.append("Z:\\PythonPhotonicsLab\\")
import numpy as np

import TestPrograms.DataGUIWindow as DataGUI
import TestPrograms.ExpCtrlGUIWindow as CtrlGUI
import TestPrograms.DataAcquisition as Daq

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class MainWindow(QWidget):
    def __init__(self):
        # self.daq = Daq.MockExperiment()
        # self.daq = Daq.PowerCurrentDaq()
        # self.daq.parameterInit()

        self.daq = Daq.LaserDiodeSpectroscopyDaq()

        self.qtApp = QApplication(sys.argv)
        self.dataGUI = DataGUI.DataGUIWindow()
        self.ctrlGUI = CtrlGUI.ExperimentCtrlGUIWindow()

        self.ctrlGUI.linkDaq(self.daq)
        self.dataGUI.linkDaq(self.daq)

        self.daq.linkCtrlGUI(self.ctrlGUI)

        self.dataGUI.move(self.ctrlGUI.geometry().width() + 1, 0)

        self.dataGUI.show()
        self.ctrlGUI.show()

    def run(self):
        self.dataGUI.startCanvasEventThread()

        self.qtApp.exec()
        self.daq.disableSources()


if __name__ == "__main__":
    program = MainWindow()
    program.run()
