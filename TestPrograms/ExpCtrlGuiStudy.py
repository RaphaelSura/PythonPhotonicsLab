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
        self.daq = Daq.MockExperiment()

        self.qtApp = QApplication(sys.argv)
        self.dataGUI = DataGUI.DataGUIWindow()
        self.ctrlGUI = CtrlGUI.SpectrometerCtrlGUIWindow()

        self.ctrlGUI.linkDaq(self.daq)
        self.dataGUI.linkDaq(self.daq)

        self.dataGUI.show()
        self.ctrlGUI.show()

    def run(self):
        self.dataGUI.startCanvasEventThread()

        self.qtApp.exec()


if __name__ == "__main__":
    program = MainWindow()
    program.run()
   