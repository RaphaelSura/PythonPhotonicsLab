"""
JL3562@cornell.edu
2020/01/23
"""
import sys
'''
PUT THE CORRECT ROOT FOLDER FOR THE PYTHON SCRIPTS
'''
sys.path.append("Z:\\PythonPhotonicsLab\\")

# from PyQt5.QtWidgets import QApplication, QWidget, QFormLayout, QGroupBox, QLabel, QLineEdit, QVBoxLayout, QDesktopWidget, QHBoxLayout
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np

'''
These are for LightField
'''
# Generic imports
import clr
import os
import time
# from System.IO import *
# from System import String
# from System.Collections.Generic import List
# from LowLevelModules.Spectroscopy import Spectrum

# Needed dll for interaction with LF
# sys.path.append(os.environ['LIGHTFIELD_ROOT'])
# sys.path.append(os.environ['LIGHTFIELD_ROOT'] + '\\AddInViews')
# clr.AddReference('PrincetonInstruments.LightFieldViewV5')
# clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
# clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')


class SpectrometerMainWindow(QWidget):
    def createSpectrometerParamFormBox(self):
        self.speParamBox = QGroupBox("Spectrometer Parameters", self)
        self.speParamLayout= QFormLayout()
        self.sliderControllers=[]
        self.typingControllers=[]

        for i, param in enumerate(self.paramNames):
            inputComboLayout = QHBoxLayout()
            
            typingControl = QDoubleSpinBox(self.speParamBox)
            
            sliderControl = QSlider(Qt.Horizontal)
            sliderControl.setTickPosition(QSlider.TicksAbove)
            
            inputComboLayout.addWidget(typingControl)
            inputComboLayout.addWidget(sliderControl)


            # print(f"connecting {i}-th signal {param} to a callback function")
            sliderControl.setMaximum(self.paramLimits[i][1])
            sliderControl.setMinimum(self.paramLimits[i][0])
            typingControl.setMaximum(self.paramLimits[i][1])
            typingControl.setMinimum(self.paramLimits[i][0])

            sliderControl.valueChanged.connect(typingControl.setValue)
            typingControl.valueChanged.connect(sliderControl.setValue)
            # this will set the value twice, I am curious what the best solution is here..
            sliderControl.valueChanged.connect(self.updateSpecifiedParam(i))
            self.sliderControllers.append(sliderControl)
            self.typingControllers.append(typingControl)
            # print(self.typingControllers)

            self.speParamLayout.addRow(QLabel(param), inputComboLayout)

        self.initializeParams()
        self.guiRefreshParams()

        self.speParamBox.setLayout(self.speParamLayout)
        return self.speParamBox

    def guiRefreshParams(self):
        '''
        Refresh the GUI to reflect the stored parameters
        '''
        for i, controller in enumerate(self.typingControllers):
            self.typingControllers[i].setValue(self.params[i])
        # pass

    def initializeParams(self):
        '''
        Initialize the params in the GUI 
        '''
        for i, param in enumerate(self.params):
            # self.params[i]= self.paramInitValues[i]
            self.params[i]= self.paramInitValues[i]
            print(self.params)
            

    def updateSpecifiedParam(self, i):
        '''
        Update the specified parameters stored in a nparray; NOT sending to the instruments yet
        '''
        def updateParam_(value):
            self.params[i] = value
            print(f"Changing {i}-th parameter: {self.paramNames[i]} to {value}")
            print(f"all params listed:")
            print(self.params)
        return updateParam_

    def createSpectrometerOperationButtonBox(self):
        '''
        Create buttons from a list of operations defined in __init__() method and link them to methods specified by getButtonAction()
        '''
        self.oprButtonBox = QGroupBox("Operations")
        self.oprButtonLayout= QHBoxLayout()
        self.buttons= []
        for opr in self.buttonNames:
            button= QPushButton(opr)
            self.oprButtonLayout.addWidget(button)
            button.clicked.connect(self.getButtonAction(opr))
            self.buttons.append(button)

        self.oprButtonBox.setLayout(self.oprButtonLayout)
        return self.oprButtonBox

    def getButtonAction(self, buttonName):
        return self.actionMap[buttonName]

    def btSetParamAction(self):
        print(f"this is btSetParamAction.")

    def btGetParamAction(self):
        print(f"this is btGetParamAction.")

    def btStartAction(self):
        print(f"this is btStartAction.")
        # this should set the spectrometer to start acquiring data

    def btAboutAction(self):
        print(f"this is btAboutAction.")

    def createSavefileBox(self):
        self.savefileBox= QGroupBox("Data saving", self)


        pass



    def __init__(self):
        super(SpectrometerMainWindow, self).__init__()
        self.setWindowTitle("Spectrometer control")
        self.paramNames= ("Wavelength [nm]", "Beam diameter [mm]", \
            "Spectrum acq. time [s]", "Spectr. acq. start wavelength [nm]", \
                "Spectr. acq. stop wavelength [nm]")

        # consider using an external text files to load the limits
        self.paramLimits= ([300, 800], [1, 5], [0, 120], [200, 800], [200, 1000])
        self.paramInitValues= (400, 3, 30, 300, 800)
        self.buttonNames= ("Set Params", "Get Params", "Start", "About")
        self.actionMap= {\
                    "Set Params": self.btSetParamAction, \
                    "Get Params": self.btGetParamAction, \
                    "Start": self.btStartAction, \
                    "About": self.btAboutAction}
        
        
        self.params = np.zeros(len(self.paramNames))
        self.NParams= len(self.paramNames)
        self.mainLayout= QVBoxLayout()

        self.mainLayout.addWidget(self.createSpectrometerParamFormBox())
        self.mainLayout.addWidget(self.createSpectrometerOperationButtonBox())
        # self.mainLayout.addWidget()
        self.setLayout(self.mainLayout)
        self.resize(300, self.NParams*50)
        self.move(QDesktopWidget().availableGeometry().center())
        self.show()

    def showAbout(self):
        pass




if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    configWindow = SpectrometerMainWindow()
    app.exec()