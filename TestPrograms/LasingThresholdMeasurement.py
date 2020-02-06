"""
JL3562@cornell.edu
2020/01/29
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
from enum import Enum
import LowLevelModules.GeneralFunctions as gf

from pymeasure.instruments.keithley import Keithley2400
from ControlHardware.ThorlabsPowermeterPM100D import ThorlabPM100D

import threading

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
    class CtrlVars(Enum):
        WAVELENGTH="PM Wavelength [nm]"
        BEAM_DIAMETER="Beam diameter [mm]"
        START_CURR="Start current [mA]"
        STOP_CURR="Stop current [mA]"
        CURR_STEP="Current step [mA]"
        SOURCE_CURRENT_RANGE="Source cur. range [mA]"
        COMPLIANCE_VOLT="Compliance Voltage [V]"

    def createSpectrometerParamFormBox(self):
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
            self.typingControllers[i].setValue(self.params_1[i])
        # pass

    def initializeParams(self):
        '''
        Initialize the params in the GUI
        '''
        for i, param in enumerate(self.params):
            # self.params[i]= self.paramInitValues[i]
            self.params_1[i]= self.paramInitValues[i]
            # print(self.params)


    def updateSpecifiedParam(self, i):
        '''
        Update the specified parameters stored in a nparray; NOT sending to the instruments yet
        '''
        def updateParam_(value):
            self.params_1[i] = value
            self.params[self.paramNames[i]]= value
            # print(f"DEBUG: Changing {i}-th parameter: {self.paramNames[i]} to {value}")
            # print(f"DEBUG: all params listed:")
            # print(f"DEBUG: {self.params}")
        return updateParam_

    def createSpectrometerOperationButtonBox(self):
        '''
        Create buttons from a list of operations defined in __init__() method and link them to methods specified by getButtonAction()
        '''
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
        print(f"DEBUG: this is btSetParamAction. Currently maybe not even needed...")

    def btGetParamAction(self):
        print(f"DEBUG: this is btGetParamAction.")
        print(f"DEBUG: Parameters are:")
        print(f"DEBUG: dict: {self.params}")
        print(f"DEBUG: nparray: {self.params_1}")

    def btStartAction(self):
        print(f"this is btStartAction.")

        print(f"Starting measurements.")
        print(f"Parameters sent to instruments are")
        print(f"{self.params}")
        data= acquirePowerCurrentData()
        #TODO: Put header and figure into the GUI panel
        gf.data_save(data, header="Description of the data the one should probably have a place in the GUI")
        # run the measurement loop?
        
        
    def acquirePowerCurrentData(self):
        self.newDataAcquired= threading.Event()
        self.newDataAcquired.clear()
        sm = Keithley2400(self.keithley2400VName) # source meter
        pm = ThorlabPM100D(self.thorlabPM100DVName) # power meter
        
        print(f'connected to instruments...')

        currentStart=   1e-3 * self.params[self.CtrlVars.START_CURR]
        currentStop=    1e-3 * self.params[self.CtrlVars.STOP_CURR] # in A
        currentStep=    1e-3 * self.params[self.CtrlVars.CURR_STEP]

        sm.apply_current()                # Sets up to source current
        sm.source_current_range = 1e-3 * self.params[self.CtrlVars.SOURCE_CURRENT_RANGE]   # Sets the source current range to 1 mA
        sm.compliance_voltage = self.params[self.CtrlVars.COMPLIANCE_VOLT]        # Sets the compliance voltage to 10 V

        ''' Safety precaution '''
        sm.source_current = 0             # Sets the source current to 0 mA
        sm.enable_source()                # Enables the source output

        #allocate np array for storing data
        currents= np.arange(currentStart, currentStop, currentStep)
        NData= current.size
        powers= np.zeros(NData)
        powersStdDev= np.zeros(NData)

        np.zeros()
        #allocate np array for currents that will be swept;

        haltTimeCurrRamp2PowerMeas=3

        print(f'starting to acquire data')
        for i, cur in enumerate(currents):
            sm.ramp_to_current(cur)
            sm.measure_current() # run this before reading with accessing field sm.current
            time.sleep(haltTimeCurrRamp2PowerMeas)

            pmDatum= pm.measure(n=3, wavelength=self.params[self.CtrlVars.WAVELENGTH],\
                beamDiameter=self.params[self.CtrlVars.BEAM_DIAMETER])
            powers[i]= pmDatum(1)
            powersStdDev[i]= pmDatum(2)
            self.newDataAcquired.set() # for notifying some drawing component
            # time.sleep(0.025)


        sm.shutdown()                     # Ramps the current to 0 mA and disables output
        self.PowerCurrentDat= np.column_stack((currents, powers))
        
        return self.PowerCurrentDat


    def btAboutAction(self):
        print(f"this is btAboutAction.")

    def createSavefileBox(self):
        # specify custom save directory
        # use data_save will save an additional copy to the predefined data folder
        self.savefileBoxLayout= QFormLayout(self.savefileBox)
        return self.savefileBox


    def showAbout(self):
        pass


    def __init__(self):
        super(SpectrometerMainWindow, self).__init__()
        self.setWindowTitle("Spectrometer control")

        self.speParamBox = QGroupBox("Experiment Parameters", self)
        self.oprButtonBox = QGroupBox("Operations")
        self.savefileBox= QGroupBox("Data saving", self)

        self.paramNames=[]
        for item in self.CtrlVars:
            self.paramNames.append(item.value)

        # consider using an external text files to load the limits
        self.paramLimits= ([300, 800], [1, 5], [0, 50], [0, 50], [0, 10], [0, 50], [0, 5])
        self.paramInitValues= (532, 3, 0, 10, 1, 10, 1)

        self.NParams= len(self.paramNames)
        self.params_1 = np.zeros(self.NParams)

        self.params= {}
        for i, name in enumerate(self.paramNames):
            self.params[name]= self.params_1[i]

        '''
        consider restructure the data here using dictionary? {name: (value, [min, max, init])}
        '''


        self.buttonNames= ("Set Params", "Get Params", "Start", "About")
        self.actionMap= {\
                    "Set Params": self.btSetParamAction, \
                    "Get Params": self.btGetParamAction, \
                    "Start": self.btStartAction, \
                    "About": self.btAboutAction}

        self.thorlabPM100DVName= 'USB0::0x1313::0x8078::P0021814::INSTR'
        self.keithley2400VName= 'GPIB0::24::INSTR'

        self.mainLayout= QVBoxLayout()

        self.mainLayout.addWidget(self.createSpectrometerParamFormBox())
        self.mainLayout.addWidget(self.createSpectrometerOperationButtonBox())
        self.mainLayout.addWidget(self.createSavefileBox())

        self.setLayout(self.mainLayout)
        self.resize(300, self.NParams*50)
        self.move(QDesktopWidget().availableGeometry().center())
        self.show()

class PowerCurrentMeasurement:
    def __init__(self):
        pass

    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    configWindow = SpectrometerMainWindow()
    

    sys.exit(app.exec())