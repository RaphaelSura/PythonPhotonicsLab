# JL3562@cornell.edu
# created: 2020/01/21

# basic operations with Keithley instrument
#%%
import sys

sys.path.append("..")

from pymeasure.instruments.keithley import Keithley2400
from ThorlabsPowermeterPM100D import ThorlabPM100D
import numpy as np
import time
import LowLevelModules.GeneralFunctions as gf
# Qt for GUI
from PyQt5.QtWidgets import *


def PIMeasure():

    pass

#%%

thorlabPM100DVName= 'USB0::0x1313::0x8078::P0021814::INSTR'
keithley2400VName= "GPIB0::24::INSTR"
sm = Keithley2400(keithley2400VName) # source meter
pm = ThorlabPM100D(thorlabPM100DVName)

currentStart= 0
currentStop= 100e-6 # in A
currentStep= 50e-6
#%%
sm.apply_current()                # Sets up to source current
sm.source_current_range = 1e-3   # Sets the source current range to 1 mA
sm.compliance_voltage = 10        # Sets the compliance voltage to 10 V
sm.source_current = 0             # Sets the source current to 0 mA
sm.enable_source()                # Enables the source output

# sm.ramp_to_current(50e-6) 
# sm.measure_current()
# print(f"{sm.current:.12f}")

for i in np.arange(currentStart, currentStop, currentStep):
    sm.ramp_to_current(i)
    sm.measure_current()
    print(f"{sm.current:.12f}")
    time.sleep(0.1)
    
# time.sleep(1)
sm.shutdown()                     # Ramps the current to 0 mA and disables output



#%%
# #%% some tests from initial learning
# sm.apply_current()                # Sets up to source current
# sm.source_current_range = 1e-3   # Sets the source current range to 1 mA
# sm.compliance_voltage = 1        # Sets the compliance voltage to 10 V
# sm.source_current = 0             # Sets the source current to 0 mA
# sm.enable_source()                # Enables the source output

# #%%
# sm.ramp_to_current(500e-9)          # Ramps the current to 5 mA
# sm.measure_voltage()              # Sets up to measure voltage

# print(sm.voltage)                 # Prints the voltage in Volts
# #%%

# #%%
# sm.shutdown()                     # Ramps the current to 0 mA and disables output

# # %%
# sm.reset()

# # %%


#%%




if __name__ == "__main__":

    app = QApplication([])
    window = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(QPushButton('Top'))
    layout.addWidget(QPushButton('Bottom'))
    window.setLayout(layout)
    window.show()
    app.exec_()

# %%
