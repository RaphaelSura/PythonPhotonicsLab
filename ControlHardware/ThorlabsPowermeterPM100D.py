# %%
# JL3562@cornell.edu
# created: 2020/01/21
#
import pyvisa
import time
from collections import deque
import sys
import numpy as np

sys.path.append("..")


class ThorlabPM100D:
    '''
    A class wrapper using PyVisa to communicate with PM100D power meter
    '''

    def __init__(self, VISAName):
        self.name = VISAName
        self.pm = None
        rm = pyvisa.ResourceManager()
        self.powerMeasurements = np.zeros(10)
        try:
            self.pm = rm.open_resource(self.name)
        except:
            print("ThorlabPM100D: Something is wrong with initialization, let's abort the program...")
            sys.exit(2)

    # measures the power

    def measure(self, n=10, wavelength=535, beamDiameter=3):
        '''
        Measures power at lambda [nm] for a beamDiameter [mm] sized beam
        return (self.powerMeasurement, powerAve, powerStdDev)
        '''
        self.pm.write(f"sense:correction:wavelength {wavelength:.0f}")
        self.pm.write(f"sense:correction:beamdiameter {beamDiameter:.0f}")
        instrWavelength = float(self.pm.query(f"sense:correction:wavelength?"))
        instrBeamDiameter = float(self.pm.query(f"sense:correction:beamdiameter?"))
        # print(f"DEBUG: Measuring power at {instrWavelength:.0f} [nm]; beam diameter is {instrBeamDiameter:.0f} [mm]")

        self.powerMeasurements = np.zeros(n)
        for i in range(0, n, 1):
            try:
                # print(f"Making {i}-th measurement")
                self.pm.write('initiate')
                self.pm.write('measure:power')
                self.powerMeasurements[i] = float(self.pm.query('fetch?'))
                time.sleep(0.005)
            except KeyboardInterrupt:
                break

        powerAve = np.average(self.powerMeasurements)
        powerStdDev = np.std(self.powerMeasurements)
        return self.powerMeasurements, powerAve, powerStdDev

# # some tests down here
# thorlabPM100DVName= 'USB0::0x1313::0x8078::P0021814::INSTR'

# pm = ThorlabPM100D(thorlabPM100DVName)
# testResult = pm.measure()
# print(type(testResult))
# print(type(testResult[0]))
# print(testResult[0])
# print(testResult[1])
# print(testResult[2])

# #%%
# test=np.array([1,2,3,4,5])
# test[0]
# for i in range(0, 5, 1):
#     print(test[i])


# # %%
# rm = pyvisa.ResourceManager()

# rm.list_resources()


# # %%
# type(rm.list_resources())


# # %%
# pm= rm.open_resource(thorlabPM100DVName)
# print(pm.query('*IDN?'))
# print(type(pm.query('*IDN?')))


# # %%
# print(pm.query('system:sensor:idn?'))


# # %%
# print(pm.query('calibration:string?'))


# # %%
# print(pm.query('status:MEAS:COND?'))


# # %%
# NAvg= 10 # each sample takes ~3ms
# cmd= f"sense:average {NAvg}"
# print(cmd)
# print(pm.write(cmd))
# print(pm.query('sense:average:count?'))


# # %%
# beamDiameterMM= float(pm.query('sense:correction:beamdiameter?'))
# print(f"beamDiameter= {beamDiameterMM:.0f} [mm]")
# wavelengthNM= float(pm.query('sense:correction:wavelength?'))
# print(f"wavelength= {wavelengthNM:.0f} [nm]")


# # %%
# powerResponse= float(pm.query('sense:correction:power?'))
# print(f"powerResponse= {powerResponse} [A/W]")


# # %%
# print(f"Measurement configuration: {pm.query('configure?')}")


# # %%
# # print(pm.write('initiate'))
# # print(pm.write('measure:power'))
# # print(pm.query('fetch?'))
# tmpData= deque()
# for i in range(0,100,1):
#     try:
#         pm.write('initiate')
#         pm.write('measure:power')
#         tmpData.append(float(pm.query('fetch?')))
#         time.sleep(0.005)
#     except KeyboardInterrupt:
#         break
# for i in range(0, 100, 1):
#     print(f"{tmpData.pop()*10**9:.0f}")


# # %%
# for i in range(0,100,1):
#     print(i)
# # %% For live plotting


# # %%
