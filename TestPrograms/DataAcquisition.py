'''
jl3562@cornell.edu
This requires another module to pass the parameters
'''
import sys
# sys.path.append("Z:\\PythonPhotonicsLab\\")
import numpy as np

from pymeasure.instruments.keithley import Keithley2400
from ControlHardware.ThorlabsPowermeterPM100D import ThorlabPM100D
from TestPrograms.ExpCtrlGUIWindow import CtrlVars
import TestPrograms.MockDataGenerator as MDG
import threading
import time

class Daq:
    '''
    I intend to make this an abstract class like the interface in Java, but I am not sure how to do so in python ...
    '''
    def __init__(self):
        self.xa= []
        self.ya= []
        self.newDataAcquired= threading.Event()
        self.newDataAcquired.clear()
        self.params= {}

    def parameterInit(self, params):
        '''
        pass the entire parameters from some interface to Daq module
        '''
        self.params= params

    def instrumentInit(self):
        pass

    def runExperiment(self):
        pass


class PowerCurrentDaq(Daq):
    def __init__(self):
        super().__init__()
        self.thorlabPM100DVName= 'USB0::0x1313::0x8078::P0021814::INSTR'
        self.keithley2400VName= 'GPIB0::24::INSTR'

        self.instrumentInit()

    def instrumentInit(self):
        if len(self.params)== 0:
            print(f'Empty params: {self.params}')
            return
        self.sm = Keithley2400(self.keithley2400VName) # source meter
        self.pm = ThorlabPM100D(self.thorlabPM100DVName) # power meter

        print(f'connected to instruments...')

        self.currentStart=   1e-3 * self.params[CtrlVars.START_CURR]
        self.currentStop=    1e-3 * self.params[CtrlVars.STOP_CURR] # in A
        self.currentStep=    1e-3 * self.params[CtrlVars.CURR_STEP]

        self.sm.apply_current()                # Sets up to source current
        self.sm.source_current_range = 1e-3 * self.params[CtrlVars.SOURCE_CURRENT_RANGE]   # Sets the source current range to 1 mA
        self.sm.compliance_voltage = self.params[CtrlVars.COMPLIANCE_VOLT]        # Sets the compliance voltage to 10 V

        ''' Safety precaution '''
        self.sm.source_current = 0             # Sets the source current to 0 mA
        self.sm.enable_source()                # Enables the source output

        print(f'Parameters set, source enabled.')


    def runExperiment(self):
        #allocate np array for storing data
        self.currents= np.arange(self.currentStart, self.currentStop, self.currentStep)
        self.NData= current.size
        self.powers= np.zeros(self.NData)
        self.powersStdDev= np.zeros(self.NData)
        #allocate np array for currents that will be swept;

        self.xa= self.currents
        self.ya= self.powers

        haltTimeCurrRamp2PowerMeas=3

        print(f'starting to acquire data')
        for i, cur in enumerate(self.currents):
            self.sm.ramp_to_current(cur)
            self.sm.measure_current() # run this before reading with accessing field sm.current
            time.sleep(haltTimeCurrRamp2PowerMeas)

            pmDatum= self.pm.measure(n=3, wavelength=self.params[CtrlVars.WAVELENGTH],\
                beamDiameter=self.params[CtrlVars.BEAM_DIAMETER])
            self.powers[i]= pmDatum(1)
            self.powersStdDev[i]= pmDatum(2)

            self.newDataAcquired.set() # for notifying some drawing component
            # time.sleep(0.025)


        self.sm.shutdown()                     # Ramps the current to 0 mA and disables output
        self.PowerCurrentDat= np.column_stack((self.currents, self.powers))

        return self.PowerCurrentDat

class MockExperiment(Daq):
    def parameterInit(self, params):
        '''
        pass the entire parameters from some interface to Daq module
        '''
        self.params= params

    def instrumentInit(self):
        pass

    def __init__(self, bufferSize=10):
        super().__init__()
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
        self.expFinished= threading.Event()
        self.expFinished.clear()
        pass

    def acquireNext(self):
        x0= self.cur % self.bufferSize
        self.x[x0]= x0
        newY= self.mockDataGen.generateNoisedLinearPair(x0, sigma=1)[1]
        self.y[x0]= newY

        self.ya.append(self.mockDataGen.generateNoisedLinearPair(self.cur, sigma=1)[1])
        self.xa.append(self.cur)

        self.cur= self.cur+1

    def startExpThread(self):
        print(f"Debug: starting experiment thread")
        self.expThread= threading.Thread(name='daq thread',
                                         target= self.runExperiment,
                                         args=tuple(),
                                         daemon= True)
        self.expThread.start()

    def runExperiment(self, intervalMean= 0.1, nData= 5):
        print(f"run one experiment with acquisition interval mean of {intervalMean} second; collecting {nData} points")
        for i in range(nData):
            tWait= self.tWaitGen.generateNoisedLinearPair(intervalMean, sigma=2)[1]
            while tWait<0:
                tWait= self.tWaitGen.generateNoisedLinearPair(intervalMean, sigma=2)[1]
            print(f"Daq: acquiring {i}-th number")
            self.acquireNext()
            self.newDataAcquired.set()
            time.sleep(tWait)
            print(f'waited for {tWait} seconds')
        self.data= np.column_stack((self.xa, self.ya))
        self.expFinished.set()
        print(f"experiment finished")
        return self.data