'''
jl3562@cornell.edu
This requires another module to pass the parameters
'''
import sys

sys.path.append("Z:\\PythonPhotonicsLab\\")
import numpy as np

from pymeasure.instruments.keithley import Keithley2400
from ControlHardware.ThorlabsPowermeterPM100D import ThorlabPM100D
from TestPrograms.ExpCtrlGUIWindow import CtrlVars
import TestPrograms.MockDataGenerator as MDG
import threading
import time

from LowLevelModules.Spectroscopy import Spectrum
from LowLevelModules.GeneralFunctions import LivePlot2D, prettify_2d_plot
from LowLevelModules.LightField import LightField


class Daq:
    '''
    I intend to make this an abstract class like the interface in Java, but I am not sure how to do so in python ...
    '''

    def __init__(self):
        self.xa = []
        self.ya = []
        self.newDataAcquired = threading.Event()
        self.newDataAcquired.clear()
        self.params = {}

        self.data = None  # TODO find a better solution: ?? currently assume 2 column data

        self.CtrlGUI = None
        self.expThread = None
        self.expIsRunningThrdEvt = threading.Event()  # false if experiment is not running
        self.expIsRunningThrdEvt.clear()
        self.expFinished = threading.Event()
        self.expFinished.clear()

    def parameterInit(self, params):
        '''
        pass the entire parameters from some interface to Daq module
        '''
        self.params = params

    def instrumentInit(self):
        pass

    def runExperiment(self):
        pass

    def linkCtrlGUI(self, gui):
        self.CtrlGUI = gui

    def startExpThread(self):
        print(f"Debug: starting experiment thread")
        self.expFinished.clear()

        self.expThread = threading.Thread(name='daq thread',
                                          target=self.runExperiment,
                                          args=tuple(),
                                          daemon=True)
        self.expThread.start()
        self.expIsRunningThrdEvt.set()  # TODO
        print(f'DEBUG: self.expIsRunningThrdEvt is not being clear() '
              f'anywhere, has not been implemented.')

    def disableSources(self):
        """
        Call this before exiting the program
        :return:
        """
        pass


class PowerCurrentDaq(Daq):
    def __init__(self):
        super().__init__()
        self.thorlabPM100DVName = 'USB0::0x1313::0x8078::P0021814::INSTR'
        self.keithley2400VName = 'GPIB0::24::INSTR'
        # self.instrumentInit()
        self.currentStart = 0
        self.currentStop = 0
        self.currentStep = 1
        self.powerCurrentDat = None
        self.currentMeasurements = np.zeros(1)

    def parameterInit(self, params):
        '''
        pass the entire parameters from some interface to Daq module
        '''
        self.params = params
        self.instrumentInit()

    def disableSources(self):
        if self.sm != None:
            self.sm.disable_source()

    def instrumentInit(self):
        if len(self.params) == 0:
            print(f'Empty params: {self.params}')
            return
        self.sm = Keithley2400(self.keithley2400VName)  # source meter
        self.pm = ThorlabPM100D(self.thorlabPM100DVName)  # power meter

        print(f'connected to instruments...')

        print(f'{self.params}')
        # return
        self.currentStart = 1e-3 * self.params[CtrlVars.START_CURR.value]
        self.currentStop = 1e-3 * self.params[CtrlVars.STOP_CURR.value]  # in A
        self.currentStep = 1e-3 * self.params[CtrlVars.CURR_STEP.value]
        # return
        self.sm.apply_current()  # Sets up to source current
        self.sm.source_current_range = 1e-3 * self.params[CtrlVars.SOURCE_CURRENT_RANGE.value]
        self.sm.compliance_voltage = self.params[CtrlVars.COMPLIANCE_VOLT.value]  # Sets the compliance voltage to 10 V

        ''' Safety precaution '''
        self.sm.source_current = 0  # Sets the source current to 0 mA
        self.sm.enable_source()  # Enables the source output

        print(f'Parameters set, source enabled.')

    def runExperiment(self):
        # allocate np array for storing data
        self.currents = np.arange(self.currentStart, self.currentStop, self.currentStep)
        self.NData = self.currents.size
        self.powers = np.zeros(self.NData)
        self.powersStdDev = np.zeros(self.NData)
        self.currentMeasurements = np.zeros(self.NData)
        # allocate np array for currents that will be swept;

        self.xa = self.currents
        self.ya = self.powers

        haltTimeCurrRamp2PowerMeas = 0.5

        print(f'starting to acquire data')
        for i, cur in enumerate(self.currents):
            self.sm.ramp_to_current(cur)
            self.sm.measure_current()  # run this before reading with accessing field sm.current
            self.currentMeasurements[i] = self.sm.current
            time.sleep(haltTimeCurrRamp2PowerMeas)

            pmDatum = self.pm.measure(n=3, wavelength=self.params[CtrlVars.WAVELENGTH.value], \
                                      beamDiameter=self.params[CtrlVars.BEAM_DIAMETER.value])
            self.powers[i] = pmDatum[1]
            self.powersStdDev[i] = pmDatum[2]

            self.newDataAcquired.set()  # for notifying some drawing component
            # time.sleep(0.025)

        self.sm.shutdown()  # Ramps the current to 0 mA and disables output
        self.data = np.column_stack((self.currents, self.powers))
        self.expFinished.set()
        return self.data


class LaserDiodeSpectroscopyDaq(Daq):
    def __init__(self):
        super().__init__()
        self.keithley2400VName = 'GPIB0::24::INSTR'
        # TODO change this back to correct code on the lightfield computer
        print(f'Daq: Initializing lightfield program')
        self.LFauto = LightField()
        # self.LFauto = None
        self.lightfieldAcquiringThrdEvt = threading.Event()
        self.lightfieldAcquiringThrdEvt.clear()

        self.wdir = f'Z:\\Projects\\Boron Nitride\\samples\\2019HenrykDiodeLaser\\tmp'
        self.currentStart = 0
        self.currentStop = 1
        self.currentStep = 1
        self.currents = np.arange(self.currentStart, self.currentStop, self.currentStep)
        self.NData = 1
        self.powers = np.zeros(self.NData)
        self.powersStdDev = np.zeros(self.NData)

    def parameterInit(self, params):
        self.params = params
        self.instrumentInit()

    def instrumentInit(self):
        if len(self.params) == 0:
            print(f'Empty params: {self.params}')
            return

        self.sm = Keithley2400(self.keithley2400VName)  # source meter

        print(f'configuring LightField...')
        self.LFauto.set_acquisition_time(self.params[CtrlVars.SPECTROMETER_ACQ_TIME.value])
        self.LFauto.set_path(self.wdir)  # TODO: move wdir to GUI
        self.LFauto.set_filename("TestSpectrum")
        self.LFauto.set_filename_increment()

        print(f'configuring Keithley...')

        self.currentStart = 1e-3 * self.params[CtrlVars.START_CURR.value]
        self.currentStop = 1e-3 * self.params[CtrlVars.STOP_CURR.value]  # in A
        self.currentStep = 1e-3 * self.params[CtrlVars.CURR_STEP.value]

        self.sm.apply_current()  # Sets up to source current
        self.sm.source_current_range = 1e-3 * self.params[
            CtrlVars.SOURCE_CURRENT_RANGE.value]  # Sets the source current range to 1 mA
        self.sm.compliance_voltage = self.params[CtrlVars.COMPLIANCE_VOLT.value]  # Sets the compliance voltage to 10 V

        ''' Safety precaution '''
        self.sm.source_current = 0  # Sets the source current to 0 mA
        self.sm.enable_source()  # Enables the source output

        print(f'Parameters set, current source enabled.')

    def runExperiment(self):
        # allocate np array for storing data
        self.newDataAcquired.clear()
        self.currents = np.arange(self.currentStart, self.currentStop, self.currentStep)
        self.NData = self.currents.size
        self.powers = np.zeros(self.NData)
        self.powersStdDev = np.zeros(self.NData)
        # allocate np array for currents that will be swept;

        self.xa = self.currents
        self.ya = self.powers

        stabilizingTime = 1
        haltTimeCurrRampBeforeSpectroscopy = self.params[CtrlVars.SPECTROMETER_ACQ_TIME.value] + stabilizingTime

        print(f'starting to acquire data, estimating to take {haltTimeCurrRampBeforeSpectroscopy} seconds')
        for i, cur in enumerate(self.currents):
            self.sm.ramp_to_current(cur)
            self.sm.measure_current()  # run this before reading with accessing field sm.current
            time.sleep(haltTimeCurrRampBeforeSpectroscopy)

            self.LFauto.acquire()
            # self.lightfieldAcquiringThrdEvt.wait()
            self.newDataAcquired.set()  # for notifying some drawing component
            print(f'DEBUG: Finished acquiring a spectrum')
            # time.sleep(0.025)

        self.sm.shutdown()  # Ramps the current to 0 mA and disables output

        return

        # def startLightFieldAcquireThread(self):
    #     threading.Thread(target=self.LFauto.acquire).start()  # TODO finish the spectrometer control here
    #     self.lightfieldAcquiringThrdEvt.set()


class MockExperiment(Daq):
    """
    Use this for testing other components...
    """

    def parameterInit(self, params):
        """
        pass the entire parameters from some interface to Daq module
        """
        self.params = params

    def instrumentInit(self):
        pass

    def __init__(self, bufferSize=10):
        super().__init__()
        self.x = np.zeros(bufferSize)
        self.y = np.zeros(bufferSize)
        self.xa = []
        self.ya = []
        self.cur = 0
        self.bufferSize = bufferSize
        self.mockDataGen = MDG.MockDataGenerator()
        self.tWaitGen = MDG.MockDataGenerator()

        self.data = None
        self.expThread = None

        self.newDataAcquired = threading.Event()
        self.newDataAcquired.clear()
        self.expFinished = threading.Event()
        self.expFinished.clear()
        pass

    def acquireNext(self):
        x0 = self.cur % self.bufferSize
        self.x[x0] = x0
        newY = self.mockDataGen.generateNoisedLinearPair(x0, sigma=1)[1]
        self.y[x0] = newY

        self.ya.append(self.mockDataGen.generateNoisedLinearPair(self.cur, sigma=1)[1])
        self.xa.append(self.cur)

        self.cur = self.cur + 1

    def startExpThread(self):
        print(f"Debug: starting experiment thread")
        self.expThread = threading.Thread(name='daq thread',
                                          target=self.runExperiment,
                                          args=tuple(),
                                          daemon=True)
        self.expThread.start()

    def runExperiment(self, intervalMean=0.1, nData=5):
        print(f"run one experiment with acquisition interval mean of {intervalMean} second; collecting {nData} points")
        for i in range(nData):
            tWait = self.tWaitGen.generateNoisedLinearPair(intervalMean, sigma=2)[1]
            while tWait < 0:
                tWait = self.tWaitGen.generateNoisedLinearPair(intervalMean, sigma=2)[1]
            print(f"Daq: acquiring {i}-th number")
            self.acquireNext()
            self.newDataAcquired.set()
            time.sleep(tWait)
            print(f'waited for {tWait} seconds')
        self.data = np.column_stack((self.xa, self.ya))
        self.expFinished.set()
        print(f"experiment finished")
        return self.data
