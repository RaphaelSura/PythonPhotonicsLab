# Generic imports
import clr
import sys
import os
import time
from System.IO import *
from System import String
from System.Collections.Generic import List
from LowLevelModules.Spectroscopy import Spectrum

# clr.AddReference('System.IO')
# clr.AddReference('System.Collections')

# Needed dll for interaction with LF
sys.path.append(os.environ['LIGHTFIELD_ROOT'])
sys.path.append(os.environ['LIGHTFIELD_ROOT'] + '\\AddInViews')
clr.AddReference('PrincetonInstruments.LightFieldViewV5')
clr.AddReference('PrincetonInstruments.LightField.AutomationV5')
clr.AddReference('PrincetonInstruments.LightFieldAddInSupportServices')

# PI imports
# clr.AddReference('PrincetonInstruments.LightField')
# clr.AddReference('PrincetonInstruments.LightField.Automation')
import PrincetonInstruments.LightField.AddIns as AddIns
from PrincetonInstruments.LightField.Automation import Automation
from PrincetonInstruments.LightField.AddIns import CameraSettings
from PrincetonInstruments.LightField.AddIns import DeviceType
from PrincetonInstruments.LightField.AddIns import ExperimentSettings


class LightField:
    def __init__(self):
        self.auto = Automation(True, List[String]())
        self.experiment = self.auto.LightFieldApplication.Experiment
        for device in self.experiment.ExperimentDevices:
            if device.Type == DeviceType.Camera:
                print("Lightfield startup and setup OK")

    def set_value(self, setting, value):
        if self.experiment.Exists(setting):
            self.experiment.SetValue(setting, value)

    def set_acquisition_time(self, acq_t):
        self.set_value(CameraSettings.ShutterTimingExposureTime, acq_t * 1000)  # ms

    def set_path(self, directory):
        self.set_value(ExperimentSettings.FileNameGenerationDirectory, directory)

    def set_filename(self, name):
        self.set_value(ExperimentSettings.FileNameGenerationBaseFileName, name)

    def set_filename_increment(self, num_incr=False, date_incr=False, time_incr=False):
        self.set_value(ExperimentSettings.FileNameGenerationAttachIncrement, num_incr)
        self.set_value(ExperimentSettings.FileNameGenerationAttachDate, date_incr)
        self.set_value(ExperimentSettings.FileNameGenerationAttachTime, time_incr)

    def acquire(self):
        while not self.experiment.IsReadyToRun:
            time.sleep(.1)
        self.experiment.Acquire()
        while self.experiment.IsRunning:
            time.sleep(.1)

    @staticmethod
    def load_acquired_data(directory, filename):
        while not os.path.exists(directory + "\\" + filename + ".spe"):
            time.sleep(.1)
        while True:
            try:
                return Spectrum(directory + "\\" + filename + ".spe")
            except PermissionError:
                print("Not permitted??")
                time.sleep(.1)