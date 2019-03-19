from LowLevelModules.NIdaqAPD import APDCounter
from LowLevelModules.SignalGenerator import SG384
from LowLevelModules.GeneralFunctions import *
import matplotlib.pyplot as plt
import numpy as np
import time

"""#######################################   USER INPUT   #################################################"""
device_address = 'GPIB0::28::INSTR'
terminal = "/Dev1/PFI1"
save_data = True
collection_time = .25  #seconds
RF_freq_start = 2.85-.5 #GHz
RF_freq_end = 2.85+.5
amplitude = 16   #dBm
num_steps = 100
"""#########################################################################################################"""

# initialize the RF generator
rf_source = SG384(device_address)
rf_source.set_frequency(RF_freq_start, 'GHz')
rf_source.set_amplitude(amplitude, 'dBm')
rf_source.enable_RF_signal()

# initialize the rest
RF_freq = np.linspace(RF_freq_start, RF_freq_end, num_steps)
freq, cts = [], []
lp = LivePlot(1, 10, 6, 'o', 'Frequency (GHz)', 'APD counts (kHz)')

for f in RF_freq:
    # change RF value
    rf_source.set_frequency(f, 'GHz')

    # collect the APD count rate
    APD1 = APDCounter(terminal)
    APD1.start()
    time.sleep(collection_time)
    APD_cts = APD1.read() / collection_time / 1000
    APD1.close()

    # get the new data and update the plot
    freq.append(f)
    cts.append(APD_cts)
    lp.plot_live(freq, cts)

rf_source.disable_RF_signal()
plt.show()

if save_data:
    data_type = 'ESRscan'
    data_header = "Frequency (GHz)     APD counts (kHz)"
    data_array = np.array([freq, cts]).T
    data_save(data_array, lp.fig, data_type, data_header)
