from LowLevelModules.NIdaqAPD import APDCounter
from LowLevelModules.GeneralFunctions import *
import matplotlib.pyplot as plt
import numpy as np
import time
import nidaqmx


"""#######################################   USER INPUT   #################################################"""
scan_terminal = '/Dev1/ao2'
terminal = "/Dev1/PFI1"
save_data = True
collection_time = .25  #seconds
V_start = 0
V_end = 2
num_steps = 50
samp_per_reading = 20
"""#########################################################################################################"""
# initialize AO
offset_voltage = 0
task_toptica = nidaqmx.Task("Toptica scan")
output_volt = task_toptica.ao_channels.add_ao_voltage_chan(scan_terminal, 'Piezo scan', min_val=-4.5, max_val=4.5)
task_toptica.write(offset_voltage, auto_start=True, timeout=5)

# initialize the rest
voltages = np.linspace(V_start, V_end, num_steps)
volt, cts, laser_power = [], [], []
lp = LivePlot(10, 6, 'o', 'Voltage (V)', 'APD counts (kHz)')

#initialize the task on DAQ board
task_PD = nidaqmx.Task("PD reading")
task_PD.ai_channels.add_ai_voltage_chan("Dev1/ai2")
task_PD.start()

for v in voltages:
    # change RF value
    task_toptica.write(v, auto_start=True, timeout=5)
    time.sleep(.1)

    #read laser power
    curr_v = np.zeros(samp_per_reading)
    for i in range(samp_per_reading):
        curr_v[i] = task_PD.read()
        time.sleep(.005)
    power_now = np.mean(curr_v)
    time.sleep(.1)

    # collect the APD count rate
    APD1 = APDCounter(terminal)
    APD1.start()
    time.sleep(collection_time)
    APD_cts = APD1.read() / collection_time / 1000
    APD1.close()

    # get the new data and update the plot
    volt.append(v)
    cts.append(APD_cts)
    laser_power.append(power_now)
    lp.plot_live(volt, cts)

task_toptica.write(0)
task_toptica.close()
plt.show()

if save_data:
    data_type = 'PLEscan'
    data_header = "Voltage (V)     APD counts (kHz)     Laser power (V)"
    data_array = np.array([volt, cts, laser_power]).T
    data_save(data_array, lp.fig, data_type, data_header)
