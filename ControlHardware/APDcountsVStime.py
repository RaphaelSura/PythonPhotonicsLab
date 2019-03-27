import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from LowLevelModules.GeneralFunctions import *
from LowLevelModules.NIdaqAPD import APDCounter, ExtClock
import time
"""#######################################   USER INPUT   #################################################"""
terminal = '/Dev1/PFI1'
frequency = 1000    # Hz
rep_time = 60   # To have the plot refreshed every so often
save_data = False
"""#########################################################################################################"""

delta_t = 1 / frequency
#lp = LivePlot(1, 8, 5, 'o', 'Time (s)', 'APD counts (kHz)')
# CREATE EXT CLOCK TO GATE THE READING OF PULSES COMING FROM THE APD
ext_clock_task = ExtClock(frequency, task_name='Clock count rate')
# CREATE THE APD COUNTER
APD1 = APDCounter(terminal, task_name='Count rate')
APD1.sync_to_ext_clock()
APD1.start()

old_cts_cum = 0
t, cts = [], []
i = 0
t_start = time.time()
while True:

    try:
        # for j in range(rep_time*frequency):

        cts_cum = APD1.read() / 1000
        current_cts = (cts_cum - old_cts_cum) * frequency

        # append new data and plot
        #t.append(i)
        #cts.append(current_cts)
        #lp.plot_live(t, cts)
        print(i, time.time()-t_start, current_cts)
        # store old data and go to next loop iteration
        old_cts_cum = cts_cum
        i += 1
        # APD1.close()
        # ext_clock_task.close()
        # plt.close(lp.fig)
    except KeyboardInterrupt:
        # press the stop button to trigger this
        APD1.close()
        ext_clock_task.close()
        break

if save_data:
    data_type = 'PLvsTime'
    data_header = "Time (s)     APD counts (kHz)"
    data_array = np.array([t, cts]).T
    data_save(data_array, lp.fig, data_type, data_header)