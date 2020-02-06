import sys

sys.path.append("..")
import nidaqmx
import time
import numpy as np
import matplotlib.pyplot as plt

app_font = ('Latex', 60)
samp_per_reading = 100

from pymeasure.instruments.keithley import Keithley2400

keithley = Keithley2400("GPIB0::24::INSTR")  # 'ASRL9::INSTR' if RS232 connection
keithley.apply_voltage()  # Sets up to source voltage
keithley.source_voltage_range = 40  # Sets the source voltage range to 30 V
keithley.compliance_current = 1e-3  # Sets the compliance current to 1 mA
keithley.source_voltage = 0  # Sets the source voltage to 0 V
keithley.measure_current()  # Sets up to measure current

keithley.enable_source()

dcSweep = np.arange(0, 40, 1)
PDReading = []

print(dcSweep)
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai4")

    for dcVal in dcSweep:
        keithley.ramp_to_voltage(dcVal)
        time.sleep(1)
        curr_v = np.zeros(samp_per_reading)
        for i in range(samp_per_reading):
            curr_v[i] = task.read() * 1000 + 13
            time.sleep(.005)
        PDReading.append(np.mean(curr_v))
        time.sleep(.005)

keithley.ramp_to_voltage(0)
keithley.disable_source()

plt.figure(1)
plt.plot(dcSweep, PDReading)
plt.savefig("DCSweep.png", format='png')
plt.show()
