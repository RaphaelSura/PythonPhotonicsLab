import sys
sys.path.append("..")
import nidaqmx
import time
import numpy as np
from tkinter import Tk, Label, StringVar
app_font = ('Latex', 60)
samp_per_reading = 20

"""  calibration values (532nm), power read at the output of microscope objective
     ND 1 in front of photo diode, , cannot read higher than 3 uW  """

power_reading = [0.1, 14.3, 29.5, 47.5, 228, 497, 1005, 1960, 2460, 3340]
daq_reading = [0.019, 0.065, 0.11, 0.155, 0.67, 1.44, 2.89, 5.67, 6.98, 9.53]


# Alternative calibration
"""
power_reading = [10, 96, 1170, 14700, 4500, 5570, 7360]
daq_reading = [0.018, .022, .089, .939, .299, .366, .477]
"""


class PowerMeterApp:

    def __init__(self, name, coeffs):
        self.win = Tk()
        self.win.resizable(1, 1)
        self.win.title(name)
        self.calib_coeffs = coeffs

        self.v = StringVar()
        self.v.set("0.00")
        self.power_label = Label(self.win, textvariable=self.v, width=8, font=app_font)
        self.power_label.grid(row=0, column=0, padx=10, pady=10)

        self.nw_label = Label(self.win, text=" nW ", font=app_font)
        self.nw_label.grid(row=0, column=1, padx=10, pady=10)

        # Display DAQ reading
        self.v2 = StringVar()
        self.v2.set("0.00")
        """
        self.nw_label = Label(self.win, textvariable=self.v2, width=8, font=app_font)
        self.nw_label.grid(row=1, column=0, padx=10, pady=10)
        """

        #initialize the task on DAQ board
        self.task = nidaqmx.Task("PD reading")
        self.task.ai_channels.add_ai_voltage_chan("Dev1/ai2")
        self.task.start()

        while True:
            v_now = self.read_voltage()
            current_power = self.volt_to_power(np.mean(v_now))
            self.v.set(str(np.round(current_power, 3)))
            self.v2.set(str(np.round(v_now, 3)))
            self.win.update()

    def read_voltage(self):
        curr_v = np.zeros(samp_per_reading)
        for i in range(samp_per_reading):
            curr_v[i] = self.task.read()
            time.sleep(.005)
        return np.mean(curr_v)

    def volt_to_power(self, v):
        return self.calib_coeffs[0] * v + self.calib_coeffs[1]

    def zeroing(self):
        v_now = self.read_voltage()


calib = np.polyfit(daq_reading, power_reading, 1)
app = PowerMeterApp("Power meter", calib)
app.win.mainloop()
