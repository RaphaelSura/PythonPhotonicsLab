import nidaqmx
import time


class APDCounter(nidaqmx.Task):

    def __init__(self, term, ctr=1, task_name="Count pulses"):
        nidaqmx.Task.__init__(self, task_name)
        # create counter channel and assign terminal
        self.counter = self.ci_channels.add_ci_count_edges_chan("/Dev1/ctr" + str(ctr), "APD", nidaqmx.constants.Edge.RISING, 0,
                                                                nidaqmx.constants.CountDirection.COUNT_UP)
        self.counter.ci_count_edges_term = term

    def read_counts(self, integration_time):
        time.sleep(integration_time)
        return self.counter.ci_count() / integration_time

    def sync_to_ext_clock(self):
        self.timing.samp_clk_src = '/Dev1/PFI12'  # 'ctr0 out is on PFI12 --> see device pin configuration in MAX
        self.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.SAMPLE_CLOCK
        self.timing.samp_quant_samp_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
        self.timing.samp_quant_samp_per_channel = 1000
        self.timing.samp_clk_rate = 1000
        self.timing.samp_clk_active_edge = nidaqmx.constants.Edge.RISING


class ExtClock(nidaqmx.Task):

    def __init__(self, freq):
        nidaqmx.Task.__init__(self, "External sample clock")
        self.pulse_train = self.co_channels.add_co_pulse_chan_freq("/Dev1/ctr0", 'pulse_train',
                                                                   nidaqmx.constants.FrequencyUnits.HZ,
                                                                   nidaqmx.constants.Level.LOW, initial_delay=0.0,
                                                                   freq=freq, duty_cycle=0.5)
        # define the external clock settings
        self.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.IMPLICIT
        self.timing.samp_quant_samp_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
        self.timing.samp_quant_samp_per_channel = 1000
        self.start()
