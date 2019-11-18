import nidaqmx
import time

class AO(nidaqmx.Task):
    #, task_name="AO Scan1"
    def __init__(self, term):
        #nidaqmx.Task.__init__(self, task_name)
        nidaqmx.Task.__init__(self)
        self.ao_channels.add_ao_voltage_chan(term, min_val=-10, max_val=10)#, 'FSM x axis1'
        
    def config_write(self,varray,scan_freq,trig_src):
        npnts = len(varray)
        # use onboard clock 
        self.timing.cfg_samp_clk_timing(rate= scan_freq, 
                                        active_edge=nidaqmx.constants.Edge.RISING ,
                                        sample_mode= nidaqmx.constants.AcquisitionType.FINITE, 
                                        samps_per_chan=npnts)
        self.triggers.start_trigger.cfg_dig_edge_start_trig(trig_src,nidaqmx.constants.Edge.RISING)        
        self.write(varray, auto_start=False, timeout=50)
        self.start()

class AI(nidaqmx.Task):
    #, task_name="AI Read1"
    def __init__(self, term,terminal_config=nidaqmx.constants.TerminalConfiguration.DEFAULT):
        nidaqmx.Task.__init__(self)
        self.ai_channels.add_ai_voltage_chan(term,terminal_config=terminal_config)#, 'FSM x axis1'
        
    def config_read(self,npnts,scan_freq,trig_src):
        # use onboard clock
        self.timing.cfg_samp_clk_timing(rate= scan_freq, 
                                        active_edge=nidaqmx.constants.Edge.FALLING ,
                                        sample_mode= nidaqmx.constants.AcquisitionType.FINITE, 
                                        samps_per_chan=npnts)   
        self.triggers.start_trigger.cfg_dig_edge_start_trig(trig_src,nidaqmx.constants.Edge.FALLING  )
        self.start()
        
    def config_read_rising(self,npnts,scan_freq,trig_src):
        # use onboard clock
        self.timing.cfg_samp_clk_timing(rate= scan_freq, 
                                        active_edge=nidaqmx.constants.Edge.RISING ,
                                        sample_mode= nidaqmx.constants.AcquisitionType.FINITE, 
                                        samps_per_chan=npnts)   
        self.triggers.start_trigger.cfg_dig_edge_start_trig(trig_src,nidaqmx.constants.Edge.RISING  )
        self.start()

        
        
class DO(nidaqmx.Task):
    def __init__(self, line):
        nidaqmx.Task.__init__(self)
        self.do_channels.add_do_chan(line)

    # self.ai_channels.add_ai_voltage_chan(term, 'FSM x axis1')
#     def generate_dig(v):
#         with nidaqmx.Task() as task:
#             task.do_channels.add_do_chan('/Dev2/port0/line0')
#             task.write(v)
#             task.wait_until_done(timeout=5)
        

class CO(nidaqmx.Task):
    #, task_name='Counter out'
    def __init__(self, ctr, freq):
        nidaqmx.Task.__init__(self)
        self.pulse_train = self.co_channels.add_co_pulse_chan_freq(ctr,
                                                           units=nidaqmx.constants.FrequencyUnits.HZ,
                                                           idle_state=nidaqmx.constants.Level.LOW, initial_delay=0.0,
                                                           freq=freq, duty_cycle=0.5)
        # define the external clock settings      
        self.timing.cfg_implicit_timing(sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS, 
                                        samps_per_chan = 1000)
        self.start()
        
class CI(nidaqmx.Task):
    #, task_name="Count pulses"
    def __init__(self, term, counterPort):
        nidaqmx.Task.__init__(self)
        # create counter channel and assign terminal
        self.counter = self.ci_channels.add_ci_count_edges_chan(counterPort, edge=nidaqmx.constants.Edge.RISING, initial_count=0,
                                                                count_direction=nidaqmx.constants.CountDirection.COUNT_UP)
        self.counter.ci_count_edges_term = term 
        #self.counter.ci_dup_count_prevention = True # true by default
        
    def read_counts(self, integration_time=1):
        return self.counter.ci_count / integration_time

#     def sync_to_ext_clock(self,clk_src):
#         self.timing.samp_clk_src = clk_src  # 'ctr0 out is on PFI12 --> see device pin configuration in MAX
#         self.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.SAMPLE_CLOCK
#         self.timing.samp_quant_samp_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
#         #self.timing.samp_quant_samp_mode = nidaqmx.constants.AcquisitionType.FINITE
#         self.timing.samp_quant_samp_per_channel = 10000   # = buffer size if acquisition CONTINUOUS
#         self.timing.samp_clk_rate = 1000          # expected sample clock rate
#         self.timing.samp_clk_active_edge = nidaqmx.constants.Edge.RISING
        
    def config_read_samples(self,clk_src,npnts,clk_rate):
        
        self.timing.cfg_samp_clk_timing(rate = clk_rate,
                                        source = clk_src,
                                        active_edge = nidaqmx.constants.Edge.RISING ,
                                        sample_mode = nidaqmx.constants.AcquisitionType.FINITE, 
                                        samps_per_chan = npnts)           
        # 'ctr0 out is on PFI12 --> see device pin configuration in MAX
        self.start()
    
