import visa

class LakeShore335:

    """ Temperature controller from Lake Shore 335
        Connection via GPIB """

    def __init__(self, addr):
        self.equipment = visa.ResourceManager().open_resource(addr)
        self.heater_range_values = {'off': 0, 'low': 1, 'medium': 2, 'high': 3}

    def set_target_temperature(self, value):
        self.equipment.write(f"SETP 1, {value}")

    def get_temp(self, sensor):
        temp = self.equipment.query(f"KRDG? {sensor}")
        return float(temp[1:].strip())

    def set_heater_range(self, value):
        self.equipment.write(f"RANGE 1, {self.heater_range_values[value]}")

    def turn_heater_off(self):
        self.equipment.write("RANGE 1, 0")


class GaussMeter475:

    """ Gauss Meter from Lake Shore 475 DSP
        Connection via GPIB """

    def __init__(self, addr):
        self.equipment = visa.ResourceManager().open_resource(addr)

    def get_field(self):
        field = self.equipment.query("RDGFIELD?")
        return float(field.strip())

    def auto_range(self):
        """ Sets the field range to automatically adjust """
        self.equipment.write("AUTO")

    def zero_probe(self):
        """ Initiates the zero field sequence to calibrate the probe """
        self.equipment.write("ZPROBE")


class SG384:
    """ Signal generator SG384 from Stanford Research system
        Connection via GPIB """

    def __init__(self, addr):
        self.equipment = visa.ResourceManager().open_resource(addr)

    def set_frequency(self, value, unit='GHz'):
        self.equipment.write(f"FREQ {value} {unit}")

    def get_frequency(self, unit='GHz'):
        return self.equipment.query(f"FREQ? {unit}")

    def set_amplitude(self, value, unit='dBm', port='BNC'):
        if port == 'BNC':
            self.equipment.write(f"AMPL {value} {unit}")
        else:
            self.equipment.write("AMPR {value} {unit}")

    def get_amplitude(self, unit='dBm', port='BNC'):
        if port == 'BNC':
            return self.equipment.query(f"AMPL? {unit}")
        else:
            return self.equipment.query(f"AMPR? {unit}")

    def set_dc_offset(self, value):
        self.equipment.write(f"OFSL {value}")

    def get_dc_offset(self):
        return float(self.equipment.query("OFSL?").strip())

    def enable_rf_signal(self, port='BNC'):
        if port == 'BNC':
            self.equipment.write("ENBL 1")
        else:
            self.equipment.write("ENBR 1")

    def disable_rf_signal(self, port='BNC'):
        if port == 'BNC':
            self.equipment.write("ENBL 0")
        else:
            self.equipment.write("ENBR 0")

    def read_error(self):
        return self.equipment.query("LERR?").strip()

    def get_mod_func(self):
        return self.equipment.query("MFNC?").strip()

class DG645:
    """ Digital Delay Generator DG645 from Stanford Research system
        Connection via GPIB """

    def __init__(self, addr):
        self.equipment = visa.ResourceManager().open_resource(addr)
    
    def close(self):
        self.equipment.close()
        
    def read_error(self):
        return self.equipment.query("LERR?").strip()
    
    def set_polarity(self,b,i):
        self.equipment.write(f"LPOL {b},{i}")

    def get_polarity(self,b):
        return self.equipment.query(f"LPOL? {b}").strip()
    
    def get_timebase(self):
        return self.equipment.query(f"TIMB?").strip()
    
    def set_trigger_source(self,i):
        self.equipment.write(f"TSRC {i}")

    def get_trigger_source(self):
        return self.equipment.query(f"TSRC?")
    
    def set_holdoff(self,t):
        self.equipment.write(f"HOLD {t}")

    def get_holdoff(self):
        return self.equipment.query(f"HOLD?")
        
    def set_trigger_level(self,v):
        self.equipment.write(f"TLVL {v}")

    def get_trigger_level(self):
        return self.equipment.query(f"TLVL?")

    def set_trigger_rate(self,f):
        self.equipment.write(f"TRAT {f}")

    def get_trigger_rate(self):
        return self.equipment.query(f"TRAT?")

    def set_burst_count(self,i):
        self.equipment.write(f"BURC {i}")

    def get_burst_count(self):
        return self.equipment.query(f"BURC?")

    def set_burst_delay(self,t):
        self.equipment.write(f"BURD {t}")

    def get_burst_delay(self):
        return self.equipment.query(f"BURD?")

    def set_burst_mode(self,i):
        self.equipment.write(f"BURM {i}")

    def get_burst_mode(self):
        return self.equipment.query(f"BURM?")

    def set_burst_period(self,t):
        self.equipment.write(f"BURP {t}")

    def get_burst_period(self):
        return self.equipment.query(f"BURP?")

    def set_burst_config(self,i):
        self.equipment.write(f"BURT {i}")

    def get_burst_config(self):
        return self.equipment.query(f"BURT?")
    
    def set_delay(self,c,d,t):
        self.equipment.write(f"DLAY {c},{d},{t}")
        
    def get_delay(self,c):
        return self.equipment.query(f"DLAY? {c}")
    
    def set_amplitude(self,b,v):
        self.equipment.write(f"LAMP {b},{v}")
        
    def get_amplitude(self,b):
        return self.equipment.query(f"LAMP? {b}")
    
    def set_offset(self,b,v):
        self.equipment.write(f"LOFF {b},{v}")
    
    def get_offset(self,b):
        return self.equipment.query(f"LOFF? {b}")