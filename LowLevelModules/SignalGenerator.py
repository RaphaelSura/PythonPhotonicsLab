import visa


class SG384:
    def __init__(self, addr):
        rm_py = visa.ResourceManager()
        self.equipment = rm_py.open_resource(addr)

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
    
    def set_DC_offset(self, value):
        self.equipment.write(f"OFSL {value}")
    
    def get_DC_offset(self):
        return float(self.equipment.query("OFSL?").strip())

    def enable_RF_signal(self, port='BNC'):
        if port == 'BNC':
            self.equipment.write("ENBL 1")
        else:
            self.equipment.write("ENBR 1")

    def disable_RF_signal(self, port='BNC'):
        if port == 'BNC':
            self.equipment.write("ENBL 0")
        else:
            self.equipment.write("ENBR 0")
        
    def read_error(self):
        return self.equipment.query("LERR?").strip()
    
    def get_mod_func(self):
        return self.equipment.query("MFNC?").strip()

