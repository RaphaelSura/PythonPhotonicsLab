import visa


class SG384:
    def __init__(self, addr):
        rm_py = visa.ResourceManager()
        self.equipment = rm_py.open_resource(addr)
        
    def set_frequency(self, value, unit='GHz'):
        self.equipment.write("FREQ %f %s" % (value, unit))
    
    def get_frequency(self, unit='GHz'):
        return self.equipment.query("FREQ? %s" % (unit))

    def set_amplitude(self, value, unit='dBm'):
        self.equipment.write("AMPR %f %s" % (value, unit))
        
    def get_amplitude(self, unit='dBm'):
        return self.equipment.query("AMPR?")

    def enable_RF_signal(self):
        self.equipment.write("ENBR 1")

    def disable_RF_signal(self):
        self.equipment.write("ENBR 0")

