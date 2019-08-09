import visa


class LakeShore335:

    """ Temperature controller from Lake Shore 335
        Connection via GPIB """

    def __init__(self, addr):
        self.equipment = visa.ResourceManager().open_resource(addr)
        self.heater_range_values = {'off': 0, 'low': 1, 'medium': 2, 'high': 3}

    def set_target_temperature(self, value):
        self.equipment.write("SETP 1, %f" % (value,))

    def get_temp(self, sensor):
        temp = self.equipment.query("KRDG? " + sensor)
        return float(temp[1:].strip())

    def set_heater_range(self, value):
        self.equipment.write("RANGE 1, %d" % (self.heater_range_values[value],))

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
