import visa


class LakeShore335:
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
    def __init__(self, addr):
        self.equipment = visa.ResourceManager().open_resource(addr)

    #     def set_target_temperature(self, value):
    #         self.equipment.write("SETP 1, %f" % (value,))

    def get_field(self):
        field = self.equipment.query("RDGFIELD?")
        return float(field.strip())

    def auto_range(self):
        """ Sets the field range to automatically adjust """
        self.equipment.write("AUTO")

    def zero_probe(self):
        """ Initiates the zero field sequence to calibrate the probe """
        self.equipment.write("ZPROBE")