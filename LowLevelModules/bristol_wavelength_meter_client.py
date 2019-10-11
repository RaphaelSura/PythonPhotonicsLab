"""
bristol_wavelength_meter_client.py
	@Purpose:
		A ctype python wrapper to access the basic measurement ability to a
		bristol wavelength meter.
		
	@Author:
		Charlie Chen <cchen@zygo.com>
"""
__version__ = '1.0'
__owner__ = 'cchen'
'''
History:
3/15/2013	cchen	first create
'''

from ctypes import *
import os
import platform


class BristolWaveLengthMeterWrapperException(Exception):
	"""
	class BristolWaveLengthMeterWrapperException
	@Purpose:
		raise specialized exception for wavelength meter
	@Inputs:
		expr = expression where excpetion occurs
		msg = exception message
	"""
	def __init__(self, expr, msg):
		self.expr = expr
		self.msg = msg
	
	def __str__(self):
		return "ERROR: BristolWaveLengthMeterWrapperException; expression = %s; %s\n" % \
				(repr(self.expr), self.msg)

c_int_p = POINTER(c_int)
c_int8_p = POINTER(c_int8)
c_ubyte_p = POINTER(c_ubyte)
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)
c_void_p_p = POINTER(c_void_p)
c_short_p = POINTER(c_short)

# dll file (private)
#__dll__ = windll.LoadLibrary(os.path.dirname(os.path.abspath(__file__))+'\\'+'__CLDevIFace.dll')

# load proper verion of DLL
if platform.architecture()[0] == '32bit':
    __dll__ = cdll.LoadLibrary(r'Z:\PythonPhotonicsLab\LowLevelModules\CLDevIFace.dll')
elif platform.architecture()[0] == '64bit':
    __dll__ = cdll.LoadLibrary(r'Z:\PythonPhotonicsLab\LowLevelModules\CLDevIFace_x64.dll')
else:
    print('Python architecture unknown')

def CLOpenUSBSerialDevice(ComNumber):
	"""
	@Purpose:
		Open the device using a USB Serial Port Interface
	@Inputs:
		ComNumber: the windows COM port number where the USB driver is installed
	@Outputs:
		A valid CLDevice handle number, or -1 on failure. This device handle will
		be used with all commands to identify the port.
	"""
	funct = __dll__.CLOpenUSBSerialDevice
	funct.restype = c_int
	funct.argtypes = [c_int]
	
	retval = funct(ComNumber)
	if retval == -1:
		raise BristolWaveLengthMeterWrapperException('CLOpenUSBSerialDevice(%s)' % ComNumber,
				'ERROR: CLOpenUSBSerialDevice(%s) failed with status -1' % ComNumber)
	return retval

def CLSetMeasHBCallback(DeviceHandle, ProcessMeasHBData):
	"""
	THIS FUNCTION IS NOT IMPLEMENTED, YET.
	@Purpose:
		Set a user defined callback function to receive measurement information from
		the instrument when it is available
	@Inputs:
		DeviceHandle: a valid CLDevice handle.
		ProcessMeasHBData: user supplied callback funciton.
	"""
	funct = __dll__.CLSetMeasHBCallback
	funct.restype = c_int
	pass
	# TO BE IMPLEMENTED.
	
def CLGetMeasurementData(DeviceHandle, data):
	"""
	THIS FUNCTION IS NOT IMPLEMENTED, YET.
	@Purpose:
		Set a user defined callback function to receive measurement information
		from the instrument when it is available.
	@Inputs:
		DeviceHandle: a valid CLDevice handle.
		Data: pointer to locaiton to write data of type tsMeasurementDataType.
	"""
	funct = __dll__.CLGetMeasurementData
	funct.restype = c_int
	pass
	# TO BE INPLEMENTED.
	
def CLSetLambdaUnits(DeviceHandle, LambdaUnits):
	"""
	@Purpose:
		Set the units returned from the CLGetLambdaReading.
	@Inputs:
		DeviceHandle: a valid CLDevice handle.
		LambdaUnits: 'nm' = nanometers
					 'GHz' = gigahertz
					 '1/cm' = inverse centimeters
	"""
	funct = __dll__.CLSetLambdaUnits
	funct.restype = c_int
	funct.argtypes = [c_int, c_uint]
	
	if LambdaUnits == 'nm':
		retval = funct(DeviceHandle, 0)
	elif LambdaUnits == 'GHz':
		retval = funct(DeviceHandle, 1)
	elif LambdaUnits == '1/cm':
		retval = funct(DeviceHandle, 2)
	else:
		raise BristolWaveLengthMeterWrapperException('LambdaUnits = %s' % LambdaUnits,
				"ERROR: LambdaUnits must be either 'nm', 'GHz', or '1/cm', and case matters!")
	
	if retval != 0:
		raise BristolWaveLengthMeterWrapperException('CLSetLambdaUnits(%s, %s)' % (DeviceHandle, LambdaUnits),
				"ERROR: CLSetLambdaUnits(%s, %s) failed." % (DeviceHandle, LambdaUnits))
				
def CLSetPowerUnits(DeviceHandle, PowerUnits):
	"""
	@Purpose:
		Set the units returned from the CLGetPowerReading.
	@Inputs:
		DeviceHandle: a valid CLDevice handle.
		PowerUnits: 'mw' = miliwatts
					'dB' = decibels
	"""
	funct = __dll__.CLSetPowerUnits
	funct.restype = c_int
	funct.argtypes = [c_int, c_uint]
	
	if PowerUnits == 'mw':
		retval = funct(DeviceHandle, 0)
	elif PowerUnits == 'dB':
		retval = funct(DeviceHandle, 1)
	else:
		raise BristolWaveLengthMeterWrapperException('PowerUnits = %s' % PowerUnits,
			"ERROR: PowerUnits must be either 'mw' or 'dB', and case matters!")
	
	if retval != 0:
		raise BristolWaveLengthMeterWrapperException('CLSetPowerUnits(%s, %s)' % (DeviceHandle, PowerUnits),
				"ERROR: CLSetPowerUnits(%s, %s) failed." % (DeviceHandle, PowerUnits))

def CLGetLambdaReading(DeviceHandle):
	"""
	@Purpose:
		Get the current wavelength reading in units set by CLSetLambdaUnits.
	@Inputs:
		DeviceHandle: a valid CLDevice handle.
	@Outputs:
		64 bit floating point wavelength reading in units set by CLSetLambdaUnits.
	"""
	funct = __dll__.CLGetLambdaReading
	funct.restype = c_double
	funct.argtypes = [c_int]
	
	return funct(DeviceHandle)
	
def CLGetPowerReading(DeviceHandle):
	"""
	@Purpose:
		Get the current power reading in units set by CLSetPowerUnits.
	@Inputs:
		DeviceHandle: a valid CLDevice handle.
	@Outputs:
		32 bit floating point value of the power in units set by CLSetPowerUnits.
	"""
	funct = __dll__.CLGetPowerReading
	funct.restype = c_float
	funct.argtypes = [c_int]
	
	return funct(DeviceHandle)
	
def CLSetMedium(DeviceHandle, medium):
	"""
	@Purpose:
		Use to set the current medium for the CLGetLambdaReading function.
	@Inputs:
		DeviceHandle: a valid CLDevice handle.
		medium: 'vacuum' or 'air'
	"""
	funct = __dll__.CLSetMedium
	funct.restype = c_int
	funct.argtypes = [c_int, c_uint]
	
	if medium == 'vacuum':
		retval = funct(DeviceHandle, 0)
	elif medium == 'air':
		retval = funct(DeviceHandle, 1)
	else:
		raise BristolWaveLengthMeterWrapperException('medium = %s' % medium,
			"ERROR: medium must be either 'vacuum' or 'air', and case matters!")
	
	if retval != 0:
		raise BristolWaveLengthMeterWrapperException('CLSetMedium(%s, %s)' % (DeviceHandle, medium),
				"ERROR: CLSetMedium(%s, %s) failed." % (DeviceHandle, medium))