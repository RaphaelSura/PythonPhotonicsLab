# ZWO EFW mini filter wheel
# use DLL from https://astronomy-imaging-camera.com/software/ZWO_EFW_wheel_Labview_demo.zip
# 32 bit DLL, check python version before use

from ctypes import *
import os
import platform
import time
c_int_p = POINTER(c_int)
c_int8_p = POINTER(c_int8)
c_ubyte_p = POINTER(c_ubyte)
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)
c_void_p_p = POINTER(c_void_p)
c_short_p = POINTER(c_short)

# dependant on 'clr' which is PythonNet package
import clr

# if the following does not work, try putting the dll on a local directory instead of on a network drive (Z:)
try:
    __efw_dll__ = cdll.LoadLibrary(r'Z:\PythonPhotonicsLab\LowLevelModules\EFW_filter.dll')
except:
    print(f'Current platform architecutre is {platform.architecture()}')
    print('This DLL is 32 bit, which requires a 32-bit platform')

class EFW:
    def __init__(self):
        self.connected_wheel_number =__efw_dll__.EFWGetNum()
        
        # use pass by reference (C method)
        idref = c_int()
        __efw_dll__.EFWGetID(self.connected_wheel_number,byref(idref))
        self.EFW_ID = idref.value
        __efw_dll__.EFWOpen(self.EFW_ID)
#         self.prop = __efw_dll__.EFWGetProperty(self.EFW_ID)
        while True:
            propref = c_int()
            prop = __efw_dll__.EFWGetProperty(self.EFW_ID,byref(propref))
            if prop != 5:
                break
            else:
                time.sleep(0.5)

        self.bUnidirecitonal=True
        __efw_dll__.EFWSetDirection(self.EFW_ID,self.bUnidirecitonal)
        
        self.EFW_wheel_filter= { '750LP': 4,'open':2}

    def __exit__(self):
        self.close()
    
    def change_filter(self,filtername):
        """filtername is the one in the dictionary"""
        __efw_dll__.EFWSetPosition(self.EFW_ID,self.EFW_wheel_filter[filtername])
    
    def read_position(self):
        while True:
            i = c_int()
            __efw_dll__.EFWGetPosition(self.EFW_ID,byref(i))
            currentPosition = i.value
            if currentPosition !=-1:
                return(currentPosition)
                break
            else:
                time.sleep(0.5)
    def close(self):
        __efw_dll__.EFWClose(self.EFW_ID)
