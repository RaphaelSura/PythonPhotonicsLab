# NewPort Conex-CC class
# Author Itay Shahak 2019
# Free code for the community!

# dependant on 'clr' which is PythonNet package
import clr
from time import sleep

# modify this to the correct directory
# it seems that saving on Z: (network drive) doesn't work
# Best to put the local directory
clr.AddReference(r'C:\Program Files (x86)\Newport\MotionControl\CONEX-CC\Bin\Newport.CONEXCC.CommandInterface')

import CommandInterfaceConexCC

DEV = 1                # hardcoded here to the first device
MAX_VELOCITY = 5.0     # mm/s, by spec of NewPort LTA-HS


class ConexCC:

    def __init__(self, com_port):
        self.min_limit = -1
        self.max_limit = -1
        self.cur_pos = -1
        self.controller_state = ''
        self.positioner_error = ''

        self.driver = CommandInterfaceConexCC.ConexCC()
        ret = self.driver.OpenInstrument(com_port)
        if ret != 0:
            print('Oops: error opening port %s' % com_port)
            self.positioner_error = 'init failed'
#         else:
#             print('ConexCC: Successfully connected to %s' % com_port)
#             self.read_velocity()
#             self.set_velocity(velocity)
#             self.set_homing_velocity(velocity)
#             self.read_limits()
#             self.read_cur_pos()

    def wait_for_ready(self, timeout=60):
        print('waiting for ready state...', end='')
        count = 0
        sleep_interval = 0.2
        last_count = (1 / sleep_interval) * timeout
        while not self.is_ready():
            count += 1
            if count % 30 == 0:
                print('<%s>' % self.controller_state)
            else:
                print('<%s>' % self.controller_state, end='', flush=True)
            sleep(sleep_interval)
            if count >= last_count:
                print('\nfailed to become ready. existing for timeout = %d seconds.' % timeout)
                return False
        print('ok')
        return True

    def is_ready(self):
        self.read_controller_state(silent=True)

        if self.controller_state in ('3D', '3C'):  # in DISABLE state
            self.exit_disable_state()
            sleep(0.2)
            self.read_controller_state()
        elif self.controller_state.startswith('0'):  # not referenced state
            self.init_positioner()
            sleep(0.4)

        # ('32','33','34') means in READY state
        ready = self.positioner_error == '' and self.controller_state in ('32', '33', '34')
        return ready

    @classmethod
    def dump_possible_states(cls):
        # https://www.newport.com/mam/celum/celum_assets/resources/CONEX-CC_-_Controller_Documentation.pdf#page=54
        help_text = '''===== Conex-CC Controller States =====
            – 0A: NOT REFERENCED from RESET.
            – 0B: NOT REFERENCED from HOMING.
            – 0C: NOT REFERENCED from CONFIGURATION.
            – 0D: NOT REFERENCED from DISABLE.
            – 0E: NOT REFERENCED from READY.
            – 0F: NOT REFERENCED from MOVING.
            – 10: NOT REFERENCED - NO PARAMETERS IN MEMORY.
            – 14: CONFIGURATION.
            – 1E: HOMING.
            – 28: MOVING.
            – 32: READY from HOMING.
            – 33: READY from MOVING.
            – 34: READY from DISABLE.
            – 36: READY T from READY.
            – 37: READY T from TRACKING.
            – 38: READY T from DISABLE T.
            – 3C: DISABLE from READY.
            – 3D: DISABLE from MOVING.
            – 3E: DISABLE from TRACKING.
            – 3F: DISABLE from READY T.
            – 46: TRACKING from READY T.
            – 47: TRACKING from TRACKING.  
            ===========================================      
        '''
        for s in help_text.split('\n'):
            print(s.strip(' '))

    def read_limits(self):
        err_str = ''
        resp = 0
        res, resp, err_str = self.driver.SL_Get(DEV, resp, err_str)
        if res != 0 or err_str != '':
            print('Oops: Negative SW Limit: result=%d,response=%.2f,errString=\'%s\'' % (res, resp, err_str))
            return -1
        else:
#             print('Negative SW Limit = %.1f' % resp)
            self.min_limit = resp            

        res, resp, err_str = self.driver.SR_Get(DEV, resp, err_str)
        if res != 0 or err_str != '':
            print('Oops: Positive SW Limit: result=%d,response=%.2f,errString=\'%s\'' % (res, resp, err_str))
            return -1
        else:
#             print('Positive SW Limit = %.1f' % resp)
            self.max_limit = resp
        return (self.min_limit,self.max_limit)
            
    def set_pos_limit(self,pos_limit):
        err_str = ''
        resp = 0
        res, err_str = self.driver.SR_Set(DEV, pos_limit, err_str)
        if res != 0 or err_str != '':
            print('Oops: Current Position: result=%d,response=%.2f,errString=\'%s\'' % (res, resp, err_str))
            return -1
        else:
            self.max_limit = pos_limit
            return 0
            
    def set_neg_limit(self,neg_limit):
        err_str = ''
        resp = 0
        res, err_str = self.driver.SL_Set(DEV, neg_limit, err_str)
        if res != 0 or err_str != '':
            print('Oops: Current Position: result=%d,response=%.2f,errString=\'%s\'' % (res, resp, err_str))
            return -1
        else:
            self.min_limit = neg_limit
            return 0
        
    def read_cur_pos(self):
        err_str = ''
        resp = 0
        res, resp, err_str = self.driver.TP(DEV, resp, err_str)
        if res != 0 or err_str != '':
            print('Oops: Current Position: result=%d,response=%.2f,errString=\'%s\'' % (res, resp, err_str))
        else:
#             print('Current Position = %.3f' % resp)
            self.cur_pos = resp
            return resp

    def read_home_velocity(self):
        err_str = ''
        resp = 0
        res, resp, err_str = self.driver.OH_Get(DEV, resp, err_str)
        if res != 0 or err_str != '':
            print('Oops: Current Velocity: result=%d,response=%.2f,errString=\'%s\'' % (res, resp, err_str))
        else:
            return resp
    
    def read_velocity(self):
        err_str = ''
        resp = 0
        res, resp, err_str = self.driver.VA_Get(DEV, resp, err_str)
        if res != 0 or err_str != '':
            print('Oops: Current Velocity: result=%d,response=%.2f,errString=\'%s\'' % (res, resp, err_str))
        else:
            return resp
#         print('Current Velocity = %.3f' % resp)

    def read_controller_state(self, silent=False):
        err_str = ''
        resp = ''
        resp2 = ''
        res, resp, resp2, errString = self.driver.TS(DEV, resp, resp2, err_str)
        if res != 0 or err_str != '':
            print('Oops: Read controller Err/State: result=%d,response=Err=\'%s\'/State=\'%s\',err_str=\'%s\'' % (
                res, resp, resp2, err_str))
        else:
            if not silent:
                print('Controller State = \'%s\', Error = \'%s\'' % (resp2, resp))
            self.positioner_error = resp
            self.controller_state = resp2

    def exit_disable_state(self):
        err_str = ''
        state = 1  # enable
        res, err_str = self.driver.MM_Set(DEV, state, err_str)
        if res != 0 or err_str != '':
            print('Oops: Leave Disable: result=%d,errString=\'%s\'' % (res, err_str))
        else:
            print('Exiting DISABLE state')

    def init_positioner(self):
        err_str = ''
        res, err_str = self.driver.OR(DEV, err_str)
        if res != 0 or err_str != '':
            print('Oops: Find Home: result=%d,errString=\'%s\'' % (res, err_str))
        else:
            print('Finding Home')
    
    
    def set_homing_velocity(self, velocity):
        """
        does not work
        """
        if velocity > MAX_VELOCITY:
            velocity = MAX_VELOCITY
        err_str = ''
        res, err_str = self.driver.OH_Set(DEV, velocity, err_str)
        if res != 0 or err_str != '':
            print('Oops: Homing velocity: result=%d,errString=\'%s\'' % (res, err_str))
            return -1
        else:
            return 0
#         print('Homing velocity set to %.1f mm/s' % velocity)

    def set_velocity(self, velocity):
        if velocity > MAX_VELOCITY:
            velocity = MAX_VELOCITY
        err_str = ''
        res, err_str = self.driver.VA_Set(DEV, velocity, err_str)
        if res != 0 or err_str != '':
            print('Oops: Set velocity: result=%d,errString=\'%s\'' % (res, err_str))
            return -1
        else:
            return 0
#         print('velocity Set to %.1f mm/s' % velocity)

    def move_relative(self, distance):
        if self.is_ready():
            err_str = ''
            res, err_str = self.driver.PR_Set(DEV, distance, err_str)
            if res != 0 or err_str != '':
                print('Oops: Move Relative: result=%d,errString=\'%s\'' % (res, err_str))
                return -1
            else:
                return 0
#             print('Moving Relative %.3f mm' % distance)

    def move_absolute(self, new_pos):
        if self.is_ready():
            err_str = ''
            res, err_str = self.driver.PA_Set(DEV, new_pos, err_str)
            if res != 0 or err_str != '':
                print('Oops: Move Absolute: result=%d,errString=\'%s\'' % (res, err_str))
                return -1
            else:
                return 0
#                 print('Moving to position %.3f mm' % new_pos)

    def close(self):
        # note that closing the communication will NOT stop the motor!
        self.driver.CloseInstrument()


if __name__ == '__main__':
    ConexCC.dump_possible_states()
    conex_cc = ConexCC(com_port='com4', velocity=0.5)
    ready = conex_cc.wait_for_ready(timeout=60)
    if ready:
        conex_cc.move_absolute(conex_cc.max_limit / 2)
        ready = conex_cc.wait_for_ready(timeout=60)
        if ready:
            conex_cc.move_relative(-3)
            ready = conex_cc.wait_for_ready(timeout=60)
            if ready:
                print('ok!')
            else:
                print('not ok 2!')
        else:
            print('not ok 1!')
        conex_cc.close()
    else:
        print('something went wrong')