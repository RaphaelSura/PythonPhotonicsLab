import sys
sys.path.append("..")
import matplotlib
import matplotlib.pyplot as plt
import nidaqmx
import numpy as np
from LowLevelModules.NIdaq import *
from LowLevelModules.NIdaqAPD import *
from LowLevelModules.GeneralFunctions import *
import time
from skimage import data
from skimage.feature import match_template
from scipy.optimize import curve_fit
# http://scipy.github.io/old-wiki/pages/Cookbook/FittingData
from scipy import optimize

# from LowLevelModules.conexagp import *
from LowLevelModules.NIdaqAPD import APDCounter

import ipywidgets as widgets
from IPython.display import display
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# # objective stage
# obj_port = 'COM5'

from LowLevelModules.XPScontroller import XPSstage
# stage stuff
ip_address = '169.254.66.147'
portnum = 5001
groupnum = 1

class XPS_stage:
    def __init__(self):
        self.stage_for_scan = XPSstage(ip_address, portnum, groupnum)
    
    def center_NV_depth(self,scan_extent=2,step_size=0.2,collection_time=0.5,terminal="/Dev1/PFI1" ):
        """
        scan in um
        time in sec
        """
        # move the stage to start scan
        steps = int(2 * scan_extent/step_size)
        pos_before_scan = self.stage_for_scan.read_position()
        time.sleep(collection_time)
        self.stage_for_scan.move_by(-1* (scan_extent+step_size)/1000)
        time.sleep(collection_time)
        shifts = np.arange(-scan_extent, scan_extent+step_size, step_size)

        lp = LivePlot(1, 1, 9, 4, 'o', 'x (um)', 'APD counts (kHz)')
        pos, cts = [], []
        for s in shifts:
            try:
                # move stage to new position
                self.stage_for_scan.move_by(step_size/1000)
                # collect the APD count rate
                APD1 = APDCounter(terminal, task_name='stage scan counts')
                APD1.start()
                time.sleep(collection_time)
                APD_cts = APD1.read() / collection_time / 1000
                APD1.close()
                # get the new data and update the plot
                pos.append(s)
                cts.append(APD_cts)
                lp.plot_live(pos, cts)
            except KeyboardInterrupt:
                # press the stop button to trigger this
                APD1.close()
                self.stage_for_scan.move_to(pos_before_scan)
                break

        # fit data and move the stage
        pos, cts = np.array(pos), np.array(cts)
        bds = ([.1, -2*scan_extent, 0.0001,0],[np.inf, 2*scan_extent, 100,100])
        # popt, pcov = curve_fit(gaussian_func, pos, cts, bounds=bds)
        popt, pcov = curve_fit(lorentzian_bkg_func, pos, cts, bounds=bds)
        
        data_y_fit = lorentzian_bkg_func(pos, *popt)
        plt.plot(pos, data_y_fit, 'r-')
        #check validity of the fit
        if r2_score(cts, data_y_fit)>.7:
            best_shift = popt[1]
            print("Optimized from fit")
        else:
            best_shift = pos[cts.argmax()]
            print("Optimized from max value in scan")
        # double check validity
        if np.abs(best_shift) > 5:
            best_shift = 0
            print("Actually not! Fit too far off. Staying at initial position")
        # move the stage
        self.stage_for_scan.move_to(pos_before_scan + best_shift/1000)
        
        print(f'Amplitude is {popt[0]}')
        print(f'Center is {popt[1]}')
        print("FWHM is %f um" % (2.3548*popt[2]))
        print(f'Background is {popt[3]}')
        print("Position before scan: ", pos_before_scan )
        print("Position after scan: ", self.stage_for_scan.read_position())
    
class FSM:
    def __init__(self, x0=0, y0=0,setup='B13_50x'):
        
        self.ao_chan = {'x': '/Dev1/ao1', 'y': '/Dev1/ao0'} 
        self.ai_chan = {'x': '/Dev1/ai1', 'y': '/Dev1/ai0'}
        
        # um/volt
        if setup == 'B702nd':
            self.conversion = {'x': -13.90, 'y': -9.69}     # for 100x NA0.8 Olympus obj
            self.APD_terminal = "/Dev1/PFI0"
        elif setup == 'B13_50x':
            self.conversion = {'x': 19.29, 'y': 13.46}
        elif setup == 'B13_100x':
            self.conversion = {'x': 8.58 , 'y': 6.22}
        else:
            self.conversion = {'x': -13.90, 'y': -9.69}
            print('unknown setup, set conversion to x: -13.90, y: -9.69  ')
        
        if setup == 'B702nd':
            self.counter_terminal= "/Dev1/PFI0"
        else:
            self.counter_terminal= "/Dev1/PFI1" # B13
            
        self.counterPort = "/Dev1/ctr1"
        
        # clock source
        self.counter_term_clk = '/Dev1/ctr0'
        self.trig_src =  '/Dev1/PFI12' 
        
        self.position = {'x': x0, 'y': y0}    # this line will run setter function of position property
        curr_x, curr_y = self.read_position()
        self.position = {'x': curr_x, 'y': curr_y}    # this line will run setter function of position property
        
#     @property
#     def position(self, unit='volts'):
#         return self._position
        
#     @position.setter
#     def position(self, new_pos=(0,0)):
#         # set the new current position
#         self._position = new_pos
    
    def micron_to_volts(self, pos_in_micron, axis):
        return pos_in_micron / self.conversion[axis]

    def volts_to_micron(self, pos_in_volts, axis):
        return pos_in_volts * self.conversion[axis]

    def read_position(self, unit='volts'):
        """ Voltage reading of the 2 AI from the DAQ
            Returns in Volts unless asked otherwise """
        with nidaqmx.Task() as fsm_task:
            fsm_task.ai_channels.add_ai_voltage_chan(self.ai_chan['x'], 'FSM x axis')
            fsm_task.ai_channels.add_ai_voltage_chan(self.ai_chan['y'], 'FSM y axis')
            curr_x, curr_y = fsm_task.read()
        self.position['x'], self.position['y'] = curr_x, curr_y
        if unit != 'volts':
            curr_x = self.volts_to_micron(curr_x, 'x')
            curr_y = self.volts_to_micron(curr_y, 'y')
        return curr_x, curr_y

    def calc_trajectory_in_V(self, final_pos, n_pts, axis):
        """  Returns an array in volts of a sine-like profile 
             for sweeping the values from one position to the next 
             final position in um
         """
        init_pos_V = self.position[axis]
        final_pos_V = self.micron_to_volts(final_pos, axis)
        delta_pos = final_pos_V - init_pos_V
        x_sine = np.linspace(-np.pi/2, np.pi/2, n_pts)
        sweep_array = (np.sin(x_sine) + 1)/2*(final_pos_V-init_pos_V) + init_pos_V
        return sweep_array

    def go_to_position(self, x, y):
        """ Smooth sinusoidal move from current position to (x,y) in um """
        #curr_x, curr_y = read_position('microns')
        curr_x = self.volts_to_micron(self.position['x'], 'x')
        curr_y = self.volts_to_micron(self.position['y'], 'y')
        
        currV_x = self.position['x']
        currV_y = self.position['y']
        finV_x = self.micron_to_volts(x,'x')
        finV_y = self.micron_to_volts(y,'y')
        nx = int(np.ceil((abs(finV_x-currV_x)*5000+10)/2))
        ny = int(np.ceil((abs(finV_y-currV_y)*5000+10)/2))    
        n_max = max(nx,ny)

        # 2 arrays need to have the same size
        xVsweep_array = self.calc_trajectory_in_V(x, n_max, 'x')
        yVsweep_array = self.calc_trajectory_in_V(y, n_max, 'y')

        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(self.ao_chan['y'], 'FSM y axis', min_val=-9, max_val=9)
            task.ao_channels.add_ao_voltage_chan(self.ao_chan['x'], 'FSM x axis', min_val=-9, max_val=9)
            task.timing.cfg_samp_clk_timing(rate= 50000, 
                                            active_edge=nidaqmx.constants.Edge.RISING ,
                                            sample_mode= nidaqmx.constants.AcquisitionType.FINITE, 
                                            samps_per_chan=len(xVsweep_array))
            task.write(np.array([yVsweep_array, xVsweep_array]), auto_start=True, timeout=5)
            task.wait_until_done(timeout=5)

        # set the new current position
        self.position['x'], self.position['y'] = self.read_position()

    def scan_1D_axis(self, x1, xf, mesh_pts=20, scan_rate=100, axis = 'x'):
        """
        either x or y axis
        x1,xf - um
        """
        counter_terminal = self.counter_terminal
        counterPort = self.counterPort 
        
        v_range = abs(self.micron_to_volts(xf,axis)-self.micron_to_volts(x1,axis))
        deltaV = 20.0/2**16 # DAC resolution, 16 bit, -10 to 10V output
        
        mult_fac = max(int(np.ceil(v_range/deltaV)),1)
        mult_fac = min(mult_fac,10) 
        
        ao_scan_rate = scan_rate*mult_fac
        ao_pts = mesh_pts*mult_fac

        # x trajectory 
        axis_scan = np.linspace(x1, xf, mesh_pts)

        # voltage scan is denser than the mesh
        v_ao_scan = self.micron_to_volts(np.linspace(x1, xf, ao_pts), axis)     
        
        # clock source
        counter_term_clk = self.counter_term_clk
        trig_src = self.trig_src       

        # setup counter
        countAPD = CI(counter_terminal, counterPort)
        countAPD.config_read_samples(trig_src, mesh_pts + 1, scan_rate)        

        # set up FSM AO AI
        FSMXtask = AO(self.ao_chan[axis])
        FSMXtask.config_write(v_ao_scan, ao_scan_rate, trig_src)

        FSMreadXtask = AI(self.ai_chan[axis])
        FSMreadXtask.config_read(ao_pts, ao_scan_rate, trig_src)

        # PFI12 start
        # CREATE EXT CLOCK TO GATE THE READING OF PULSES COMING FROM THE APD
        ext_clock_task = CO(counter_term_clk, scan_rate)

        # wait until done and read data
        FSMXtask.wait_until_done()
        FSMXtask.close()

        aiV = FSMreadXtask.read(number_of_samples_per_channel=ao_pts)
        thisX = self.volts_to_micron(np.asarray(aiV), axis)
        downSampleX = thisX[::mult_fac]
        FSMreadXtask.close()

        rawAPD=countAPD.read(number_of_samples_per_channel=mesh_pts+1)
        correctAPD = np.diff(rawAPD)*scan_rate/1000 # kHz

        countAPD.close()
        ext_clock_task.close()
    
        # set the new current position
        self.position['x'], self.position['y'] = self.read_position()
        return downSampleX, correctAPD
    
    def scan_2D(self,center_x = 0,center_y = 0,x_size = 20,y_size=20,mesh_x=20,mesh_y=20,scan_rate=200,save_data = True):
        """  
        Line by line live plot
        allow x and y to be of different number of points and different length
        curr_x,curr_y - um
        x_size, y_size - um
        mesh_x, mesh_y - number of points
        
        Parameters: 
        Returns:
        xx
        yy
        FSM2D (array): Scan image in kHz
        Data format
        FSM[row,col] corresponds to coordinate (X=xx[col],Y=yy[row])  
        """
        start_x = center_x - x_size/2.0
        end_x = center_x + x_size/2.0

        start_y = center_y + y_size/2.0
        end_y = center_y - y_size/2.0

        x_scan = np.linspace(start_x, end_x, mesh_x)
        y_scan = np.linspace(start_y, end_y, mesh_y)

        FSM2D = np.zeros((mesh_y, mesh_x)) # (rows,columns)
        lp = LivePlotFSM((6, 8), x_scan, y_scan, FSM2D, 'x (um)', 'y (um)', 'APD kcts')

        for i, y0 in enumerate(y_scan):
            try:
                self.go_to_position(start_x, y0)
                xx, cts = self.scan_1D_axis(start_x, end_x, mesh_x, scan_rate,axis='x')
                FSM2D[i,:] = cts
                lp.plot_live(x_scan, cts, FSM2D)

            except KeyboardInterrupt:
                # press the stop button to trigger this
                FSMXtask.close()
                FSMreadXtask.close()
                countAPD.close()
                ext_clock_task.close()
                break
#         lp.plot_live(x_scan, cts, FSM2D)
        plt.close(lp.fig)
        # back to initial location
        self.go_to_position(center_x, center_y)
        # set the new current position
        self.position['x'], self.position['y'] = self.read_position()
        if save_data:
            #replot figure
            fig1, ax1 = plt.subplots(1, 1, figsize=(5,5))
            #2D sub plot
            extent_x = (np.max(x_scan) - np.min(x_scan)) / 2
            extent_y = (np.max(y_scan) - np.min(y_scan)) / 2
            extent = [np.min(x_scan), np.max(x_scan), np.min(y_scan), np.max(y_scan)]
#             aspect_ratio = abs((x_scan[-1] - x_scan[0]) / (y_scan[-1] - y_scan[0]))
            cp1 = ax1.imshow(FSM2D, cmap='gray', extent=extent,
                             interpolation='nearest', aspect=1)
            cb1 = fig1.colorbar(cp1, fraction=0.046, pad=0.04)
            ax1.set_xlabel('x (um)', labelpad=10, fontsize=18)
            ax1.set_ylabel('y (um)', labelpad=10, fontsize=18)
            cb1.set_label('APD counts (kHz)', labelpad=20, fontsize=18)
            cb1.ax.set_yticklabels(cb1.ax.get_yticklabels(), fontsize=16)
            ax1.tick_params(axis='both', labelsize=16)
        #     fig1.subplots_adjust(hspace=.3, left=0.3, bottom=0.15, right = 0.8)
            # rest of the info
            data_type = 'FSM-2D'
            save_file_names = data_save_2D(x_scan, y_scan, FSM2D, fig1, data_type)
        return x_scan,y_scan,FSM2D
    
#     def scan_1D_cross_section(self,center = 0,size = 20,mesh_size=20,zstart=-2,zend=30,zstep=0.4,axis='x',scan_rate=200,save_data = True):
#         """  
#         Line by line live plot
#         allow x and y to be of different number of points and different length
#         curr_x,curr_y - um
#         x_size, y_size - um
#         mesh_x, mesh_y - number of points
        
#         Parameters: 
#         Returns:
#         xx
#         yy
#         FSM2D (array): Scan image in kHz
#         Data format
#         FSM[row,col] corresponds to coordinate (X=xx[col],Y=yy[row])  
#         """
#         curr_x,curr_y = self.read_position('micron')
        
#         start_plane = center - size/2.0
#         end_plane = center + size/2.0

#         plane_scan = np.linspace(start_plane, end_plane, mesh_size)
#         if zend<=zstart:
#             print('zend need to be larger than zstart')
#             return 0
#         z_scan = np.arange(zstart,zend,zstep)
#         mesh_z = len(z_scan)

#         FSM1Dcross = np.zeros((mesh_z, mesh_size)) # (rows,columns)
#         lp = LivePlotFSM((6, 8), plane_scan, z_scan, FSM1Dcross, axis+' (um)', 'z (um)', 'APD kcts')
        
#         objective_stage = ConexAGP(obj_port)
#         objective_stage.move_relative(zstart/1000)
#         for i, y0 in enumerate(z_scan):
#             try:
#                 if axis == 'x':                    
#                     self.go_to_position(start_plane, curr_y)
#                 else:
#                     self.go_to_position(curr_x, start_plane)
#                 objective_stage.move_relative(zstep/1000)
# #                 scan_1D_axis(self, x1, xf, mesh_pts=20, scan_rate=100, axis = 'x', counter_terminal= "/Dev1/PFI0"):
#                 xx, cts = self.scan_1D_axis(start_plane, end_plane, mesh_size, scan_rate,axis)
#                 FSM1Dcross[mesh_z-1-i,:] = cts
#                 lp.plot_live(plane_scan, cts, FSM1Dcross)

#             except KeyboardInterrupt:
#                 # press the stop button to trigger this
#                 FSMXtask.close()
#                 FSMreadXtask.close()
#                 countAPD.close()
#                 ext_clock_task.close()
#                 break
# #         lp.plot_live(x_scan, cts, FSM2D)
#         plt.close(lp.fig)
#         # back to initial location
#         if axis == 'x':                   
#             self.go_to_position(center, curr_y)
#         else:
#             self.go_to_position(curr_x, center)
                    
#         objective_stage.move_relative(-zend/1000)
#         objective_stage.close()
#         # set the new current position
#         self.position['x'], self.position['y'] = self.read_position()
#         if save_data:
#             #replot figure
#             fig1, ax1 = plt.subplots(1, 1, figsize=(5,5))
#             #2D sub plot
#             extent_x = (np.max(plane_scan) - np.min(plane_scan)) / 2
#             extent_y = (np.max(z_scan) - np.min(z_scan)) / 2
#             extent = [np.min(plane_scan), np.max(plane_scan), np.min(z_scan), np.max(z_scan)]
# #             aspect_ratio = abs((x_scan[-1] - x_scan[0]) / (y_scan[-1] - y_scan[0]))
#             cp1 = ax1.imshow(FSM1Dcross, cmap='gray', extent=extent,
#                              interpolation='nearest', aspect=1)
#             cb1 = fig1.colorbar(cp1, fraction=0.046, pad=0.04)
#             ax1.set_xlabel(axis+' (um)', labelpad=10, fontsize=18)
#             ax1.set_ylabel('z (um)', labelpad=10, fontsize=18)
#             cb1.set_label('APD counts (kHz)', labelpad=20, fontsize=18)
#             cb1.ax.set_yticklabels(cb1.ax.get_yticklabels(), fontsize=16)
#             ax1.tick_params(axis='both', labelsize=16)
#         #     fig1.subplots_adjust(hspace=.3, left=0.3, bottom=0.15, right = 0.8)
#             # rest of the info
#             data_type = 'FSM-1D-cross-section'
#             save_file_names = data_save_2D(plane_scan, z_scan, FSM1Dcross, fig1, data_type)
#         return plane_scan,z_scan,FSM1Dcross
    
    def center_NV_2D(self,size_x=2,size_y=2,mesh_x=60,mesh_y=60,scan_rate=200):
        """
        do a small area scan to center an NV
        """
        curr_x,curr_y = self.read_position('micron')
        xx1,yy1,FSM2D = self.scan_2D(curr_x,curr_y,x_size=size_x,y_size=size_y,mesh_x=mesh_x,mesh_y=mesh_y,scan_rate=scan_rate)
        
        data = FSM2D
        plt.matshow(data, cmap=plt.cm.gist_earth_r)

        params = fitgaussian2D(data)
        fit = gaussian2D(*params)

        plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
        ax = plt.gca()
        (height, row, col, width_row, width_col,bkg) = params

        plt.text(0.95, 0.05, """
        height : %.1f
        row : %.1f
        col : %.1f
        width_row : %.1f
        width_col : %.1f
        bkg : %.1f""" %(height,row, col, width_row, width_col,bkg),
                fontsize=16, horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes)

        xx1ind = np.array(range(len(xx1)))
        xfin = np.interp(col,xx1ind,xx1) 
        yy1ind = np.array(range(len(yy1)))
        yfin = np.interp(row,yy1ind,yy1)
        
        if abs(xfin-curr_x)>size_x/2 or abs(yfin-curr_y)>size_y/2:
            print('not a good fit')
        else:
            self.go_to_position(xfin,yfin)
        self.position['x'], self.position['y'] = self.read_position()
        
        return 0
    
    def track_NV_2D(self,template, size_x,size_y,scan_rate=200):
        """ 
        Track NV given a template (with the NV at its geometric center)
        2D scan keeping objective fixed

        Parameters: 
        template (array): NV template with an NV at its geometric center
        size_x,size_y (float): length of the image in um

        Returns: 
        (x,y) (float,float): final NV location in um   
        """    
        # pattern match

        # Find NV through either LabView or Python
        # Scan image, say 10 x 10 um with the NV at center, 100 x 100 pts
        # store the image
        # use the center 5 x 5 um as a template (50 x 50 pts)
        templateX_pt = template.shape[1] # number of columns
        templateY_pt = template.shape[0] # number of rows

        startX_pt = int(np.floor(templateX_pt/4))
        endX_pt = int(np.ceil(templateX_pt*3/4))
        
        startY_pt = int(np.floor(templateY_pt/4))
        endY_pt = int(np.ceil(templateY_pt*3/4))

        row_center = startY_pt #(1+template_pt)/2
        col_center = startX_pt #(1+template_pt)/2    

        template_center = template[startY_pt:endY_pt, startX_pt:endX_pt]

        # use current x ,y
        curr_x,curr_y = self.read_position('micron')
        xx1,yy1,FSM2D = self.scan_2D(curr_x,curr_y,x_size=size_x,y_size=size_y,mesh_x=templateX_pt,mesh_y=templateY_pt,scan_rate=scan_rate)
        # tracking
        # scan 10 x 10 um
        # use the 5 x 5 um image as a template and find maximum
        # compute the difference

        result = match_template(FSM2D, template_center)
        ij = np.unravel_index(np.argmax(result), result.shape)
        col,row = ij[::-1]

        dy = size_y/templateY_pt*(row_center-row)
        dx = size_x/templateX_pt*(col-col_center)
        print(dx,dy)
        if abs(dx)>size_x or abs(dy)>size_y:
            print("tracked (dx, yy) = ", dx ,dy)
            print("moving back to old (x, y) = ", curr_x ,curr_y)
            # set the new current position
            self.position['x'], self.position['y'] = self.read_position()
            return (curr_x,curr_y)            
        if curr_x+dx <-50 or curr_x+dx>50 or curr_y+dx <-50 or curr_y+dy>50:
            print("Out of range, tracked (x, y) = ", curr_x+dx ,curr_y+dy)
            print("moving back to old (x, y) = ", curr_x ,curr_y)
            # set the new current position
            self.position['x'], self.position['y'] = self.read_position()
            return (curr_x,curr_y)
        else:    
            # go to new location
            self.go_to_position(curr_x+dx,curr_y+dy)
            # set the new current position
            self.position['x'], self.position['y'] = self.read_position()
            return (curr_x+dx,curr_y+dy)