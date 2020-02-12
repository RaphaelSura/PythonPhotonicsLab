import sys
sys.path.append("..")
import matplotlib
import matplotlib.pyplot as plt
import nidaqmx
import numpy as np
from LowLevelModules.NIdaq import *
from LowLevelModules.NIdaqAPD import *
from LowLevelModules.GeneralFunctions import *
from LowLevelModules.bristol_wavelength_meter_client import *


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

import pandas as pd
import trackpy as tp

from LowLevelModules.XPScontroller import XPSstage

class XPS_stage:
    """B13 objective stage """
    def __init__(self,address='169.254.66.147',portnum=5001,groupnum=1):
        
        self.ip_address = address
        self.portnum = portnum
        self.groupnum = groupnum
        self.stage_for_scan = XPSstage(self.ip_address, self.portnum, self.groupnum)
        
    def center_NV_depth(self,scan_extent=2,step_size=0.2,collection_time=0.5,terminal="/Dev1/PFI1" ):
        """
        scan in um
        time in sec
        """
        # move the stage to start scan
        if scan_extent>10:
            print("Scan extent is too large. It must be <= 10 um.")
            return 0
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
        
        p0 = [max(cts),pos[np.argmax(cts)],scan_extent,  min(cts)]
        try:
            popt, pcov = curve_fit(lorentzian_bkg_func, pos, cts,p0=p0, bounds=bds)
            # The on axis laser intensity should be a Lorentzian
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
        except:
            best_shift = 0
            print("Fit failed. Staying at initial position")
            self.stage_for_scan.move_to(pos_before_scan + best_shift/1000)

class FSM:
    """self.position is the position in (volt) that is WRITTEN to the FSM, except when starting up, where it uses its read position
    """
    def __init__(self, x0=0, y0=0,setup='B13_50x'):
        
        self.ao_chan = {'x': '/Dev1/ao1', 'y': '/Dev1/ao0'} 
        self.ai_chan = {'x': '/Dev1/ai1', 'y': '/Dev1/ai0'}
        
        # unit in um/volt
        # conversion from WRITE voltage to displacement
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
#         self.position_um={'x': x0, 'y': y0}
        
   
        curr_x, curr_y = self.read_position()
        self.position = {'x': curr_x, 'y': curr_y}    # this line will run setter function of position property
        
#         curr_x, curr_y = self.read_position('um')
#         self.position_um={'x': curr_x, 'y': curr_y}
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
        
        self.position['x'], self.position['y'] = curr_x,curr_y
#         self.position_um['x'], self.position_um['y'] = self.volts_to_micron(curr_x,'x'),self.volts_to_micron(curr_y,'y')
                
        if unit != 'volts':
            curr_x = self.volts_to_micron(curr_x, 'x')
            curr_y = self.volts_to_micron(curr_y, 'y')
        return curr_x, curr_y
    
    def return_position(self, unit='volts'):
        """ return position in attribute in either volts or micron """
        if unit == 'volts':
            curr_x, curr_y = self.position['x'], self.position['y']
        else:
            curr_x, curr_y = self.volts_to_micron(self.position['x'],'x'), self.volts_to_micron(self.position['y'],'y')
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
        curr_x, curr_y = self.return_position('um')
#         curr_x = self.volts_to_micron(self.position['x'], 'x')
#         curr_y = self.volts_to_micron(self.position['y'], 'y')
        currV_x, currV_y = self.return_position('volts')
#         currV_x = self.position['x']
#         currV_y = self.position['y']
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
            
        # set the new current position based on the written position    
        self.position['x'], self.position['y'] = self.micron_to_volts(x,'x'),self.micron_to_volts(y,'y')
        return n_max
    
#         self.position_um['x'], self.position_um['y'] = x,y
#         self.position['x'], self.position['y'] = self.read_position()

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
        self.position[axis] = self.micron_to_volts(xf,axis)
#         self.position_um[axis] = xf
#         self.position['x'], self.position['y'] = self.read_position()
        return downSampleX, correctAPD
    
    def scan_2D(self,center_x = 0,center_y = 0,x_size = 20,y_size=20,mesh_x=20,mesh_y=20,scan_rate=200,save_data = True,live_plot=True):
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
        lp = LivePlotFSM((6, 8), x_scan, y_scan, FSM2D, 'x (um)', 'y (um)', 'APD (kct/s)')

        for i, y0 in enumerate(y_scan):
            try:
                self.go_to_position(start_x, y0)
                xx, cts = self.scan_1D_axis(start_x, end_x, mesh_x, scan_rate,axis='x')
                FSM2D[i,:] = cts
                if live_plot:
                    lp.plot_live(x_scan, cts, FSM2D)

            except KeyboardInterrupt:
                # press the stop button to trigger this
                FSMXtask.close()
                FSMreadXtask.close()
                countAPD.close()
                ext_clock_task.close()
                break
        lp.plot_live(x_scan, cts, FSM2D)
        plt.close(lp.fig)
        # back to initial location
        self.go_to_position(center_x, center_y)
        # set the new current position
#         self.position['x'], self.position['y'] = self.read_position()
        self.position['x'], self.position['y'] = self.micron_to_volts(center_x,'x'),self.micron_to_volts(center_y,'y')
#         self.position_um['x'], self.position_um['y'] = center_x,center_y
        
        
        if save_data:
            #replot figure
            fig1, ax1 = plt.subplots(1, 1, figsize=(5,5))
            #2D sub plot
            extent_x = (np.max(x_scan) - np.min(x_scan)) / 2
            extent_y = (np.max(y_scan) - np.min(y_scan)) / 2
            extent = [np.min(x_scan), np.max(x_scan), np.min(y_scan), np.max(y_scan)]
#             aspect_ratio = abs((x_scan[-1] - x_scan[0]) / (y_scan[-1] - y_scan[0]))
            cp1 = ax1.imshow(FSM2D, cmap=lp.bluecmap, extent=extent,
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
#         curr_x,curr_y = self.read_position('micron')
        
#         curr_x = self.volts_to_micron(self.position['x'], 'x')
#         curr_y = self.volts_to_micron(self.position['y'], 'y')
        curr_x, curr_y = self.return_position('um')
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
            print('Not a good fit. Going back to original position')
            x1fin = curr_x
            y1fin = curr_y
        
        self.go_to_position(xfin,yfin)
#         self.position['x'], self.position['y'] = self.read_position()

        self.position['x'], self.position['y'] = (self.micron_to_volts(x1fin,'x'),self.micron_to_volts(y1fin,'y'))
#         self.position_um['x'], self.position_um['y'] = (x1fin,y1fin)
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
        
        
#         curr_x,curr_y = self.read_position('micron')
#         curr_x = self.volts_to_micron(self.position['x'], 'x')
#         curr_y = self.volts_to_micron(self.position['y'], 'y')
        curr_x, curr_y = self.return_position('um')
        self.go_to_position(curr_x,curr_y)
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
#             self.position['x'], self.position['y'] = self.read_position()
            self.position['x'], self.position['y'] = (self.micron_to_volts(x,'x'),self.micron_to_volts(y,'y'))
#             self.position_um['x'], self.position_um['y'] = (x,y)
            return (curr_x,curr_y)
        else:    
            # go to new location
            self.go_to_position(curr_x+dx,curr_y+dy)
            # set the new current position
#             self.position['x'], self.position['y'] = self.read_position()
            self.position['x'], self.position['y'] = (self.micron_to_volts(x,'x'),self.micron_to_volts(y,'y'))
#             self.position_um['x'], self.position_um['y'] = (x,y)
            return (curr_x+dx,curr_y+dy)                

def scan_1D_cross_section(FSM1,XPS1,center = 0,size = 20,mesh_size=20,zstart=-2,zend=30,zstep=0.4,axis='x',scan_rate=200,save_data = True):
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
    curr_x,curr_y = FSM1.return_position('micron')

    start_plane = center - size/2.0
    end_plane = center + size/2.0

    plane_scan = np.linspace(start_plane, end_plane, mesh_size)
    if zend<=zstart:
        print('zend need to be larger than zstart')
        return 0
    z_scan = np.arange(zstart,zend,zstep)
    mesh_z = len(z_scan)

    FSM1Dcross = np.zeros((mesh_z, mesh_size)) # (rows,columns)
    lp = LivePlotFSM((6, 8), plane_scan, z_scan, FSM1Dcross, axis+' (um)', 'z (um)', 'APD kcts')

#     objective_stage = ConexAGP(obj_port)
#     objective_stage.move_relative(zstart/1000)
    XPS1.move_by(zstart/1000)
    for i, y0 in enumerate(z_scan):
        try:
            if axis == 'x':                    
                FSM1.go_to_position(start_plane, curr_y)
            else:
                FSM1.go_to_position(curr_x, start_plane)
                
            XPS1.move_by(zstep/1000)
            xx, cts = FSM1.scan_1D_axis(start_plane, end_plane, mesh_size, scan_rate,axis)
            FSM1Dcross[mesh_z-1-i,:] = cts
            lp.plot_live(plane_scan, cts, FSM1Dcross)

        except KeyboardInterrupt:
            # press the stop button to trigger this
#             FSMXtask.close()
#             FSMreadXtask.close()
#             countAPD.close()
#             ext_clock_task.close()
            break
#         lp.plot_live(x_scan, cts, FSM2D)
    plt.close(lp.fig)
    # back to initial location
    if axis == 'x':                   
        FSM1.go_to_position(center, curr_y)
    else:
        FSM1.go_to_position(curr_x, center)

    XPS1.move_by(-zend/1000)
#     objective_stage.close()

    if save_data:
        #replot figure
        fig1, ax1 = plt.subplots(1, 1, figsize=(5,5))
        #2D sub plot
        extent_x = (np.max(plane_scan) - np.min(plane_scan)) / 2
        extent_y = (np.max(z_scan) - np.min(z_scan)) / 2
        extent = [np.min(plane_scan), np.max(plane_scan), np.min(z_scan), np.max(z_scan)]
#             aspect_ratio = abs((x_scan[-1] - x_scan[0]) / (y_scan[-1] - y_scan[0]))
        cp1 = ax1.imshow(FSM1Dcross, cmap='gray', extent=extent,
                         interpolation='nearest', aspect=1)
        cb1 = fig1.colorbar(cp1, fraction=0.046, pad=0.04)
        ax1.set_xlabel(axis+' (um)', labelpad=10, fontsize=18)
        ax1.set_ylabel('z (um)', labelpad=10, fontsize=18)
        cb1.set_label('APD counts (kHz)', labelpad=20, fontsize=18)
        cb1.ax.set_yticklabels(cb1.ax.get_yticklabels(), fontsize=16)
        ax1.tick_params(axis='both', labelsize=16)
    #     fig1.subplots_adjust(hspace=.3, left=0.3, bottom=0.15, right = 0.8)
        # rest of the info
        data_type = 'FSM-1D-cross-section'
        save_file_names = data_save_2D(plane_scan, z_scan, FSM1Dcross, fig1, data_type)
    return plane_scan,z_scan,FSM1Dcross


class Bristol:
    """Bristol wavemeter"""
    def __init__(self,COM=7):
        self.COM = COM
        self.handle = self.setupBristol()
        
    def setupBristol(self):
        """
        input - COM number
        set wavelength unit to nm-air, power unit to mW
        return wavemeter handle
        """
        bristol521handle=CLOpenUSBSerialDevice(self.COM)
        CLSetLambdaUnits(bristol521handle,'nm')
        CLSetPowerUnits(bristol521handle,'mw')
        CLSetMedium(bristol521handle,'air')
        return bristol521handle

    def readBristol(self):
        """
        return wavelength, power 
        """
        currentLambda = CLGetLambdaReading(self.handle)
        currentPower = CLGetPowerReading(self.handle)
        return currentLambda,currentPower

def load_and_track(directory,fileprefix,fileext=".dat",radius = 3,minmass=None,preprocess=False):
    """load FSM 2D scan and track
    return dataframe of tracked particles coordinates
    i.e. convert the raw pixel data to coordinates
    """
#     directory = r'C:\data\200203\FSM-2D'
#     fileprefix = '200203FSM54 FSM-2D'
#     fileext = ".dat"
#     filename = '200203FSM54 FSM-2D Z.dat'
    filename = fileprefix +' Z'+fileext
    # fileprefix = 
    # also read from file name X Y Z
    usersDf = pd.read_csv(directory+'\\' + fileprefix + ' Z' + fileext, skiprows=0,delim_whitespace=True,header=None).values
    numX,numY=usersDf.shape
    f = tp.locate(usersDf, radius, invert=False,minmass=minmass,preprocess=preprocess)

    usersDfX = pd.read_csv(directory+'\\' + fileprefix + ' X' + fileext, skiprows=0,delim_whitespace=True,header=None).values
    usersDfY = pd.read_csv(directory+'\\' + fileprefix + ' Y' + fileext, skiprows=0,delim_whitespace=True,header=None).values

    usersDfX = np.reshape(usersDfX,usersDfX.size)
    usersDfY = np.reshape(usersDfY,usersDfY.size)

    if usersDfY[0]<usersDfY[-1]:
        originPosition = 'lower'
    else:
        originPosition = 'upper'
        
    pltstyle = {
      "origin": originPosition
    }
        
    extent = [min(usersDfX[0], usersDfX[-1]),max(usersDfX[0], usersDfX[-1]), min(usersDfY[0], usersDfY[-1]),max(usersDfY[0], usersDfY[-1]) ]
    fig2,ax2 = plt.subplots(figsize=(6,6))

    plt.imshow(usersDf,origin=originPosition,extent=extent,cmap=black_blue_white1)
    # use lifeplot and take into account of the X,Y axis data
#     lp = LivePlot2DV2((6, 8), usersDfX, usersDfY, usersDf, 'x (um)', 'y (um)', 'APD (kct/s)')
#     lp = LivePlot2DV2( usersDfX, usersDfY, usersDf)
    
    # tp.annotate(f, usersDf,plot_style={'markersize':7},imshow_style=pltstyle)
    for i in list(range(  len(f))):

        xcoor = usersDfX[0] + (usersDfX[-1]-usersDfX[0])*(f.iloc[i].x   )/( numX  )
        ycoor = usersDfY[0] + (usersDfY[-1]-usersDfY[0])*(f.iloc[i].y   )/( numY  )
        
        f.at[i,'x'] = xcoor
        f.at[i,'y'] = ycoor
        plt.scatter(xcoor, ycoor, s=100, facecolors='none', edgecolors='r')
        ax2.annotate(str(i),(xcoor,ycoor),fontsize=12,color='black',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
    plt.xlabel('', labelpad=16, fontsize=16)
    plt.ylabel('', labelpad=16, fontsize=16)    
    ax2.tick_params(axis='both', labelsize=16)
    plt.title(directory+"\\"+filename,fontsize=16)
    return f
    # # https://stackoverflow.com/questions/24108063/matplotlib-two-different-colors-in-the-same-annotate/49041502
    
    
def FP_scan(ao_pts_half = 250,v_final = 6.0):
    """
    Fabry Perot triangular scan - 0 V to v_final to 0 V
    scan and acquisition only
    return time series, FP piezo voltage, PD voltage (array)
    """
    ao_trig_src =  '/Dev2/PFI0'
    ai_trig_src =  '/Dev2/PFI1'        
    ao_scan_rate = 5000

    dig_trig_src = '/Dev2/port0/line0'
    piezo_src = '/Dev2/ao0'
    pd_src = '/Dev2/ai0'

    a = np.linspace(0,v_final,ao_pts_half) 
    b = np.linspace(v_final,0,ao_pts_half)
    v_ao_scan = np.concatenate((a,b),axis=None)
    ao_pts = ao_pts_half*2

    # time
    t0 = np.linspace(0,(ao_pts-1)/ao_scan_rate,ao_pts)

    dig_trig = DO(dig_trig_src)
    dig_trig.write(False)

    # set up  AO AI
    FPtask = AO(piezo_src)
    cavityPDtask = AI(pd_src)

    FPtask.config_write(v_ao_scan,ao_scan_rate,ao_trig_src)
    cavityPDtask.config_read_rising(ao_pts,ao_scan_rate,ai_trig_src)

    # start FP scan
    dig_trig.write(True)
    FPtask.wait_until_done()
    aiV = cavityPDtask.read(number_of_samples_per_channel=ao_pts)

    FPtask.close()
    cavityPDtask.close()   
    dig_trig.write(False)
    dig_trig.close()
    return t0,v_ao_scan,aiV

def FP_fit(tarray,piezo, PD, userange=250):
    """
    fit a lorentzian + sine to the input
    assume piezo is either linearly increasing or decreasing
    only use the wave in [0:userange-1]    
    return (amp, cen, fwhm, bkg, asin,fsin,phisin)
    """
    npts = min(len(piezo),userange)
    vrange = np.amax(piezo[0:npts-1])-np.amin(piezo[0:npts-1])
    scan_rate = 1/(tarray[1]-tarray[0])
    init_vals = [np.amin(PD), piezo[np.argmin(PD)], 0.2, 0.05,0.1,60*npts/scan_rate/vrange,1]
    try:
        best_vals, covar = curve_fit(lorentziansin, piezo[0:npts-1], PD[0:npts-1], p0=init_vals)
        return best_vals
    except:
        print('Fit error')
        return init_vals*0        
    
def toptica_bounded_write(v,scan_terminal='/Dev2/ao1'):
    """clamp the output voltage v [Volt] to be -5 <= v <= +5
    scan_terminal - laser scan
    """
    with nidaqmx.Task() as task_toptica:
        task_toptica.ao_channels.add_ao_voltage_chan(scan_terminal, 'Piezo scan', min_val=-5, max_val=5)
        vup = 5
        vdown = -5
        v_bound = max(  min(vup,v),vdown    )    
        task_toptica.write(v, auto_start=True, timeout=5) # write a single data   
        
# piecewise linear fit
# https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python
# https://github.com/cjekel/piecewise_linear_fit_py 
# though this is for a CONTINUOUS piecewise linear fit
# python a[n1:n2] := a[n1],a[n1+1],...,a[n2-1]
def fit_wavelength_segments(directory,fileprefix,fileext=".txt",skiprows=5):
    """ https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
    fit wavelength scan into linear segments
    """
    filename = fileprefix +fileext
    usersDf = pd.read_csv(directory+'\\' + fileprefix  + fileext, skiprows=skiprows,delimiter =' ',header=None).values
    
    piezoVlist = usersDf[:,0]
    lambdalist = usersDf[:,1]
    lambdaFPlist = usersDf[:,3]
    mode_hop_free=0
    mode_hop_free = np.empty([0,7]) 
#     lpPiezo = LivePlot(1,1,8,5,'o','Piezo control (V)','FP peak (V)')
#     lpPiezo.plot_live(piezoVlist,lambdaFPlist)
#     lpFit = LivePlot(1, 1, 8, 5, 'o', 'Piezo control (V)',"Wavelength (nm-air)")
#     lpFit.plot_live(piezoVlist,lambdalist)
    
    lpLR = LivePlotLR(1,1,8,5,'o','Piezo control (V)',"Wavelength (nm-air)",'FP peak (V)')
    lpLR.plot_live(piezoVlist,lambdalist,lambdaFPlist)
    
    steps = lambdaFPlist[1:]-lambdaFPlist[:-1] 
    b = abs(steps-np.percentile(steps,50))  > 0.5*abs(np.percentile(steps,50))

    ptlist=np.append(np.insert(np.nonzero(b)[0],0,-1),len(lambdaFPlist)-1)
#     print(ptlist)
    for ind,v in enumerate(ptlist[:-1]):
        startpt =  ptlist[ind]+1
        endpt = ptlist[ind+1]

        startfit = startpt +1
        endfit = endpt -1 
        if (endpt-startpt)<4:
#             print(endpt)
            continue
        else:
            x = piezoVlist[startpt:endpt+1]
            y = lambdalist[startpt:endpt+1]
            yr = lambdaFPlist[startpt:endpt+1]

            xfit = piezoVlist[startfit:endfit+1]
            yfit = lambdalist[startfit:endfit+1]
            yrfit = lambdaFPlist[startfit:endfit+1]

            coef = np.polyfit(xfit,yfit, 1)
            poly1d_fn = np.poly1d(coef)
            coefr = np.polyfit(xfit,yrfit, 1)
            poly1d_fnr = np.poly1d(coefr)
            
#             plt.plot(x, poly1d_fn(x), '--r')

            lpLR.ax1.plot(xfit, poly1d_fn(xfit), '--r')
            lpLR.ax1r.plot(xfit, poly1d_fnr(xfit), '--b')
#             print(f'{poly1d_fn(xfit[0]): .4f},{poly1d_fn(xfit[-1]): .4f}, {poly1d_fn(xfit[0])-poly1d_fn(xfit[-1]):.4f},{xfit[0]: .3f},{xfit[-1]: .3f},{coef[0]:.4f},{coef[1]:.3f}')
#             print(endpt)
            print(f'{poly1d_fn(x[0]): .4f},{poly1d_fn(x[-1]): .4f}, {poly1d_fn(x[0])-poly1d_fn(x[-1]):.4f},{x[0]: .3f},{x[-1]: .3f},{coef[0]:.4f},{coef[1]:.3f}')
            a=np.asarray([[poly1d_fn(x[0]),poly1d_fn(x[-1]),poly1d_fn(x[0])-poly1d_fn(x[-1]),x[0],x[-1],coef[0],coef[1]]])

            mode_hop_free=np.append(mode_hop_free,a,axis=0)
        
            
#             print(a)
#             mode_hop_free=np.append(mode_hop_free,a,axis=0)
    
    mid80 = np.percentile(lambdalist,90)-np.percentile(lambdalist,10)
    mid80r = np.percentile(lambdaFPlist,90)-np.percentile(lambdaFPlist,10)
    if ind>=3:    
        lpLR.ax1.set_ylim([np.percentile(lambdalist,10)-mid80/3,np.percentile(lambdalist,90)+mid80/3])
        lpLR.ax1r.set_ylim([np.percentile(lambdaFPlist,10)-mid80r/3,np.percentile(lambdaFPlist,90)+mid80r/3])
    lpLR.fig.tight_layout()
    return lpLR,mode_hop_free

def monitor_wavelength(bristol):
    """Monitor wavelength given a bristol wavemeter instance"""
    lambdalist=[]
    powerlist=[]
    t0 = []
    timestep = 0.1 # sec
    i=0.0
    lpLR = LivePlotLR(1, 1, 8, 5, 'o', 'Time (s)', 'Wavelength (nm-air)','Bristol power (mW)')
    tinit = time.time()
    while True:
        try:
            currentLambda,currentPower = bristol.readBristol()
            lambdalist.append(currentLambda)
            powerlist.append(currentPower)
            t0.append(time.time()-tinit)
            lpLR.plot_live(t0, lambdalist,powerlist,f'Mean = {np.mean(lambdalist): .4f}\n Std = {np.std(lambdalist): .4f}',fontsize=40)
    #         lpLR.ax1.set_ylim([736.85,736.87])
            time.sleep(timestep)
        except:
#             print("End")
            break
    return t0,lambdalist,powerlist

def scan_laser_piezo(bristol,PM,vstart=-5,vend=5,scanpts = 50,PowerMeterOn=False,save_data=True,lasercurrent=np.nan,potreading=np.nan):
    """scan laser piezo voltage and monitor wavelength and power
    scan laser frequency in discrete steps
    software timed
    """
    toptica_bounded_write(vstart)
    voltage = np.linspace(vstart,vend,scanpts)
    FP_FSR_V = 4.783
    lpFP = LivePlot(1, 2, 5, 3, 'o', 'Time (s)',"PD (V)")
    # lpLRFP = LivePlotLR(1, 1, 8, 5, 'o', 'Laser piezo (V)', 'Peak center (V)','Peak amplitude (V)')
    lpLRFP = LivePlotLR(1, 1, 8, 5, 'o', 'Laser piezo (V)', 'Peak center (V)','Peak center (V)')

    if PowerMeterOn:
        lpLR = LivePlotLR(1, 1, 8, 5, 'o', 'Laser piezo (V)', 'Wavelength (nm-air)','Thorlabs power (mW)')
    else:
        lpLR = LivePlotLR(1, 1, 8, 5, 'o', 'Laser piezo (V)', 'Wavelength (nm-air)','Bristol power (mW)')

    volt_so_far=[]
    powerlist=[]
    lambdalist=[]
    thorpowerlist = []

    FPlambda=[]
    FPpower=[]
    for ind,v in enumerate(voltage):
        toptica_bounded_write(v)
        volt_so_far.append(v)

        # read fabry perot - need to start the read and write channel at the same time
        t0_this,v_ao_scan_this,aiV_this = FP_scan()
        best_vals = FP_fit(t0_this[0:249],v_ao_scan_this[0:249], aiV_this[0:249])
        if ind==0:
            FPlambda.append(best_vals[1])
        else:
            FPlambda.append(unwrap_by_pt(best_vals[1],FPlambda[ind-1],FP_FSR_V))

        FPpower.append(-best_vals[0])

        yfit = lorentziansin(v_ao_scan_this, best_vals[0], best_vals[1],best_vals[2],best_vals[3],best_vals[4],best_vals[5],best_vals[6])
        lpFP.plot_live(t0_this,aiV_this,yfit)
        lpFP.ax1.set_title(f'Center at {best_vals[1]:.3f} V')
        plt.tight_layout()
        time.sleep(0.1)

    #     lpLRFP.plot_live(volt_so_far, FPlambda,FPpower)
        lpLRFP.plot_live(volt_so_far, FPlambda,FPlambda)

        # read wavemeter
        currentLambda,currentPower = bristol.readBristol()
        lambdalist.append(currentLambda)
        powerlist.append(currentPower)

        # read Thorlabs powermeter
        if PowerMeterOn:        
            currentPMpower = PM.measure_power(737)*1e3
            thorpowerlist.append(currentPMpower)
            lpLR.plot_live(volt_so_far, lambdalist,thorpowerlist)
            lpLR.ax1r.set_ylim([0,1.1*max(thorpowerlist)])
        else:
            thorpowerlist.append(np.nan)
            lpLR.plot_live(volt_so_far, lambdalist,powerlist)
            lpLR.ax1r.set_ylim([0,1.1*max(powerlist)])

        mid80 = np.percentile(lambdalist,90)-np.percentile(lambdalist,10)
        if ind>=3:    
            lpLR.ax1.set_ylim([np.percentile(lambdalist,10)-mid80/3,np.percentile(lambdalist,90)+mid80/3])

    toptica_bounded_write(0)

#     save_data = True
    if save_data:
        data_type = 'TopticaScan'

        data_header=f"""
        current (mA, at 0V) =  {lasercurrent}
        pot reading = {potreading}
        volt,lambda (nm-air),power (mW),FP lambda (V),FP power (V), Thorlabs power (mW)
        """
        data_array = np.array([volt_so_far, lambdalist,powerlist,FPlambda,FPpower,thorpowerlist]).T
        data_save(data_array, lpLR.fig, data_type, data_header)