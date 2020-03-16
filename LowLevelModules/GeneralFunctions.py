import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import datetime
import os
from scipy.special import erf
from more_itertools import locate
from scipy import optimize
from LowLevelModules.ImportSDT import SdtFile

# ##################################   General constants   ##############################################
FS_LABEL = 22
FS_TICKS = 18

cdict1 = {'red':   [[0.0,  0.0, 0.0],
                   [0.55,  0.0, 0.0],
                   [0.8, 1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.55, 0.0, 0.0],
                   [0.8, 1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.55,  1.0, 1.0],
                   [0.8, 1.0, 1.0],
                   [1.0,  1.0, 1.0]]}

black_blue_white1 = LinearSegmentedColormap('BlackBlueWhite1', cdict1)
# ##################################   General classes   #################################################
class MyPlot:
    def __init__(self, nl, nr, fs=(8,6), xlabel='', ylabel=''):
        self.fig, self.ax = plt.subplots(nl, nr, figsize=fs)
        self.ax.set_xlabel(xlabel, labelpad=15, fontsize=22)
        self.ax.set_ylabel(ylabel, labelpad=15, fontsize=22)
        self.ax.tick_params(direction='in', axis='both', top=True, right=True, labelsize=18, pad=5, length=6, width=1)
        plt.tight_layout()
    
    def xlim(self, x0=None, xf=None):
        if x0 is not None:
            self.ax.set_xlim([x0, xf])
        else:
            x_lim = self.ax.get_lines()[0].get_xdata()
            self.ax.set_xlim([x_lim[0], x_lim[-1]])
    
    def plot(self, x, y, mc, **kwargs):
        self.ax.plot(x, y, mc, **kwargs)
        self.xlim()
        plt.tight_layout()
    
    def semilogy(self, x, y, mc, **kwargs):
        self.plot(x, y, mc, **kwargs)
        self.ax.set_yscale('log')
        
    def savefig(self, path):
        self.fig.savefig(path, format='png', bbox_inches='tight')
    

class TRMData(SdtFile):
    def __init__(self, path):
        super().__init__(path)
        self.time = self.times[0]
        self.info = self.measure_info[0]
        self.x = self.time*1e9
        self.y = self.data[0][0]
        self.bin_size = (self.time[-1]-self.time[0])/(len(self.time)-1)

        #trim data
        self.x = self.x[self.y>0]
        self.y = self.y[self.y>0]
        trim = -1 * int(len(self.y)*0.04)
        self.x = self.x[:trim]
        self.y = self.y[:trim]
        
        #shift x axis
        self.x -= self.x[0]

        #count rate
        info_ctr = self.info['StopInfo']
        min_sync_rate = info_ctr['min_sync_rate'][0]
        min_cfd_rate = info_ctr['min_cfd_rate'][0]
        min_tac_rate = info_ctr['min_tac_rate'][0]
        max_sync_rate = info_ctr['max_sync_rate'][0]
        max_cfd_rate = info_ctr['max_cfd_rate'][0]
        max_tac_rate = info_ctr['max_tac_rate'][0]
        self.col_t = info_ctr['stop_time'][0]

        self.APD1_rate = (min_cfd_rate + max_cfd_rate) / 2
        self.APD1_rate_alt = (min_tac_rate + max_tac_rate) / 2
        self.laser_rep_rate = (min_sync_rate + max_sync_rate) / 2
        self.APD2_rate = self.laser_rep_rate
        
        #normalization
        self.total_counts = np.sum(self.y)        

class LivePlot:
    def __init__(self, subpl=2, num_y_data=1, x_ext=10, y_ext=6, mk='o', xlabel='x', ylabel='y'):
        self.fig = plt.figure(figsize=(x_ext, y_ext))
        self.subpl_num = subpl
        if subpl == 1:
            self.ax1 = self.fig.add_subplot(111)
        else:
            self.ax1 = self.fig.add_subplot(121)
        plt.plot([], [], marker=mk)
        if num_y_data == 2:
            plt.plot([], [], marker=mk)

        plt.xlabel(xlabel, labelpad=10, fontsize=FS_LABEL)
        plt.ylabel(ylabel, labelpad=10, fontsize=FS_LABEL)
        self.ax1.tick_params(axis='both', labelsize=FS_TICKS)

        if subpl > 1:
            self.ax2 = self.fig.add_subplot(122, polar=True)
            plt.plot([], [], marker=mk)
            plt.xlabel('', labelpad=10, fontsize=FS_LABEL)
            plt.ylabel('', labelpad=10, fontsize=FS_LABEL)
            self.ax2.tick_params(axis='both', labelsize=14)

        self.fig.show()
        self.fig.canvas.draw()
        plt.tight_layout()

    def plot_live(self, xdata, ydata, ydata2=None,title1=''):
        self.ax1.lines[0].set_xdata(xdata)
        self.ax1.lines[0].set_ydata(ydata)
        if ydata2 is not None:
            self.ax1.lines[1].set_xdata(xdata)
            self.ax1.lines[1].set_ydata(ydata2)
        self.ax1.relim()
        self.ax1.autoscale_view()

        if self.subpl_num > 1:
            xdata = [xx * np.pi / 180 for xx in xdata]
            self.ax2.lines[0].set_xdata(xdata)
            self.ax2.lines[0].set_ydata(ydata)
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.ax1.set_title(title1)
        self.fig.canvas.draw()
        plt.pause(1e-7)


class LivePlotLR:
    """
    plot on left and right axes
    """
    def __init__(self, subpl=1, num_y_data=1, x_ext=10, y_ext=6, mk='bo', xlabel='x', ylabel='y',yrlabel='yr'):
        self.fig = plt.figure(figsize=(x_ext, y_ext))
        self.subpl_num = subpl
        if subpl == 1:
            self.ax1 = self.fig.add_subplot(111)
        else:
            self.ax1 = self.fig.add_subplot(121)
        self.ax1.plot([], [], marker=mk)
        if num_y_data == 2:
            self.ax1.plot([], [], marker=mk)

        self.ax1r = self.ax1.twinx()
#         plt.xlabel(xlabel, labelpad=10, fontsize=FS_LABEL)
#         plt.ylabel(ylabel, labelpad=10, fontsize=FS_LABEL)
        self.ax1.set_xlabel( xlabel,labelpad=10, fontsize=FS_LABEL)
        self.ax1.set_ylabel( ylabel, color='b',labelpad=10, fontsize=FS_LABEL)
        self.ax1r.set_ylabel( yrlabel, color='g',labelpad=10, fontsize=FS_LABEL)
        self.ax1.tick_params(axis='both', labelsize=FS_TICKS)
        self.ax1r.tick_params(axis='both', labelsize=FS_TICKS)
        self.ax1r.plot([],[],'g.-')


        if subpl > 1:
            self.ax2 = self.fig.add_subplot(122, polar=True)
            plt.plot([], [], marker=mk)
            plt.xlabel('', labelpad=10, fontsize=FS_LABEL)
            plt.ylabel('', labelpad=10, fontsize=FS_LABEL)
            self.ax2.tick_params(axis='both', labelsize=14)

        self.fig.show()
        self.fig.canvas.draw()
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def plot_live(self, xdata, ydata, yrdata=None,title1='',fontsize=60):
        self.ax1.lines[0].set_xdata(xdata)
        self.ax1.lines[0].set_ydata(ydata)
        if yrdata is not None:
            self.ax1r.lines[0].set_xdata(xdata)
            self.ax1r.lines[0].set_ydata(yrdata)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax1r.relim()
        self.ax1r.autoscale_view()

        if self.subpl_num > 1:
            xdata = [xx * np.pi / 180 for xx in xdata]
            self.ax2.lines[0].set_xdata(xdata)
            self.ax2.lines[0].set_ydata(ydata)
            self.ax2.relim()
            self.ax2.autoscale_view()
            
        self.ax1.set_title(title1,fontsize=fontsize)
        self.fig.canvas.draw()
        self.fig.tight_layout()
        plt.pause(1e-7)

class LivePlot2D:
    def __init__(self, x_data, y_data, z_data, x_ext=18, y_ext=6):
        self.fig = plt.figure(figsize=(x_ext, y_ext))
        self.ax = self.fig.add_subplot(111)
        self.extent = [np.min(x_data), np.max(x_data), np.min(y_data), np.max(y_data)]
        self.fig.show()
        aspect_ratio = abs((x_data[-1] - x_data[0]) / (y_data[-1] - y_data[0]) )
        self.cp = self.ax.imshow(z_data, cmap='jet', origin='center', extent=self.extent,
                                 interpolation='nearest', aspect=aspect_ratio)

        self.cb = self.fig.colorbar(self.cp, fraction=0.046/2, pad=0.04)
        self.fig.canvas.draw()
        self.fig.tight_layout()

    def plot_live(self, z_data):
        self.cp.set_data(z_data)
        self.cb.set_clim(vmin=np.min(z_data),vmax=np.max(z_data))
        self.cb.draw_all()
        self.fig.canvas.draw()
        self.fig.tight_layout()
        plt.pause(1e-6)


class LivePlot2DV2:
    def __init__(self, x_data, y_data, z_data, x_ext=18, y_ext=6,cmap=black_blue_white1):
        self.fig = plt.figure(figsize=(x_ext, y_ext))
        self.ax = self.fig.add_subplot(111)
        self.extent = [np.min(x_data), np.max(x_data), np.min(y_data), np.max(y_data)]
        self.fig.show()
        self.bluecmap = black_blue_white1
        self.x_data= x_data
        aspect_ratio = abs((x_data[-1] - x_data[0]) / (y_data[-1] - y_data[0]) )
        self.cp = self.ax.imshow(z_data, cmap=cmap ,
                                 interpolation='nearest',origin='center',extent=self.extent, aspect=aspect_ratio)
# origin='center', extent=self.extent
        self.cb = self.fig.colorbar(self.cp, fraction=0.046/2, pad=0.04)
        self.fig.canvas.draw()
        self.fig.tight_layout()

    def plot_live(self, z_data,y_data=None,cmap=black_blue_white1):
        if y_data is not None:
            self.extent = [self.extent[0], self.extent[1], np.min(y_data), np.max(y_data)]
            self.cp.set_extent(self.extent)
            aspect_ratio = abs((self.x_data[-1] - self.x_data[0]) / (y_data[-1] - y_data[0]) )            
            self.ax.set_aspect( aspect_ratio  )
            
        self.cp.set_data(z_data)
        self.cNorm      = matplotlib.colors.Normalize(vmin=np.min(z_data),vmax=np.max(z_data))
        self.scalarMap  = matplotlib.cm.ScalarMappable(norm=self.cNorm, cmap=cmap )
#         plt.get_cmap('gray')
        self.cb.update_normal(self.scalarMap)
        self.cp.set_norm(self.cNorm)
        self.cb.draw_all()
        self.fig.canvas.draw()
        self.fig.tight_layout()
        plt.pause(1e-6)

class LivePlotFSM:
    def __init__(self, dims, x_data, y_data, z_data, xlabel='x', ylabel='y', zlabel='Cts'):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1,
                                                      #sharex=True,
                                                      figsize=dims,
                                                      gridspec_kw={'height_ratios': [1, 2]})
        #current X scan in 1D
        self.ax1.plot(x_data, z_data[0,:])
        self.ax1.set_xlabel('', labelpad=10, fontsize=18)
        self.ax1.set_ylabel(zlabel, labelpad=10, fontsize=18)
        self.ax1.tick_params(axis='both', labelsize=16)
        
        self.bluecmap = black_blue_white1
        
        #2D sub plot
        extent_x = (np.max(x_data) - np.min(x_data)) / 2
        extent_y = (np.max(y_data) - np.min(y_data)) / 2
        self.extent = [np.min(x_data), np.max(x_data), np.min(y_data), np.max(y_data)]
#         aspect_ratio = abs((x_data[-1] - x_data[0]) / (y_data[-1] - y_data[0]))
        self.cp2 = self.ax2.imshow(z_data, cmap=self.bluecmap, extent=self.extent,
                                 interpolation='nearest', aspect=1)

        self.cb2 = self.fig.colorbar(self.cp2, fraction=0.046, pad=0.04)
        self.ax2.set_xlabel(xlabel, labelpad=10, fontsize=18)
        self.ax2.set_ylabel(ylabel, labelpad=10, fontsize=18)
        self.cb2.set_label(zlabel, labelpad=20, fontsize=18)
        self.cb2.ax.set_yticklabels(self.cb2.ax.get_yticklabels(), fontsize=16)
        self.ax2.tick_params(axis='both', labelsize=16)

        self.fig.show()
        self.fig.canvas.draw()
        self.fig.subplots_adjust(hspace=.3, left=0.15, bottom=0.15, right = 0.8)
        #self.fig.tight_layout()

    def plot_live(self, x_data, y_data, z_data):
        self.ax1.lines[0].set_xdata(x_data)
        self.ax1.lines[0].set_ydata(y_data)
        self.ax1.relim()
        self.ax1.autoscale_view()

        self.cp2.set_data(z_data)
        self.cNorm      = matplotlib.colors.Normalize(vmin=np.min(z_data),vmax=np.max(z_data))
        self.scalarMap  = matplotlib.cm.ScalarMappable(norm=self.cNorm, cmap=self.bluecmap )
#         plt.get_cmap('gray')
        self.cb2.update_normal(self.scalarMap)
        self.cp2.set_norm(self.cNorm)

        self.cb2.draw_all()
        self.fig.canvas.draw()
        #self.fig.tight_layout()
        plt.pause(1e-7)

        
# ##################################   General functions   ##############################################

def fetch_date_and_make_folder(data_type):
    now = datetime.datetime.now()
    date_str = now.strftime("%y%m%d %H%M")[:6]
    time_str = now.strftime("%y%m%d %H%M%S")[-6:]
    output_dir = r'C:\data\%s\%s' % (date_str, data_type)
    output_data = r'%s\%s_%s' % (output_dir, data_type, time_str)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    return output_data

def data_save(array, figure=None, data_type=None, header=None):
    save_data_file = fetch_date_and_make_folder(data_type)
    if figure is not None:
        figure.savefig(save_data_file + '.png', format='png', bbox_inches='tight')
    np.savetxt(save_data_file + '.txt', array, header=header)
    print(save_data_file + '.txt')
    return save_data_file

def data_save_2D(x_data, y_data, z_data, figure=None, data_type=None):
    save_data_file = fetch_date_and_make_folder(data_type)
    if figure is not None:
        figure.savefig(save_data_file + '.png', format='png', bbox_inches='tight')
    np.savetxt(save_data_file + ' X.txt', x_data)
    np.savetxt(save_data_file + ' Y.txt', y_data)
    np.savetxt(save_data_file + ' Z.txt', z_data)
    print(save_data_file + ' Z.txt')

def get_work_dir():
    # this function is useful when working from different computers in Dropbox, Google Drive...
    wd = os.getcwd()
    idx = list(locate(wd, lambda x: x == "\\"))
    return wd[:idx[-1]]


def prettify_plot(fig_ax, xlabel='x', ylabel='y'):
    fig_ax.set_xlabel(xlabel, labelpad=10, fontsize=FS_LABEL)
    fig_ax.set_ylabel(ylabel, labelpad=10, fontsize=FS_LABEL)
    fig_ax.tick_params(axis='both', labelsize=FS_TICKS)
    x_lim = fig_ax.get_lines()[0].get_xdata()
    fig_ax.set_xlim([x_lim[0], x_lim[-1]])



def prettify_2d_plot(fig_ax, fig_cb, xlabel='x', ylabel='y', zlabel='z'):
    fig_ax.set_xlabel(xlabel, labelpad=10, fontsize=FS_LABEL)
    fig_ax.set_ylabel(ylabel, labelpad=10, fontsize=FS_LABEL)
    fig_cb.set_label(zlabel, labelpad=20, fontsize=FS_LABEL)
    fig_cb.ax.tick_params(labelsize=FS_TICKS)
    fig_ax.tick_params(axis='both', labelsize=FS_TICKS)


def prettify_3d_plot(fig_ax, fig_cb, xlabel='x', ylabel='y', zlabel='z'):
    fig_ax.set_xlabel(xlabel, labelpad=15, fontsize=FS_LABEL)
    fig_ax.set_ylabel(ylabel, labelpad=20, fontsize=FS_LABEL)
    fig_cb.set_label(zlabel, labelpad=20, fontsize=FS_LABEL)
    fig_cb.ax.set_yticklabels(fig_cb.ax.get_yticklabels(), fontsize=FS_TICKS)
    fig_ax.tick_params(axis='both', labelsize=FS_TICKS)
    fig_ax.set_zticks([])
    fig_ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    fig_ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    fig_ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    fig_ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    fig_ax.grid(False)

def line(x,  a, b):
    return a*x+b

def unwrap_custom(arrayin,period):
    """unwrap with custom period"""
    arrayout =period/2/np.pi*np.unwrap(np.array(arrayin)*2*np.pi/period)
    return arrayout

def unwrap_by_pt(curr_pt,prev_pt,period):
    """unwrap with custom period"""
    return curr_pt-round((curr_pt-prev_pt)/period)*period

def lorentzian_func(x_array, a0, x0, gamma):
    # I0 = something proportional to the height
    # x0 = peak position
    # gamma = FWHM
    # bg = background
    return a0 * gamma / (np.pi * ((x_array - x0) ** 2 + gamma ** 2))


def lorentzian_bkg_func(x_array, a0, x0, fwhm,bkg):
    return a0  / ( 1+4*( (x_array-x0)/fwhm )**2   )+bkg

def lorentzian_discrete_func(x_array, aTot, x0, fwhm,bkg,step):
    """
    discretized with a step size
    """
    return bkg*step  +aTot/np.pi *(np.arctan(2*(x_array-x0+step/2)/fwhm)-np.arctan(2*(x_array-x0-step/2)/fwhm))

def lorentziansin(x,  amp, cen, fwhm, bkg, asin,fsin,phisin):
    """
    Lorentizan plus a sine wave - for fitting Fabry perot response in the presence of 60 Hz noise
    """
    return amp / (1+   ( 2*(x-cen)/fwhm )**2   ) + asin*np.sin(2*np.pi*fsin*x+phisin) +bkg

def lorentzianFSR(x,amp,cen,fwhm,bkg,x_FSR):
    """ sum of multiple lorentzian"""
    
    a1 = amp / (1+   ( 2*(x-cen-2*x_FSR)/fwhm )**2   )
    a2 = amp / (1+   ( 2*(x-cen-x_FSR)/fwhm )**2   )
    a3 = amp / (1+   ( 2*(x-cen)/fwhm )**2   )
    a4 = amp / (1+   ( 2*(x-cen+x_FSR)/fwhm )**2   )
    a5 = amp / (1+   ( 2*(x-cen+2*x_FSR)/fwhm )**2   )
    
    return a1+a2+a3+a4+a5 +bkg
    
    
def gaussian_func(x_array, a0, x0, sigma):
    return a0 * np.exp(-(x_array - x0) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)

def gaussian_bkg_func(x_array, a0, x0, sigma,bkg):
    return bkg+a0 * np.exp(-(x_array - x0) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


def voigt_func(x, a0, x0, sigma, gamma):
    # Voigt function = line profile of a Gaussian distribution convoluted with a Lorentzian distribution.
    # input coeff should be one array [height, position, gaussian width sigma, lorentzian width gamma]

    z1 = (x - x0 + 1j * gamma) / (sigma * np.sqrt(2))
    arg1 = np.exp(-z1 ** 2) * (1 - erf(-1j * z1))
    # genuine definition of Voigt function
    f1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.real(arg1)

    return a0 * f1 / np.max(f1)


def voigt_func_2p(x, a01, x01, sigma1, gamma1, a02, x02, sigma2, gamma2):
    # Voigt function = line profile of a Gaussian distribution convoluted with a Lorentzian distribution.
    # input coeff should be two line vector

    z1 = (x - x01 + 1j * gamma1) / (sigma1 * np.sqrt(2))
    z2 = (x - x02 + 1j * gamma2) / (sigma2 * np.sqrt(2))

    arg1 = np.exp(-z1 ** 2) * (1 - erf(-1j * z1))
    arg2 = np.exp(-z2 ** 2) * (1 - erf(-1j * z2))

    # genuine definition of Voigt function
    f1 = 1 / (sigma1 * np.sqrt(2 * np.pi)) * np.real(arg1)
    f2 = 1 / (sigma2 * np.sqrt(2 * np.pi)) * np.real(arg2)

    # row = isnan(F)
    # n_row = find(row == 1)
    # if numel(n_row) ~= 0:
    # F(n_row(1):n_row(end)) = 0
    return a01 * f1 / np.max(f1) + a02 * f2 / np.max(f2)
    #return voigt_func(x, a01, x01, sigma1, gamma1) + voigt_func(x, a02, x02, sigma2, gamma2)



def gaussian2D(height, center_x, center_y, width_x, width_y,bkg):
    """Returns a gaussian function with the given parameters
    width_x, width_y := standard deviation
    """
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: bkg+height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments2D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    bkg = data.min()
    return height, x, y, width_x, width_y,bkg

def fitgaussian2D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments2D(data)
    errorfunction = lambda p: np.ravel(gaussian2D(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def pulse_wfm(t,tstart,width):
    """a 0-1 pulse that is on in [tstart,tend]
    return the pulse value at time t
    """
    return np.heaviside(t-tstart,0)-np.heaviside(t-tstart-width,0)
