import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from scipy.special import erf
from more_itertools import locate

# ##################################   General constants   ##############################################
FS_LABEL = 22
FS_TICKS = 18

# ##################################   General classes   #################################################


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

    def plot_live(self, xdata, ydata, ydata2=None):
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

        self.fig.canvas.draw()
        plt.pause(1e-7)


class LivePlot2D:
    def __init__(self, x_data, y_data, z_data, x_ext=18, y_ext=6):
        self.fig = plt.figure(figsize=(x_ext, y_ext))
        self.ax = self.fig.add_subplot(111)
        self.extent = [np.min(x_data), np.max(x_data), np.min(y_data), np.max(y_data)]
        self.fig.show()
        aspect_ratio = (x_data[-1] - x_data[0]) / (y_data[-1] - y_data[0]) / 2
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

# ##################################   General functions   ##############################################


def data_save(array, figure, type, header):
    # fetch date and time
    now = datetime.datetime.now()
    date_str = now.strftime("%y%m%d %H%M")[:6]
    time_str = now.strftime("%y%m%d %H%M%S")[-6:]
    save_data_dir = r'C:\data\%s\%s' % (date_str, type)
    save_data_file = r'%s\%s_%s' % (save_data_dir, type, time_str)

    if not os.path.isdir(save_data_dir):
        os.makedirs(save_data_dir)
    figure.savefig(save_data_file + '.png', format='png', bbox_inches='tight')
    np.savetxt(save_data_file + '.txt', array, header=header)


def get_work_dir():
    # this function is useful when working from different computers in Dropbox, Google Drive...
    wd = os.getcwd()
    idx = list(locate(wd, lambda x: x == "\\"))
    return wd[:idx[-1]]


def prettify_plot(fig_ax, xlabel='x', ylabel='y'):
    fig_ax.set_xlabel(xlabel, labelpad=10, fontsize=FS_LABEL)
    fig_ax.set_ylabel(ylabel, labelpad=10, fontsize=FS_LABEL)
    fig_ax.tick_params(axis='both', labelsize=FS_TICKS)


def prettify_2d_plot(fig_ax, fig_cb, xlabel='x', ylabel='y', zlabel='z'):
    fig_ax.set_xlabel(xlabel, labelpad=10, fontsize=FS_LABEL)
    fig_ax.set_ylabel(ylabel, labelpad=10, fontsize=FS_LABEL)
    fig_cb.set_label(zlabel, labelpad=20, fontsize=FS_LABEL)
    fig_cb.ax.set_yticklabels(fig_cb.ax.get_yticklabels(), fontsize=FS_TICKS)
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


def lorentzian_func(x_array, a0, x0, gamma):
    # I0 = something proportional to the height
    # x0 = peak position
    # gamma = FWHM
    # bg = background
    return a0 * gamma / (np.pi * ((x_array - x0) ** 2 + gamma ** 2))


def gaussian_func(x_array, a0, x0, sigma):
    return a0 * np.exp(-(x_array - x0) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


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