import igor.binarywave as bw
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("..")
from LowLevelModules.GeneralFunctions import *


class AFMdata:
    def __init__(self, path, rotate=0):

        self.file = bw.load(path)
        self.data = self.file['wave']
        self.signal = self.data['wData']
        self.note = self.data['note']
        self.header = self.data['wave_header']
        self.name = str(self.header['bname'].decode('utf-8'))
        self.units = "".join([x.decode() for x in self.header['dataUnits']])
        # parsing to get the scan size
        self.info = self.note[:500].decode().split('\r')
        self.scan_size = float(self.info[0].split(' ')[1])
        self.scan_size *= 1e6  # micrometer

        self.height = self.signal[:, :, 0].T * 1e9
        self.amplitude = self.signal[:, :, 1].T
        self.phase = self.signal[:, :, 2].T
        self.plot_unit = {'height': ' Height (nm)', 'amplitude': 'Amplitude (%)'}
        # make x and y arrays
        self.dim = self.height.shape
        self.X = np.linspace(-self.scan_size / 2, self.scan_size / 2, self.dim[0])
        self.Y = np.linspace(-self.scan_size / 2, self.scan_size / 2, self.dim[1])
        if rotate > 0:
            self.height = np.rot90(self.height, rotate)
            self.amplitude = np.rot90(self.amplitude, rotate)
            self.phase = np.rot90(self.phase, rotate)

    def plot2D(self, case='height'):
        self.fig1 = plt.figure(figsize=(7, 6))
        self.ax1 = self.fig1.add_subplot(111)
        extent = np.array([-.5, .5, -.5, .5]) * self.scan_size
        self.cp1 = self.ax1.imshow(getattr(self, case), cmap='terrain', origin='center', extent=extent)
        self.cb1 = plt.colorbar(self.cp1, fraction=0.046, pad=0.04)
        prettify_2d_plot(self.ax1, self.cb1, 'X (µm)', 'Y (µm)', self.plot_unit[case])
        self.fig1.tight_layout()

    def plot3D(self, case='height'):
        self.fig2 = plt.figure(figsize=(8, 6))
        self.ax2 = self.fig2.gca(projection='3d')
        self.ax2.view_init(elev=30., azim=225)

        XX, YY = np.meshgrid(self.X, self.Y)
        surf = self.ax2.plot_surface(XX, YY, getattr(self, case), cmap='afmhot')
        self.cb2 = self.fig2.colorbar(surf, shrink=.8, fraction=0.046, pad=0.04)
        prettify_3d_plot(self.ax2, self.cb2, 'X (µm)', 'Y (µm)', self.plot_unit[case])
        self.fig2.tight_layout()

    def line_cut(self, pos=0, axis='X', tilt=0, shift=0, case='height'):

        idx_list = np.where(np.round(self.X, 2) == pos)
        sub_idx = int(len(idx_list[0]) / 2)
        idx = idx_list[0][sub_idx - 1]
        data_2d = getattr(self, case)
        if axis == 'X':
            # line cut along Y, at a constant X value
            data_1d = data_2d[:, idx]
            self.line_cut_axis_label = 'Y (µm)'
        else:
            # line cut along X, at a constant Y value
            data_1d = data_2d[idx, :]
            self.line_cut_axis_label = 'X (µm)'

        pos_txt = axis + '=' +str(pos)
        self.fig3 = plt.figure(figsize=(6, 5))
        self.fig3.text(.2, .85, pos_txt, fontsize=18)
        self.ax3 = self.fig3.add_subplot(111)
        # tilt correction
        y_tilt = tilt * self.X + shift
        self.ax3.plot(self.X, data_1d + y_tilt)
        prettify_plot(self.ax3, self.line_cut_axis_label, self.plot_unit[case])
        self.fig3.tight_layout()