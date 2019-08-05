import numpy as np
import pandas as pd
import sdtfile as sdt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scipy.special

import matplotlib.pyplot as plt
import pathlib
import glob
import os
from matplotlib import cm
import inspect
#from PIL import Image  # pillow
import random as rn
from numpy.lib.stride_tricks import as_strided
from matplotlib.colors import LogNorm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def getModelVars(function):
    params = inspect.signature(function).parameters
    vd= {}
    vd['parameters'] = []
    vd['name'] = function.__name__
    for i,pps in enumerate(params.items()):
        if i ==0:
            continue
        else:
            pname, pr = pps
        if pr.default == inspect._empty:
            vd['parameters'].append(pname)
    return vd


def printModel(model, pfit, pcov, chi2, oneliner=True):
    vd = getModelVars(model)
    line = '{} (chi2 = {:.3f}): '.format(vd["name"], chi2)
    if not oneliner:
        line += '\n'
    for i in range(len(pfit)):
        if oneliner:
            line += '{} = {:.3f} \u00B1 {:.3f}, '.format(vd['parameters'][i], pfit[i], np.sqrt(np.diag(pcov))[i])
        else:
            line += '{: <5} = {:.3f} \u00B1 {:.3f}\n'.format(vd['parameters'][i], pfit[i], np.sqrt(np.diag(pcov))[i])
    return line[:line.rfind(',')]
        

def meanDecay(FCube):
    means = []
    for i in range(FCube.timesteps):
        im = np.ma.masked_array(FCube.data[:, :, i], mask=FCube.mask)
        means.append(np.mean(im))
    times = np.arange(FCube.timesteps) * FCube.tresolution
    return times, means


def fitPixel(x, y, initial_p=None, norm=False, threshold=None, model=None, bounds=(0, np.inf)):
    imax = np.argmax(y)
    dx = x[1] - x[0]
    # xf = x[imax:]
    yf = np.array(y[imax:])
    xf = np.arange(len(yf)) * dx
    if norm:
        yf = yf / np.max(yf)
    if threshold is not None:
        wabove = np.where(yf > threshold)[0]
        xf = xf[wabove]
        yf = yf[wabove]
    pfit, pcov = curve_fit(model, xf, yf, initial_p, bounds=bounds)
    chi2, pp = chisquare(yf, model(xf, *pfit), len(pfit))
    return xf, yf, pfit, pcov, chi2


def getKernel(bin=1, kernel="mean", sigma=None):
    N = 2 * bin + 1
    if kernel == "linear":
        prob= list(np.arange(bin+1)+1)
        probs = prob+prob[::-1][1:]
        _kernel = np.outer(probs, probs)
        _kernel = _kernel/np.sum(_kernel)
    if kernel == "mean":
        _kernel = np.ones((N, N)) / (N ** 2)
    if kernel == "airy":
        k = bin
        x = np.linspace(-10,10,1001)
        probs = []
        for z in x:
            if z == 0:
                probs.append(1.0)
            else:
                probs.append(4 * (scipy.special.j1(z) / z)**2)
        if sigma is not None:
            s = sigma
        else:
            s = 3.8317
        xt = x/3.8317*s
        zt = np.arange(-k, k+1)
        probt = np.interp(zt, xt, probs)
        _kernel = np.outer(probt, probt)
        _kernel /= np.sum(_kernel)
    if kernel == "gauss":
        if sigma is None:
            s = (bin+1)/3.
        else:
            s = sigma
        k = bin
        probs = [
            np.exp(-z * z / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)
            for z in range(-k, k + 1)
        ]
        _kernel = np.outer(probs, probs)
        _kernel /= np.sum(_kernel)
    return _kernel

def binCube(FCube, channel=None, bin=1, kernel="mean", sigma=None):
    if channel is None:
        idx = np.arange(FCube.data.shape[2])
        outarray = np.empty((FCube.data.shape))
    else:
        idx = [channel]
        outarray = np.empty((FCube.xpix, FCube.ypix, 1))
    _kernel = getKernel(bin, kernel, sigma)
    for j in range(len(idx)):
        outarray[:, :, j] = signal.convolve2d(
            FCube.data[:, :, idx[j]], _kernel, mode="same"
        )
    header = FCube.header
    header["flimview"]["binned"] = {}
    header["flimview"]["binned"]["bin"] = bin
    header["flimview"]["binned"]["kernel"] = kernel
    header["flimview"]["binned"]["sigma"] = sigma
    return FlimCube(outarray, header, binned=True)





class FlimCube(object):
    def __init__(self, data, header, binned=False, masked=False):
        self.xpix = header["flimview"]["xpix"]
        self.ypix = header["flimview"]["ypix"]
        self.timesteps = header["flimview"]["tpix"]
        self.tresolution = header["flimview"]["tresolution"]
        self.data = data
        self.header = header
        self.binned = binned
        self.masked = masked
        self.mask = None
        self.intensity = np.sum(self.data, axis=2)
        self.peak = np.max(self.data, axis=2)

    def show_header(self):
        """Display information about the file"""
        for k, v in self.header.items():
            if k == "flimview":
                print("---------------------")
                for k, v in self.header["flimview"].items():
                    if k == "binned":
                        continue
                    print("{}: {}".format(k, v))
            else:
                print("{}: {}".format(k, v))
        print("---------------------------")
        print(f"Data Shape: {np.shape(self.data)}")
        print(f"Time resolution [ps]: {self.tresolution}")
        print("---------------------------")
        if self.binned:
            print("Binned Information")
            print("------------------")
            for k, v in self.header["flimview"]["binned"].items():
                print("{}: {}".format(k, v))

    def mask_intensity(self, minval, mask=None):
        self.unmask()
        if mask is None:
            self.intensity = np.ma.masked_where(self.intensity < minval, self.intensity)
        else:
            self.intensity = np.ma.masked_array(self.intensity, mask=mask)
        self.masked = True
        self.mask = self.intensity.mask
        self.peak = np.ma.masked_array(self.peak, mask=self.mask)

        
    def mask_peak(self, minval, mask=None):
        self.unmask()
        if mask is None:
            self.peak = np.ma.masked_where(self.peak < minval, self.peak)
        else:
            self.peak = np.ma.masked_array(self.peak, mask=mask)
        self.masked = True
        self.mask = self.peak.mask
        self.intensity = np.ma.masked_array(self.intensity, mask=self.mask)

    def unmask(self):
        self.intensity = np.sum(self.data, axis=2)
        self.peak = np.max(self.data, axis=2)
        self.masked = False
        self.mask = None


class FLIM1(object):
    def __init__(self, tiffile, sdtfile):
        self.tiffile = tiffile
        self.sdtfile = sdtfile
        self.tif = Image.open(self.tiffile)
        self.sdt = sdt.SdtFile(self.sdtfile)
        self.timesteps = 256
        self.xpix = 256
        self.ypix = 256
        if np.shape(self.sdt.data)[0] == 0:
            print(f"There is an error with this file: {sdtfile}")
        else:
            self.sdt_meta = pd.DataFrame.from_records(self.sdt.measure_info[0])
            self.sdt_meta = self.sdt_meta.append(
                pd.DataFrame.from_records(self.sdt.measure_info[1]), ignore_index=True
            )
            self.sdt_meta = self.sdt_meta.append(
                pd.DataFrame.from_records(self.sdt.measure_info[2]), ignore_index=True
            )
            self.sdt_meta = self.sdt_meta.append(
                pd.DataFrame.from_records(self.sdt.measure_info[3]), ignore_index=True
            )

    @classmethod
    def from_type(cls, diagnose, layer):
        tiff = "Psoriasis/test_dataset/c1_{0}/60_{0}_760nm_t{1:03}-c1.tif".format(
            diagnose, layer
        )
        sdt = "Psoriasis/test_dataset/FLIM_{0}/60_{0}_760nm_c{1:02}.sdt".format(
            diagnose, layer
        )
        return cls(tiff, sdt)

    def show_info(self):
        """Display information about the file"""
        print(self.sdt.info)
        print(f"Data Shape: {np.shape(self.sdt.data)}")
        print(f"Time resolution [ps]: {self.sdt.times[0][1]/1e-12}")

    def show_tif(self):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ims = ax.imshow(self.tif, cmap=cm.inferno)
        ax.set_title(os.path.basename(self.tiffile))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(ims, cax=cax)
        plt.show()

    def extract2D(self, channel=0, t=0, summed=False):
        """Extract a 2D snapshot for a given channel for a given timestep, it can also returned a integrated
        across channels

        Parameters
        ----------
        channel: int (default is 0)
            Channel used during sampling, only 0 is used
        t: int (default is 0)
            Time step (out of 256)
        summed: bool (default is False)
            Whether return or not integrated count across all channels

        Returns
        -------
        ndarray
            2D array containing data with `int` type with photon count for a
            single timestamp

        """
        if summed:
            vector = np.sum(self.sdt.data, axis=0)[:, t]
        else:
            vector = self.sdt.data[channel][:, t]
        return np.reshape(vector, (self.xpix, self.ypix))

    def extractI(self, channel=0, summed=False):
        """Returns Intensity (integrated across time) of the photon count for a given channel

        Parameters
        ----------
        channel: int (default is 0)
            Channel used during sampling, only 0 is used
        summed: bool (default is False)
            Whether return or not integrated count across all channels

        Returns
        -------
        ndarray
            2D array containing data with `int` type with photon count integrated across
            timesteps (and channels if enabled)

        """
        self.I = np.zeros((self.xpix, self.ypix))
        for i in range(self.timesteps):
            self.I += self.extract2D(channel=channel, t=i, summed=summed)

    def show_I(self, savefig=None):
        """Display Intensity image"""
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        if not hasattr(self, "I"):
            self.extractI()
        ims = ax.imshow(self.I, cmap=cm.inferno)
        ax.set_title(self.sdt.name)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(ims, cax=cax)
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()

    def extractBin(self, channel=0, t=0, bin=1, kernel="mean", sigma=1.0, summed=False):
        """Extracts a `binned` image for a given channel for a given timestep with a given kernel

        Parameters
        ----------
        channel: int (default is 0)
            Channel
        t: int
            Timestep (default is  0 out of 256)
        bin: int (default is 1)
            size of array to use centered on pixel, 0 = do nothing, 1 = use a 3x3 array,
            2 = use a 5x5 array, i.e, use a 2 * bin + 1 squared array
        kernel: str (default is mean)
             What kernel used to do the convolution,
             `mean` : simple mean across neighbor pixels
             `gauss`: gaussian kernel
        sigma: float (default is 1.0)
            Sigma used for the gaussian kernel
        summed: bool (default is False)
            Results integrated across channels

        Returns
        -------
        ndarray
            2D array (256 x 256) containing data with `float` type with photon count convoled with a kernel

        """
        imgarray = self.extract2D(channel, t, summed=summed)
        N = 2 * bin + 1
        if kernel == "mean":
            _kernel = np.ones((N, N)) / (N ** 2)
        if kernel == "gauss":
            s = sigma
            k = bin
            probs = [
                np.exp(-z * z / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)
                for z in range(-k, k + 1)
            ]
            _kernel = np.outer(probs, probs)
            _kernel /= np.sum(_kernel)
        return signal.convolve2d(imgarray, _kernel, mode="same")

    def extractAllBin(self, channel=0, bin=1, kernel="mean", sigma=1.0, summed=False):
        """Extracts binned images for all timesteps"""
        self.bin = bin
        self.binned_data = np.zeros((self.xpix, self.ypix, self.timesteps))
        for i in range(self.timesteps):
            self.binned_data[:, :, i] = self.extractBin(
                channel=channel, t=i, bin=bin, kernel=kernel, sigma=sigma, summed=summed
            )

    def extractMeanDecay(self, channel=0):
        means = []
        for i in range(self.timesteps):
            imgarray = self.extract2D(channel, i, summed=False)
            means.append(np.mean(imgarray))
        self.mean_decay = np.array(means)
        return self.mean_decay

    def plot_mean_decay(self, channel=0):
        if not hasattr(self, "mean_decay"):
            _ = self.extractMeanDecay(channel)
        plt.figure(figsize=(6, 6))
        plt.plot(self.sdt.times[channel] / 1e-9, self.mean_decay)
        plt.xlabel("time [ns]", fontsize=15)
        plt.ylabel("Mean photon count", fontsize=15)
        plt.show()

    @staticmethod
    def fit_data(x, y, initial_p=[1.0, 1.0, 1.0, 1.0], norm=False, threshold=None):
        imax = np.argmax(y)
        dt = x[1] - x[0]
        # xf = x[imax:]
        yf = y[imax:]
        xf = np.arange(len(yf)) * dt
        if threshold is not None:
            wabove = np.where(yf > threshold)[0]
            xf = xf[wabove]
            yf = yf[wabove]
        if norm:
            yf = yf / np.max(yf)
        pfit, pcov = curve_fit(model, xf, yf, initial_p)
        return xf, yf, pfit, pcov
