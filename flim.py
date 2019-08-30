import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scipy.special
from tqdm import tqdm
import inspect


def getModelVars(function):
    """Get the variables for a given model/function. Returns a list of ordered parameters and the
    function's name.

    Parameters
    ----------
    function : function
        Model defined functions

    Returns
    -------
    vd : dict
        Dictionary containing the name and the input variables for the model.

    """
    params = inspect.signature(function).parameters
    vd = {}
    vd["parameters"] = []
    vd["name"] = function.__name__
    for i, pps in enumerate(params.items()):
        if i == 0:
            continue
        else:
            pname, pr = pps
        if pr.default == inspect._empty:
            vd["parameters"].append(pname)
    return vd


def printModel(model, pfit, pcov, chi2, oneliner=True):
    """Prints string of resulting models.

    Parameters
    ----------
    model : Object
        Model defined functions
    pfit :
        Resulting array with parameters
    pcov :
        Resulting covariance matrix from the fit
    chi2 : float
        Resulting chi2
    oneliner : bool, optional
        Print everyting in one line (default is True)

    Returns
    -------
    line : str
        String with the resulting models
    """

    vd = getModelVars(model)
    line = "{} (chi2 = {:.3f}): ".format(vd["name"], chi2)
    if not oneliner:
        line += "\n"
    for i in range(len(pfit)):
        if oneliner:
            line += "{} = {:.3f} \u00B1 {:.3f}, ".format(
                vd["parameters"][i], pfit[i], np.sqrt(np.diag(pcov))[i]
            )
        else:
            line += "{: <5} = {:.3f} \u00B1 {:.3f}\n".format(
                vd["parameters"][i], pfit[i], np.sqrt(np.diag(pcov))[i]
            )
    return line[: line.rfind(",")]


def meanDecay(FCube):
    """Computes the mean decay across all of the pixels. Used to get initial estimates
    of parameters

    Parameters
    ----------
    FCube : flim.flimCube
        Instance of a flimCube object

    Returns
    -------
    times: float
        Array with timesteps
    means : float
        Array with mean values across all of the pixel for a given timestep
    """
    means = []
    for i in range(FCube.timesteps):
        im = np.ma.masked_array(FCube.data[:, :, i], mask=FCube.mask)
        means.append(np.mean(im))
    times = FCube.t
    return times, means


def cleanCurve(x, y, norm=False, threshold=0.01, xshift=None):
    """Cleans the decay dta before fitting.

    Parameters
    ----------
    x : float
        Array with time values
    y : float
        Array with decay values
    norm : bool
        If True, it normalizes the data to its maximum (default is True)
    threshold : float
        If True, only data above threshold is considered for the fitting (default is 0.01)
    xshift : float
        Shifts the time axis by `xshift` to fixed the initial decay timestep (default is None)
        If None, data is shifted to start at maximum point at t=0

    Returns
    -------
    x1 : float
        Cleaned x
    y1 : float
        Cleaned y
    ymax : float
        Maximum value for y in case data is normalized
    xshift : float
        Time shift applied to the data
    """
    x = np.array(x)
    y = np.array(y)
    ymax = np.max(y)
    if norm:
        y = y / np.max(y)
    if threshold is not None:
        wabove = np.where(y > threshold)[0]
        x = x[wabove]
        y = y[wabove]
    imax = np.argmax(y)
    y1 = y[imax:]
    x1 = x[imax:]
    if xshift is None:
        xshift = np.min(x1)
    x1 -= xshift
    return x1, y1, ymax, xshift


def fitPixel(x, y, model, initial_p=None, norm=False, threshold=0.01, xshift=None, bounds=None):
    """Fit a given pixel from the fitCube by using a custom model

    Parameters
    ----------
    x : float
        Array with time values
    y : float
        Array with decay values
    model : callable
        function to be used defined in flim.models
    initial_p : array_like
        Initial guess for the fitting function (default is None, used ones)
    norm : bool
        If True, it normalizes the data to its maximum (default is True)
    threshold : float
        If True, only data above threshold is considered for the fitting (default is 0.01)
    xshift : float
        Shifts the time axis by `xshift` to fixed the initial decay timestep (default is None)
        If None, data is shifted to start at maximum point at t=0
    bounds : 2-tuple of array_like
        Lower and upper bounds on parameters. Defaults to no bounds.

    Returns
    -------
    xf : ndarray
        Time array used in the fitting (after cleaning). Usually smaller than x
    yf : float
        Cleaned y-axis data used in the fitting. Different from input y
    pfit : 1d ndarray
        Fitted model parameters
    pcov : 2d ndarray
        Covariance matrix from the fitted Parameters.
    chi2 : float
        Chi square result from the fitting
    """
    if bounds is None:
        bounds = (-np.inf, np.inf)
    xf, yf, yf_max, xf_shift = cleanCurve(
        x, y, norm=norm, threshold=threshold, xshift=xshift
    )
    pfit, pcov = curve_fit(model, xf, yf, initial_p, bounds=bounds)
    chi2, pp = chisquare(yf, model(xf, *pfit), len(pfit))
    return xf, yf, pfit, pcov, chi2


def fitCube(FCube, model, guessp=None, bounds=None, norm=False, threshold=None, xshift=None):
    """Apply fitPixel to every pixel in a flimCube

    Parameters
    ----------
    FCube : flim.flimCube object
        Instance of a flimCube
    model : callable
        function to be used defined in flim.models
    guessp : array_like
        Initial guess for the fitting function (default is None, used ones)
    norm : bool
        If True, it normalizes the data to its maximum (default is True)
    threshold : float
        If True, only data above threshold is considered for the fitting (default is 0.01)
    xshift : float
        Shifts the time axis by `xshift` to fixed the initial decay timestep (default is None)
        If None, data is shifted to start at maximum point at t=0
    bounds : 2-tuple of array_like
        Lower and upper bounds on parameters. Defaults to no bounds.

    Returns
    -------
    FFit : flim.fitFit Object
        A fitFit object with all of the parameters results for each pixel
    """
    PP = np.zeros((FCube.xpix, FCube.ypix, 2, len(guessp) + 2)) - 1
    failed = np.zeros((FCube.xpix, FCube.ypix))
    for i in tqdm(range(FCube.xpix)):
        for j in range(FCube.ypix):
            if FCube.mask[i, j]:
                continue
            try:
                y = FCube.data[i, j]
                x = FCube.t
                xf, yf, pfit_i, pcov_i, chi2_i = fitPixel(
                    x,
                    y,
                    model,
                    initial_p=guessp,
                    bounds=bounds,
                    norm=norm,
                    threshold=threshold,
                    xshift=xshift,
                )
                for k in range(len(guessp)):
                    PP[i, j, 0, k] = pfit_i[k]
                    PP[i, j, 1, k] = np.sqrt(np.diag(pcov_i)[k])
                PP[i, j, 0, -1] = chi2_i
                PP[i, j, 0, -2] = np.mean(yf - model(xf, *pfit_i))
            except:
                failed[i, j] = 1
    FFit = FlimFit(FCube, model)
    FFit.load_results(PP)
    return FFit


def getKernel(bin=4, kernel="flat", sigma=None):
    """Get a kernel used to bin the data

    Parameters
    ----------
    bin : int
        Size of the binning window around a given pixel, square with 2 * bin + 1 side
        (default 4)
    kernel : str
        Kernel to be used among `flat`, `linear`, `gauss` and `airy`. (default to `flat`)
    sigma : float
        Sigma for the gaussian kernel or for the Airy disk. (default is None, (bin+1)/3)

    Returns
    -------
    _kernel : 2D ndarray
        Kernel to be used for binning and filtering
    """
    N = 2 * bin + 1
    if kernel == "linear":
        prob = list(np.arange(bin + 1) + 1)
        probs = prob + prob[::-1][1:]
        _kernel = np.outer(probs, probs)
        _kernel = _kernel / np.sum(_kernel)
    if kernel == "flat":
        _kernel = np.ones((N, N)) / (N ** 2)
    if kernel == "airy":
        k = bin
        x = np.linspace(-10, 10, 1001)
        probs = []
        for z in x:
            if z == 0:
                probs.append(1.0)
            else:
                probs.append(4 * (scipy.special.j1(z) / z) ** 2)
        if sigma is not None:
            s = sigma
        else:
            s = 3.8317
        xt = x / 3.8317 * s
        zt = np.arange(-k, k + 1)
        probt = np.interp(zt, xt, probs)
        _kernel = np.outer(probt, probt)
        _kernel /= np.sum(_kernel)
    if kernel == "gauss":
        if sigma is None:
            s = (bin + 1) / 3.0
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


def binCube(FCube, channel=None, bin=1, kernel="flat", sigma=None):
    """Apply a binning kernel to a flimCube instance and returns another flimCube instance with
    with the binned data

    Parameters
    ----------
    FCube: flim.flimCube
        An instance of a flimCube
    channel: int
        Apply binning to just one channel instead of every timestep.
    bin : int
        Size of the binning window around a given pixel, square with 2 * bin + 1 side
        (default 4)
    kernel : str
        Kernel to be used among `flat`, `linear`, `gauss` and `airy`. (default to `flat`)
    sigma : float
        Sigma for the gaussian kernel or for the Airy disk. (default is None, (bin+1)/3)

    Returns
    -------
    FlimCube : flim.flimCube
        Retuns an instance of a flimCube with the binned data.
    """
    if channel is None:
        idx = np.arange(FCube.data.shape[2])
        outarray = np.empty((FCube.data.shape))
    else:
        idx = [channel]
        outarray = np.empty((FCube.xpix, FCube.ypix, 1))
    _kernel = getKernel(bin, kernel, sigma)
    for j in range(len(idx)):
        temp = signal.convolve2d(FCube.data[:, :, idx[j]], _kernel, mode="same")
        if np.sum(FCube.data[:, :, idx[j]]) > 0:
            temp = temp / np.sum(temp) * np.sum(FCube.data[:, :, idx[j]])
        outarray[:, :, j] = temp
    header = FCube.header
    header["flimview"]["binned"] = {}
    header["flimview"]["binned"]["bin"] = bin
    header["flimview"]["binned"]["kernel"] = kernel
    header["flimview"]["binned"]["sigma"] = sigma
    return FlimCube(outarray, header, binned=True)


class FlimFit(object):
    def __init__(self, Fcube, model):
        self.Fcube = Fcube
        self.model = model
        self.mask = Fcube.mask
        self.masked = Fcube.masked
        vd = getModelVars(self.model)
        self.parameters = vd["parameters"]
        self.model_name = vd["name"]

    def load_results(self, results):
        for i, name in enumerate(self.parameters):
            setattr(self, name, np.ma.masked_array(results[:, :, 0, i], mask=self.mask))
            setattr(
                self,
                name + "_err",
                np.ma.masked_array(results[:, :, 1, i], mask=self.mask),
            )
        self.chi2 = np.ma.masked_array(results[:, :, 0, -1], mask=self.mask)
        self.residuals = np.ma.masked_array(results[:, :, 0, -2], mask=self.mask)
        del results

    def load_single(self, name, data):
        setattr(self, name, np.ma.masked_array(data, mask=self.mask))
        del data


class FlimCube(object):
    """
    A class to represent 3d data, masking and binned data

    Attributes
    ----------
    xpix : int
        Xsize
    ypix : int
        Ysize
    timestep : int
        Size of the time dimension
    tresolution : float
        Resolution of the time axis in nano seconds
    data : 3d ndarray
        The actual data cube
    header : dict
        Header with information from reading the file and metadata
    binned : bool
        Whether this cube is binned or not
    masked : bool
        Whether the data is masked or not
    mask : array_like
        The 2d masking array
    intensity : 2d ndarray
        Integrated counts of the cube summed along the time axis
    peak : 2d ndarray
        Map of the peak for each pixel
    t : 1d ndarray
        Time steps

    Methods
    -------
    show_header()
        Display information about the file
    mask_intensity(minval, mask=None)
        Mask the data by intensity values or a passed mask
    mask_peak(minval, mask=None)
        Mask the data by the peak values or a passed mask
    unmask()
        Remove any applied mask
    """
    def __init__(self, data, header, binned=False, masked=False):
        """
        Creates a flimCube instance.

        Parameters
        ----------
        data : 3d ndarray
            Data cube comming from io methods
        header : dict
            Header with information about the file and metadata
        binned : bool
            Whether the input data is binnned
        masked : bool
            Whether the input data is masked
        """
        self.xpix = int(header["flimview"]["xpix"])
        self.ypix = int(header["flimview"]["ypix"])
        self.timesteps = int(header["flimview"]["tpix"])
        self.tresolution = float(header["flimview"]["tresolution"])
        self.data = data
        self.header = header
        self.binned = binned
        self.masked = masked
        self.mask = None
        self.intensity = np.sum(self.data, axis=2) * 1.0
        self.peak = np.max(self.data, axis=2) * 1.0
        self.t = np.arange(self.timesteps) * self.tresolution / 1000.0

    def show_header(self):
        """
        Display information about the file
        """
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
        print("Data Shape: {}".format(np.shape(self.data)))
        print("Time resolution [ps]: {}".format(self.tresolution))
        print("---------------------------")
        if self.binned:
            print("Binned Information")
            print("------------------")
            for k, v in self.header["flimview"]["binned"].items():
                print("{}: {}".format(k, v))

    def mask_intensity(self, minval, mask=None):
        """
        Mask the data by intensity values or a passed mask, it recomputes the intensity, the peak

        Parameters
        ----------
        minval : float
            Min value for the intensity to be masked
        mask : 2d bool ndarray
            If this is passed it overrides the minval (default is None)
        """
        self.unmask()
        if mask is None:
            self.intensity = np.ma.masked_where(self.intensity < minval, self.intensity)
        else:
            self.intensity = np.ma.masked_array(self.intensity, mask=mask)
        self.masked = True
        self.mask = self.intensity.mask
        self.peak = np.ma.masked_array(self.peak, mask=self.mask)

    def mask_peak(self, minval, mask=None):
        """
        Mask the data by the values or a passed mask, it recomputes the intensity, the peak

        Parameters
        ----------
        minval : float
            Min value for the peak to be masked
        mask : 2d bool ndarray
            If this is passed it overrides the minval (default is None)
        """
        self.unmask()
        if mask is None:
            self.peak = np.ma.masked_where(self.peak < minval, self.peak)
        else:
            self.peak = np.ma.masked_array(self.peak, mask=mask)
        self.masked = True
        self.mask = self.peak.mask
        self.intensity = np.ma.masked_array(self.intensity, mask=self.mask)

    def unmask(self):
        """
        Remove any applied mask and recomputes the peak and the intensity
        """
        self.intensity = np.sum(self.data, axis=2) * 1.0
        self.peak = np.max(self.data, axis=2) * 1.0
        self.masked = False
        self.mask = None
