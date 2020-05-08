import struct
import time
import numpy as np
from sdtfile import SdtFile
import pandas as pd
import copy
import os
import h5py
import json
from .flim import FlimCube, FlimFit
from . import models


def read_sdt_file(sdtfile, channel=0, xpix=256, ypix=256, tpix=256):
    """
    Reads a sdtfile and returns the header and a data cube.

    Parameters
    ----------
    sdtfile : str
        Path to SdtFile
    channel : int
    xpix : int
    ypix : int
    tpix : int

    Returns
    -------
    3d ndarray
        Read dataset with shape (xpix, ypix, tpix)
    dict
        Header information
    """
    sdt = SdtFile(sdtfile)
    if np.shape(sdt.data)[0] == 0:
        print("There is an error with this file: {}".format(sdtfile))
    sdt_meta = pd.DataFrame.from_records(sdt.measure_info[0])
    sdt_meta = sdt_meta.append(
        pd.DataFrame.from_records(sdt.measure_info[1]), ignore_index=True
    )
    sdt_meta.append(pd.DataFrame.from_records(sdt.measure_info[2]), ignore_index=True)
    sdt_meta = sdt_meta.append(
        pd.DataFrame.from_records(sdt.measure_info[3]), ignore_index=True
    )
    header = {}
    header["flimview"] = {}
    header["flimview"]["sdt_info"] = sdt.info
    header["flimview"]["filename"] = os.path.basename(sdtfile)
    header["flimview"]["pathname"] = os.path.dirname(sdtfile)
    header["flimview"]["xpix"] = xpix
    header["flimview"]["ypix"] = ypix
    header["flimview"]["tpix"] = tpix
    header["flimview"]["tresolution"] = sdt.times[0][1] / 1e-12
    return np.reshape(sdt.data[channel], (xpix, ypix, tpix)), header


def read_ptu_header(infile):
    """ Read header from a ptu file and returns a dictionary with all
    the information
    Largely inspired from PicoQuant examples:
    https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos
    https://gist.github.com/tritemio/734347586bc999f39f9ffe0ac5ba0e66


    Parameters
    ----------
    infile : str
        Filepath to ptu file

    Returns
    -------
    dict
        Dictionary
    """

    # struct types fron the header, taken/inspired from :
    tyEmpty8 = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
    tyBool8 = struct.unpack(">i", bytes.fromhex("00000008"))[0]
    tyInt8 = struct.unpack(">i", bytes.fromhex("10000008"))[0]
    tyBitSet64 = struct.unpack(">i", bytes.fromhex("11000008"))[0]
    tyColor8 = struct.unpack(">i", bytes.fromhex("12000008"))[0]
    tyFloat8 = struct.unpack(">i", bytes.fromhex("20000008"))[0]
    tyTDateTime = struct.unpack(">i", bytes.fromhex("21000008"))[0]
    tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
    tyAnsiString = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
    tyWideString = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
    tyBinaryBlob = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

    # Record types,specific for HydarHrt T3 Version 2
    rtHydraHarp2T3 = struct.unpack(">i", bytes.fromhex("01010304"))[0]
    inputfile = open(infile, "rb")
    magic = inputfile.read(8).decode("utf-8").strip("\0")
    version = inputfile.read(8).decode("utf-8").strip("\0")
    tagDataList = []  # Contains tuples of (tagName, tagValue)
    header = {}
    while True:
        tagIdent = inputfile.read(32).decode("utf-8").strip("\0")
        tagIdx = struct.unpack("<i", inputfile.read(4))[0]
        tagTyp = struct.unpack("<i", inputfile.read(4))[0]
        if tagIdx > -1:
            evalName = tagIdent + "(" + str(tagIdx) + ")"
        else:
            evalName = tagIdent
        if tagTyp == tyEmpty8:
            inputfile.read(8)
            tagDataList.append((evalName, "<empty Tag>"))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if tagInt == 0:
                tagDataList.append((evalName, "False"))
            else:
                tagDataList.append((evalName, "True"))
        elif tagTyp == tyInt8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyBitSet64:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyColor8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyFloat8Array:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyTDateTime:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagTime = int((tagFloat - 25569) * 86400)
            tagTime = time.gmtime(tagTime)
            tagDataList.append((evalName, tagTime))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = (
                inputfile.read(tagInt).decode("utf-16le", errors="ignore").strip("\0")
            )
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        else:
            print("ERROR: Unknown tag type")
            exit(0)
        if tagIdent == "Header_End":
            break
    tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
    tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]
    for k, v in zip(tagNames, tagValues):
        header[k] = v
    header["Header_End_Bytes"] = inputfile.tell()
    header["Ffilename"] = os.path.basename(infile)
    header["Fpathname"] = os.path.dirname(infile)
    inputfile.close()
    return header


def read_ptu_records(infile, header, skip_records=0, read_records=None):
    """
    Reads the records from a binary blob and returns structured data

    Parameters
    ----------
    infile : str
        Filepath to ptu file
    header : dict
        Header of the ptu file
    skip_records : int
        Number fo records to skip (default is 0)
    read_records : int
        Number of records to read (default is None which reads total number from header)

    Returns
    -------
    sync : 1d ndarray uint64
        Sync values
    tcspc : 1d ndarray uint16
        Time values
    channel : 1d ndarray uint8
        Channel values
    special : 1d ndarray uint8
        Marker values
    index : 1d ndarray uint8
        Overflow index values
    """
    if read_records is None:
        read_records = header["TTResult_NumberOfRecords"]
    inputfile = open(infile, "rb")
    shift = skip_records * 4
    inputfile.seek(header["Header_End_Bytes"] + shift, 0)
    buffer = inputfile.read(4 * read_records)
    records = np.frombuffer(buffer, dtype="uint32", count=read_records)
    # Lowest 10 bits
    sync = np.bitwise_and(records, 2 ** 10 - 1)
    # Next 15 bits, dtime can be obtained from header
    tcspc = np.bitwise_and(np.right_shift(records, 10), 2 ** 15 - 1)
    # Next 6 bits
    chan = np.bitwise_and(np.right_shift(records, 25), 2 ** 6 - 1)
    # Last bit for special markers
    special = np.bitwise_and(records, 2 ** 31) > 0
    # Find overflow locations
    index = (special * 1) * ((chan == 63) * 1)
    special = (special * 1) * chan
    sync = sync.astype(np.uint64)
    tcspc = tcspc.astype(np.uint16)
    channel = chan.astype(np.uint8)
    special = special.astype(np.uint8)
    index = index.astype(np.uint8)
    # del index
    del records
    inputfile.close()
    return sync, tcspc, channel, special, index


def clean_ptu_oflow(header, sync, tcspc, channel, special, index, wrap=1024):
    """
    Cleans and reformat ptu records by Overflow

    Parameters
    ----------
    header : dict
        Header of the ptu inputfile
    sync : 1d ndarray uint64
        Sync values
    tcspc : 1d ndarray uint16
        Time values
    channel : 1d ndarray uint8
        Channel values
    special : 1d ndarray uint8
        Marker values
    index : 1d ndarray uint8
        Overflow index values
    wrap : int
    Overflow wrapping bytes (default 1024)

    Returns
    -------
    header : dict
        Updated header of the ptu inputfile
    sync : 1d ndarray uint64
        Cleaned sync values
    tcspc : 1d ndarray uint16
        Cleaned time values
    channel : 1d ndarray uint8
        Cleaned channel values
    special : 1d ndarray uint8
        Clean marker values
    line_start : 1d ndarray
        Array with index for the starting of the new line for the image
    line_stop : 1d ndarray
        Array with index for the starting of the end line for the image
    start_frames : 1d ndarray
        Array with the index of the starting point for a new frame
    """
    oflow = np.where(index == 1)
    sync = sync + (wrap * np.cumsum(index * sync))
    sync = np.delete(sync, oflow, axis=0)
    tcspc = np.delete(tcspc, oflow, axis=0)
    channel = np.delete(channel, oflow, axis=0)
    special = np.delete(special, oflow, axis=0)
    del index
    LineStartMarker = 2 ** (header["ImgHdr_LineStart"] - 1)
    LineStopMarker = 2 ** (header["ImgHdr_LineStop"] - 1)
    FrameMarker = 2 ** (header["ImgHdr_Frame"] - 1)
    header["flimview"] = {}
    header["flimview"]["filename"] = header["Ffilename"]
    header["flimview"]["pathname"] = header["Fpathname"]
    header["flimview"]["xpix"] = header["ImgHdr_PixX"]
    header["flimview"]["ypix"] = header["ImgHdr_PixY"]
    header["flimview"]["tpix"] = np.max(tcspc) - np.min(tcspc) + 1
    header["flimview"]["tresolution"] = header["MeasDesc_Resolution"] / 1e-12
    header["flimview"]["ptu"] = {}
    line_start = sync[np.where(special == LineStartMarker)]
    line_stop = sync[np.where(special == LineStopMarker)]
    start_frames = np.where(special == FrameMarker)[0]
    header["flimview"]["ptu"]["nframes"] = len(start_frames)
    header["flimview"]["ptu"]["nentries"] = len(sync)
    return header, sync, tcspc, channel, special, line_start, line_stop, start_frames


def read_ptu_frame(
    header,
    sync,
    tcspc,
    channel,
    special,
    start_frames,
    L1,
    L2,
    view=0,
    nframes=1,
    frame_shift=0,
    frames_per_view=20,
    res_factor=1,
):
    """
    Reads one or more frames from the given input and aggregates them to create
    a final 3d cube

    Parameters
    ----------
    header : dict
        Updated header of the ptu inputfile
    sync : 1d ndarray uint64
        Cleaned sync values
    tcspc : 1d ndarray uint16
        Cleaned time values
    channel : 1d ndarray uint8
        Cleaned channel values
    special : 1d ndarray uint8
        Clean marker values
    start_frames : 1d ndarray
        Array with the index of the starting point for a new frame
    line_start : 1d ndarray
        Array with index for the starting of the new line for the image
    line_stop : 1d ndarray
        Array with index for the starting of the end line for the image
    view : int
        Which view to process, usually 4 (default is 0)
    nframes : int
        How many frames to read at the same time, frames read are aggregated (default is 1)
    frame_shift : int
        How many frames to shift in a given view, this is useful when reading in parallel
        (default is 0)
    frames_per_view : int
        How many frames are expected in a view (default is 20)
    res_factor : int
        Resolution factor to apply in the time axis, if 1 no time binning is applied and the
        original time axis is returned. If 2 or more timesteps are divided by such factor.
        (default is 1)

    Returns
    -------
    3d ndarray
        The data cube for the frames read for the given view and time resolution, if many frames
        are read individually, these need to be aggregated
    dict
        Updated header with general information and metadata added
    """
    out = []
    xpix = header["flimview"]["ypix"]
    ypix = header["flimview"]["xpix"]
    tpix = int(np.ceil(header["flimview"]["tpix"] / res_factor))
    tresolution = header["flimview"]["tresolution"] * res_factor
    headerf = copy.deepcopy(header)
    headerf["flimview"]["tpix"] = tpix
    headerf["flimview"]["tresolution"] = tresolution
    im1 = np.zeros((xpix + 1, ypix, tpix))
    try:
        i = start_frames[view * frames_per_view + frame_shift] + 1
    except:
        return im1[1:, ][:][:], headerf  # remove extra column
    line = (view * frames_per_view + frame_shift) * (xpix + 1)
    frame = 0
    currentLine = 0
    inline = True
    syncStart = L1[line]
    syncPulsesPerLine = L2[line] - L1[line]
    while True:
        try:
            sp = special[i]
        except:
            break
        if sp == 1:
            inline = True
            syncStart = L1[line]
            syncPulsesPerLine = L2[line] - L1[line]
        if sp == 2:
            currentLine += 1  # next line
            inline = False
            line += 1
            try:
                syncStart = L1[line]
                syncPulsesPerLine = L2[line] - L1[line]
            except:
                pass
        if sp == 0 and inline:
            currentSync = sync[i]
            currentPixel = int(
                np.floor((((currentSync - syncStart) / syncPulsesPerLine) * xpix))
            )
            # ch = int(np.log2(channel[i]))  # 0,1,2
            tc = tcspc[i] // res_factor
            try:
                im1[currentLine][currentPixel][tc] += 1
            except:
                out.append([currentLine, currentPixel])
        i += 1  # next entry
        if sp == 4:  # next frame
            currentLine = 0
            #print("frame {}, view {} done".format(frame + frame_shift, view))
            frame += 1
        if frame == nframes:
            break
    return im1[1:, ][:][:], headerf  # remove extra column


def saveCube(FCube, filename=None, group=None, subgroup=None):
    """
    Saves the instance of a flimCube into a HDF5 format, it grabs the name from the header and it
    support root group and subgroup

    Parameters
    ----------
    FCube : A instance of a flim.FlimCube
        The flimCube to be stored
    filename : str
        HDF5 filename where the flimCube is saved (default is None and filename is grabed from the
        header)
    group : str
        Root group inside the HDF5 if provided, this is useful to store multiple views in the same
        file (default is None)
    subgroup : str
        Subgroup provided which help to separate multiple flimCube within a group, useful when use
        to save raw and binned data
    """
    file_name, file_extension = os.path.splitext(FCube.header["flimview"]["filename"])
    if filename is None:
        h5file = os.path.join(
            FCube.header["flimview"]["pathname"],
            FCube.header["flimview"]["filename"].replace(file_extension, ".h5"),
        )
    else:
        h5file = filename
    f = h5py.File(h5file, "a")
    if group is None:
        grp0 = file_name
    else:
        grp0 = group
    try:
        grp = f.create_group(grp0)
    except ValueError:
        grp = f[grp0]
    if subgroup is not None:
        try:
            grp = grp.create_group(subgroup)
        except:
            grp = grp[subgroup]
    if not FCube.binned:
        try:
            subg = grp.create_group("raw")
        except:
            subg = grp["raw"]
    else:
        try:
            subg = grp.create_group("binned")
        except:
            subg = grp["binned"]
    subg.attrs["header"] = json.dumps(FCube.header)
    subg.attrs["binned"] = FCube.binned
    subg.attrs["masked"] = FCube.masked
    try:
        dset = subg.create_dataset(
            "data", shape=FCube.data.shape, dtype=FCube.data.dtype, compression="gzip"
        )
    except:
        dset = subg["data"]
    dset[...] = FCube.data
    if FCube.masked:
        try:
            mset = subg.create_dataset(
                "mask",
                shape=FCube.mask.shape,
                dtype=FCube.mask.dtype,
                compression="gzip",
            )
        except:
            mset = subg["mask"]
        mset[...] = FCube.mask
    f.close()


def loadCube(filename, kind, group=None, subgroup=None):
    """
    Loads a flimCube from a HDF5 filename

    Parameters
    ----------
    filename : str
        HDF5 filename where the flimCube is saved
    kind : str
        Kind of data to read, `raw`,  or `binned`
    group : str
        Root group inside the HDF5 if provided, this is useful to store multiple views in the same
        file (default is None)
    subgroup : str
        Subgroup provided which help to separate multiple flimCube within a group, useful when use
        to save raw and binned data

    Returns
    -------
    flimCube
        Returns a instance of a flimCube with the read data
    """
    f = h5py.File(filename, "r")
    file_name, file_extension = os.path.splitext(os.path.basename(filename))
    if group is None:
        grp = file_name
    else:
        grp = group
    g = f[grp]
    if subgroup is not None:
        g = g[subgroup]
    try:
        sg = g[kind]
    except KeyError:
        print("Error: {} does not exists".format(kind))
        return None
    header = json.loads(sg.attrs["header"])
    masked = sg.attrs["masked"]
    C = FlimCube(np.array(sg["data"]), header, sg.attrs["binned"])
    if masked:
        C.mask_peak(0, np.array(sg["mask"]))
    f.close()
    return C


def saveFit(FFit, filename=None, group=None, subgroup=None):
    """
    Saves a flimFit instance inside a HDF5 file, which might contain other datasets

    Parameters
    ----------
    FFit : flimFit instance
        An instance of a flimFit with the fitted results
    filename : str
        HDF5 filename where the flimCube is saved (default is None and filename is grabed from the
        header)
    group : str
        Root group inside the HDF5 if provided, this is useful to store multiple views in the same
        file (default is None)
    subgroup : str
        Subgroup provided which help to separate multiple flimCube within a group, useful when use
        to save raw and binned data
    """
    file_name, file_extension = os.path.splitext(
        FFit.Fcube.header["flimview"]["filename"]
    )
    if filename is None:
        h5file = os.path.join(
            FFit.Fcube.header["flimview"]["pathname"],
            FFit.Fcube.header["flimview"]["filename"].replace(file_extension, ".h5"),
        )
    else:
        h5file = filename
    f = h5py.File(h5file, "a")
    if group is None:
        grp0 = file_name
    else:
        grp0 = group
    try:
        grp = f.create_group(grp0)
    except ValueError:
        grp = f[grp0]
    if subgroup is not None:
        try:
            grp = grp.create_group(subgroup)
        except:
            grp = grp[subgroup]
    try:
        fit = grp.create_group("fit")
    except:
        fit = grp["fit"]
    fit.attrs["model"] = FFit.model.__name__
    fit.attrs["parameters"] = json.dumps(FFit.parameters)
    masked = FFit.Fcube.masked
    fit.attrs["masked"] = masked
    if masked:
        try:
            mset = fit.create_dataset(
                "mask", shape=FFit.mask.shape, dtype=FFit.mask.dtype, compression="gzip"
            )
        except:
            mset = fit["mask"]
        mset[...] = FFit.mask
    for k in FFit.parameters:
        kerr = "{}_err".format(k)
        try:
            data = getattr(FFit, k)
            kset = fit.create_dataset(
                k, shape=data.shape, dtype=data.dtype, compression="gzip"
            )
        except:
            kset = fit[k]
        kset[...] = data
        try:
            data = getattr(FFit, kerr)
            kset = fit.create_dataset(
                kerr, shape=data.shape, dtype=data.dtype, compression="gzip"
            )
        except:
            kset = fit[kerr]
        kset[...] = data
    try:
        mset = fit.create_dataset(
            "chi2", shape=FFit.chi2.shape, dtype=FFit.chi2.dtype, compression="gzip"
        )
    except:
        mset = fit["chi2"]
    mset[...] = FFit.chi2
    try:
        mset = fit.create_dataset(
            "residuals",
            shape=FFit.residuals.shape,
            dtype=FFit.residuals.dtype,
            compression="gzip",
        )
    except:
        mset = fit["residuals"]
    mset[...] = FFit.residuals
    f.close()


def loadFit(filename, group=None, subgroup=None):
    """
    Loads a flimFit from a HDF5 filename

    Parameters
    ----------
    filename : str
        HDF5 filename where the flimCube is saved
    group : str
        Root group inside the HDF5 if provided, this is useful to store multiple views in the same
        file (default is None)
    subgroup : str
        Subgroup provided which help to separate multiple flimCube within a group, useful when use
        to save raw and binned data

    Returns
    -------
    flimFit
        Returns a instance of a flimFit with the read data
    """
    file_name, file_extension = os.path.splitext(os.path.basename(filename))
    if group is None:
        grp = file_name
    else:
        grp = group
    C = loadCube(filename, "binned", group=grp, subgroup=subgroup)
    f = h5py.File(filename, "r")
    g = f[grp]
    if subgroup is not None:
        g = g[subgroup]
    try:
        sg = g["fit"]
    except KeyError:
        print("Error: {} does not exists".format("fit"))
        return None
    model = getattr(models, sg.attrs["model"])
    F = FlimFit(C, model)
    F.load_single("mask", sg["mask"])
    F.masked = True
    for k in sg.keys():
        if k == "mask":
            continue
        else:
            F.load_single(k, sg[k])
    f.close()
    return F


def descend_obj(obj, sep='----'):
    """
    Iterate through groups in a HDF5 file and prints the groups and
    datasets names and datasets attributes

    Parameters
    ----------
    obj :
        Current group or dataset
    sep : str
        Tab-like string to separate entries
    """
    if type(obj) in [h5py.Group,h5py.File]:
        for key in obj.keys():
            if isinstance(obj[key], h5py.Dataset):
                print('{}> {}: {}'.format(sep, key, obj[key].shape))
            else:
                print('{}> {}: {}'.format(sep, key, obj[key].name))
            descend_obj(obj[key],sep=sep+'----')
    elif type(obj) == h5py.Dataset:
        for key in obj.attrs.keys():
            print('{}----+{}: {}'.format(sep, key, obj.attrs[key]))


def viewH5(path, group='/'):
    """
    print HDF5 file metadata structure

    Parameters
    ----------
    group : str
        Specific group, defaults to the root group
    """
    print('File: {}'.format(path))
    with h5py.File(path, 'r') as f:
        descend_obj(f[group])
