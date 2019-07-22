import struct
import time
import numpy as np


def read_ptu_header(infile):
    """ Read header from a ptu file and returns a dictionary with all the information

    Parameters
    ----------
    infile: `str`
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
    inputfile.close()
    return header


def read_ptu_records(params):
    infile = params[0]
    header = params[1]
    skip_records = params[2]
    read_records = params[3]
    if read_records is None:
        read_records = header["TTResult_NumberOfRecords"]
    inputfile = open(infile, "rb")
    shift = skip_records * 4
    inputfile.seek(header["Header_End_Bytes"] + shift, 0)
    buffer = inputfile.read(4 * read_records)
    records = np.frombuffer(buffer, dtype="uint32", count=read_records)
    sync = np.bitwise_and(records, 2 ** 10 - 1)  # Lowest 10 bits
    tcspc = np.bitwise_and(
        np.right_shift(records, 10), 2 ** 15 - 1
    )  # Next 15 bits, dtime can be obtained from header
    chan = np.bitwise_and(np.right_shift(records, 25), 2 ** 6 - 1)  # Next 6 bits
    special = np.bitwise_and(records, 2 ** 31) > 0  # Last bit for special markers
    index = (special * 1) * ((chan == 63) * 1)  # Find overflow locations
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
    header["flimview"]["xpix"] = header["ImgHdr_PixX"]
    header["flimview"]["ypix"] = header["ImgHdr_PixY"]
    header["flimview"]["tpix"] = np.max(tcspc) - np.min(tcspc) + 1
    header["flimview"]["tresolution"] = header["MeasDesc_Resolution"]
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
):
    out = []
    xpix = header["flimview"]["ypix"]
    ypix = header["flimview"]["xpix"]
    tpix = header["flimview"]["tpix"]
    tresolution = header["flimview"]["tresolution"]
    im1 = np.zeros((xpix + 1, ypix, tpix))
    i = start_frames[view * frames_per_view + frame_shift] + 1
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
            tc = tcspc[i]  # // 2
            try:
                im1[currentLine][currentPixel][tc] += 1
            except:
                out.append([currentLine, currentPixel])
        i += 1  # next entry
        if sp == 4:  # next frame
            currentLine = 0
            print(f"frame {frame+frame_shift}, view {view} done")
            frame += 1
        if frame == nframes:
            break
    return im1[1:,][:][:]  # remove extra column
