"""Reading images from ccv files, ported from Matlab"""

import os
import math
import numpy as np
#import data.unpackbits as upb # Arne edit 

VERBOSE = False  # NOTE: For debugging

def get_header(f):
    """Read out header info of ccv file.

    Parameters
    ----------
    f : string
        Path to file.

    Returns
    -------
    info : dict
        Containing header information.
    """
    info = {}

    with open(f, 'rb') as fid:
        headerbytes = fread(fid, 1, np.uint32)
        assert headerbytes[0] < 1e5, 'invalid header'

        camtype_nchars = fread(fid, 1, np.uint32)
        assert camtype_nchars[0] > 1, 'invalid header'
        assert camtype_nchars[0] < 1e3, 'invalid header'

        camtype_nullterm = fread(fid, camtype_nchars[0] + 1, np.str)
        camtype_nullterm_char = camtype_nullterm.view('c')
        camtype_nullterm_str = camtype_nullterm_char.tostring()[:-1]
        assert camtype_nullterm[-1] == 0, 'invalid header'
        assert camtype_nullterm_str.lower() in (b'basler', b'aptina'), 'invalid camtype'
        if VERBOSE:
            print('camtype: {}'.format(camtype_nullterm_str))

        filetypeversion = fread(fid, 1, np.float64)
        assert filetypeversion[0] > 0, 'invalid header'
        assert filetypeversion[0] < 100, 'invalid header'
        if filetypeversion[0] > 0.13:
            raise ValueError('unrecognized ccv file version')

        imagetype_nchars = fread(fid, 1, np.uint32)
        imagetype_nullterm = fread(fid, imagetype_nchars[0] + 1, np.str)
        assert imagetype_nullterm[-1] == 0, 'invalid header'

        imagetype = imagetype_nullterm.view('c').tostring()[:-1]
        if VERBOSE:
            print('imagetype: {}'.format(imagetype))

        bytesperpixel = fread(fid, 1, np.uint32)
        bitsperpixel = fread(fid, 1, np.uint32)
        framebytesondisk = fread(fid, 1, np.uint32)
        w = fread(fid,1,np.uint32)
        h = fread(fid,1,np.uint32)
        framerate = fread(fid,1,np.float64)

        bitsarepacked = False
        if filetypeversion >= 0.12:
            bitsarepacked = fread(fid, 1, np.str)
            if bitsarepacked[0] != 0:
                bitsarepacked = 1
        if VERBOSE:
            print('bitsarepacked: {}'.format(bitsarepacked[0]))

        nframes = fread(fid, 1, np.uint32)
        if filetypeversion > 0.12:
            offset = fread(fid, 2, np.uint32)
            sensorsize = fread(fid, 2, np.uint32)
            clockrate = fread(fid, 1, np.uint64)
            exposuretime_ms = fread(fid, 1, np.float64)
            gain = fread(fid, 1, np.float64)

        if bytesperpixel[0] == 1:
            pixeldata_dtype = np.uint8
        if bytesperpixel[0] == 2:
            pixeldata_dtype = np.uint16
        if bytesperpixel[0] == 3:
            pixeldata_dtype = np.uint32
        if bytesperpixel[0] == 4:
            pixeldata_dtype = np.uint64

        info['bitsarepacked'] = bitsarepacked[0]
        info['bitsperpixel'] = bitsperpixel[0]
        info['bytesperpixel'] = bytesperpixel[0]
        info['framebytesondisk'] = framebytesondisk[0]
        info['framerate'] = framerate[0]
        info['h'] = h[0]
        info['headerbytes'] = headerbytes[0]
        info['nframes'] = nframes[0]
        info['pixeldata_dtype'] = pixeldata_dtype
        info['sensorsize'] = tuple(sensorsize)
        info['w'] = w[0]
        
        # Arne Monsees edit
        if filetypeversion > 0.12:
            info['offset'] = tuple(offset)
            info['clockrate'] = clockrate[0]
            info['exposuretime_ms'] = exposuretime_ms[0]
            info['gain'] = gain[0]

    return info


def get_frame(f, frameind):
    """Get frame with specified index.

    Parameters
    ----------
    f : string
        Path to file.
    frameind : int
        Index of frame to read out, starting at 1.

    Returns
    -------
    img : array
        Containing the frame.
    """

    info = get_header(f)
    prev_offset = info['headerbytes']
    prev_frame = 1

    new_offset = prev_offset + (frameind-prev_frame) * info['framebytesondisk']
    fileinfobytes = os.path.getsize(f)
    ctrl_bytes = np.array(np.array([info['framebytesondisk']], dtype=np.int64) *
                          np.array([info['nframes']], dtype=np.int64),
                            dtype=np.int64)[0]
    assert fileinfobytes - info['headerbytes'] == ctrl_bytes, 'size mismatch'

    with open(f, 'rb') as fid:
        fid.seek(new_offset  + info['w']*info['h']*info['bytesperpixel'],
                 os.SEEK_SET)  # absolute seek
        #frameindex = fread(fid, 1, np.uint32)
        #assert frameindex == frameind, 'frame not found'
        fid.seek(new_offset, os.SEEK_SET)  # absolute seek
        if info['bitsarepacked'] == 0:
            rawdata = fread(fid, info['w'] * info['h'], np.uint8)\
                        .reshape(info['h'], info['w'])  # NOTE: HEIGHT x WIDTH
            img = rawdata
        else:
            rawdata = fread(fid, math.ceil(info['w'] * info['h'] *
                        info['bitsperpixel'] / 8.0), np.uint8)
            #unpacked = upb.upb_16_10(rawdata) # Arne edit 
            unpacked = np.unpackbits(rawdata)
            if info['bytesperpixel'] == 1:
                img = unpacked.reshape(info['h'], info['w'])
            elif info['bytesperpixel'] == 2:
                img = unpacked[::2] + 256*unpacked[1::2]
                img = img.reshape(info['h'], info['w'])
            else:
                raise ValueError('invalid bytesperpixel')

    return img


def fread(fid, nelements, dtype):
    """Read binary file, wrapping np.fromfile."""
    dt = dtype
    if dtype is np.str:
        dt = np.uint8  # NOTE: assuming 8-bit ASCII for np.str
    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, )
    if VERBOSE:
        print(data_array)
    return data_array
