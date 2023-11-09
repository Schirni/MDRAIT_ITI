import os
import random
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from random import randint
from urllib import request

import astropy.io.ascii
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import pandas as pd
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, AsinhStretch
from dateutil.parser import parse
from scipy import ndimage
from skimage.measure import block_reduce
from skimage.transform import pyramid_reduce
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header, all_coordinates_from_map, header_helper
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from scipy.interpolate import interp2d
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table


class Editor(ABC):

    def convert(self, data, **kwargs):
        result = self.call(data, **kwargs)
        if isinstance(result, tuple):
            data, add_kwargs = result
            kwargs.update(add_kwargs)
        else:
            data = result
        return data, kwargs

    @abstractmethod
    def call(self, data, **kwargs):
        raise NotImplementedError()


hinode_norms = {'continuum': ImageNormalize(vmin=0, vmax=50000, stretch=LinearStretch(), clip=True),
                'gband': ImageNormalize(vmin=0, vmax=25000, stretch=LinearStretch(), clip=True), }

solo_norm = {174: ImageNormalize(vmin=0, vmax=2200, stretch=AsinhStretch(0.005), clip=True),
             304: ImageNormalize(vmin=0, vmax=6500, stretch=AsinhStretch(0.001), clip=True)
             }
proba2_norm = {174: ImageNormalize(vmin=0, vmax=370, stretch=AsinhStretch(0.001), clip=True)}

hri_norm = {174: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=True)}

sdo_norms = {94: ImageNormalize(vmin=0, vmax=340, stretch=AsinhStretch(0.005), clip=True),
             131: ImageNormalize(vmin=0, vmax=1400, stretch=AsinhStretch(0.005), clip=True),
             171: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=True),
             193: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=True),
             211: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=True),
             304: ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.001), clip=True),
             335: ImageNormalize(vmin=0, vmax=600, stretch=AsinhStretch(0.005), clip=True),
             1600: ImageNormalize(vmin=0, vmax=4000, stretch=AsinhStretch(0.005), clip=True),
             1700: ImageNormalize(vmin=0, vmax=4000, stretch=AsinhStretch(0.005), clip=True),
             'mag': ImageNormalize(vmin=-3000, vmax=3000, stretch=LinearStretch(), clip=True),
             'continuum': ImageNormalize(vmin=0, vmax=70000, stretch=LinearStretch(), clip=True),
             }


class MapToDataEditor(Editor):
    def call(self, s_map, **kwargs):
        return s_map.data, {"header": s_map.meta}



class BlockReduceEditor(Editor):

    def __init__(self, block_size, func=np.mean):
        self.block_size = block_size
        self.func = func

    def call(self, data, **kwargs):
        return block_reduce(data, self.block_size, func=self.func)


class NanEditor(Editor):
    def __init__(self, nan=0):
        self.nan = nan

    def call(self, data, **kwargs):
        data = np.nan_to_num(data, nan=self.nan)
        return data



class NormalizeEditor(Editor):
    def __init__(self, norm, **kwargs):
        self.norm = norm

    def call(self, data, **kwargs):
        data = self.norm(data).data * 2 - 1
        return data


class ImageNormalizeEditor(Editor):

    def __init__(self, vmin=None, vmax=None, stretch=LinearStretch()):
        self.norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)

    def call(self, data, **kwargs):
        data = self.norm(data).data * 2 - 1
        return data


class MinMaxQuantileNormalizeEditor(Editor):
    def call(self, data, **kwargs):

        vmin = np.quantile(data, 0.001)
        vmax = np.quantile(data, 0.999)
        #vmax = np.max(data)

        data = (data - vmin) / (vmax - vmin) * 2 - 1
        data = np.clip(data, -1, 1)
        return data

class RemoveOffLimbEditor(Editor):

    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        hpc_coords = all_coordinates_from_map(s_map)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
        s_map.data[r > 1] = self.fill_value
        return s_map


class MinMaxNormalizeEditor(Editor):
    def call(self, data, **kwargs):

        vmin = np.min(data)
        vmax = np.max(data)

        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=True)
        data = norm(data).data * 2 - 1
        return data


class StretchPixelEditor(Editor):
    def call(self, data, **kwargs):

        vmin = np.min(data)
        vmax = np.max(data)

        data = (data - vmin) / (vmax - vmin) * 2 - 1
        return data


class WhiteningEditor(Editor):
    """ Mean value is set to 0 (remove contrast) and std is set to 1"""

    def call(self, data, **kwargs):
        data_mean = np.nanmean(data)
        data_std = np.nanstd(data)
        data = (data - data_mean) / (data_std + 1e-6)
        return data



class ExpandDimsEditor(Editor):
    def __init__(self, axis=0):
        self.axis = axis

    def call(self, data, **kwargs):
        return np.expand_dims(data, axis=self.axis).astype(np.float32)



class DistributeEditor(Editor):
    def __init__(self, editors):
        self.editors = editors

    def call(self, data, **kwargs):
        return np.concatenate([self.convertData(d, **kwargs) for d in data], 0)

    def convertData(self, data, **kwargs):
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data


class BrightestPixelPatchEditor(Editor):
    def __init__(self, patch_shape, idx=0, random_selection=0.2):
        self.patch_shape = patch_shape
        self.idx = idx
        self.random_selection = random_selection

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)

        if random.random() <= self.random_selection:
            x = randint(0, data.shape[1] - self.patch_shape[0])
            y = randint(0, data.shape[2] - self.patch_shape[1])
            patch = data[:, x:x + self.patch_shape[0], y:y + self.patch_shape[1]]
        else:
            smoothed = ndimage.gaussian_filter(data[self.idx], sigma=5)
            pixel_pos = np.argwhere(smoothed == np.nanmax(smoothed))
            pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
            pixel_pos = np.min([pixel_pos[0], smoothed.shape[0] - self.patch_shape[0] // 2]), np.min(
                [pixel_pos[1], smoothed.shape[1] - self.patch_shape[1] // 2])
            pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), np.max(
                [pixel_pos[1], self.patch_shape[1] // 2])

            x = pixel_pos[0]
            y = pixel_pos[1]
            patch = data[:,
                    x - int(np.floor(self.patch_shape[0] / 2)):x + int(np.ceil(self.patch_shape[0] / 2)),
                    y - int(np.floor(self.patch_shape[1] / 2)):y + int(np.ceil(self.patch_shape[1] / 2)), ]
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        return patch


class AIAPrepEditor(Editor):
    def __init__(self, calibration='auto'):
        super().__init__()
        assert calibration in ['aiapy', 'auto', 'none',
                               None], "Calibration must be one of: ['aiapy', 'auto', 'none', None]"
        self.calibration = calibration
        self.table = get_auto_calibration_table() if calibration == 'auto' else get_local_correction_table()

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        if self.calibration == 'auto':
            s_map = self.correct_degradation(s_map, correction_table=self.table)
        elif self.calibration == 'aiapy':
            s_map = correct_degradation(s_map, correction_table=self.table)
        data = np.nan_to_num(s_map.data)
        data = data / s_map.meta["exptime"]
        return Map(data.astype(np.float32), s_map.meta)

    def correct_degradation(self, s_map, correction_table):
        index = correction_table["DATE"].sub(s_map.date.datetime).abs().idxmin()
        num = s_map.meta["wavelnth"]
        return Map(s_map.data / correction_table.iloc[index][f"{int(num):04}"], s_map.meta)


class DarkestPixelPatchEditor(Editor):
    def __init__(self, patch_shape, idx=0, random_selection=0.2):
        self.patch_shape = patch_shape
        self.idx = idx
        self.random_selection = random_selection

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)

        if random.random() <= self.random_selection:
            x = randint(0, data.shape[1] - self.patch_shape[0])
            y = randint(0, data.shape[2] - self.patch_shape[1])
            patch = data[:, x:x + self.patch_shape[0], y:y + self.patch_shape[1]]
        else:
            smoothed = ndimage.gaussian_filter(data[self.idx], sigma=5)
            pixel_pos = np.argwhere(smoothed == (np.nanmin(smoothed)))
            pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
            pixel_pos = np.min([pixel_pos[0], smoothed.shape[0] - self.patch_shape[0] // 2]), np.min(
                [pixel_pos[1], smoothed.shape[1] - self.patch_shape[1] // 2])
            pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), np.max(
                [pixel_pos[1], self.patch_shape[1] // 2])

            x = pixel_pos[0]
            y = pixel_pos[1]
            patch = data[:,
                    x - int(np.floor(self.patch_shape[0] / 2)):x + int(np.ceil(self.patch_shape[0] / 2)),
                    y - int(np.floor(self.patch_shape[1] / 2)):y + int(np.ceil(self.patch_shape[1] / 2)), ]
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        return patch


class LoadFITSEditor(Editor):

    def call(self, map_path, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            hdul = fits.open(map_path)
            hdul.verify("fix")
            data, header = hdul[1].data, hdul[1].header
            hdul.close()
        return data, {"header": header}



class DataToMapEditor(Editor):

    def call(self, data, **kwargs):
        return Map(data[0], kwargs['header'])


class LoadCDFEditor(Editor):

    def call(self, data, **kwargs):
        read = nc.Dataset(data)
        return read


class PaddingEditor(Editor):
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def call(self, data, **kwargs):
        s = data.shape
        p = self.target_shape
        x_pad = (p[0] - s[-2]) / 2
        y_pad = (p[1] - s[-1]) / 2
        pad = [(int(np.floor(x_pad)), int(np.ceil(x_pad))),
               (int(np.floor(y_pad)), int(np.ceil(y_pad)))]
        if len(s) == 3:
            pad.insert(0, (0, 0))
        return np.pad(data, pad, 'constant', constant_values=np.nan)


class UnpaddingEditor(Editor):
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def call(self, data, **kwargs):
        s = data.shape
        p = self.target_shape
        x_unpad = (s[-2] - p[0]) / 2
        y_unpad = (s[-1] - p[1]) / 2
        #
        unpad = [(None if int(np.floor(y_unpad)) == 0 else int(np.floor(y_unpad)),
                 None if int(np.ceil(y_unpad)) == 0 else -int(np.ceil(y_unpad))),
                 (None if int(np.floor(x_unpad)) == 0 else int(np.floor(x_unpad)),
                  None if int(np.ceil(x_unpad)) == 0 else -int(np.ceil(x_unpad)))]
        data = data[:, unpad[0][0]:unpad[0][1], unpad[1][0]:unpad[1][1]]
        return data


class ReshapeEditor(Editor):

    def __init__(self, shape):
        self.shape = shape

    def call(self, data, **kwargs):
        data = data[:self.shape[1], :self.shape[2]]
        return np.reshape(data, self.shape).astype(np.float32)


class ImagePatch(Editor):

    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data, **kwargs):
        assert data.shape[0] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[1] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)

        x = randint(0, data.shape[0] - self.patch_shape[0])
        y = randint(0, data.shape[1] - self.patch_shape[1])
        patch = data[x:x + self.patch_shape[0], y:y + self.patch_shape[1]]

        return patch


class RandomPatchEditor(Editor):
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(data.shape)
        x = randint(0, data.shape[1] - self.patch_shape[0])
        y = randint(0, data.shape[2] - self.patch_shape[1])
        patch = data[:, x:x + self.patch_shape[0], y:y + self.patch_shape[1]]
        patch = np.copy(patch)  # copy from mmep
        assert np.std(patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        assert not np.any(np.isnan(patch)), 'NaN found'
        return patch


class StackEditor(Editor):

    def __init__(self, data_sets):
        self.data_sets = data_sets

    def call(self, idx, **kwargs):
        results = [dp.getIndex(idx) for dp in self.data_sets]
        return np.concatenate([img for img, kwargs in results], 0), {'kwargs_list': [kwargs for img, kwargs in results]}


class ContrastNormalizeEditor(Editor):

    def __init__(self, use_median=False, shift=None, normalization=None):
        self.use_median = use_median
        self.shift = shift
        self.normalization = normalization

    def call(self, data, **kwargs):
        shift = np.mean(data)
        data = (data - shift)
        data = np.clip(data, -1, 1)
        return data


class NormalizeExposureEditor(Editor):
    def __init__(self, target=1 * u.s):
        self.target = target
        super().__init__()

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        data = s_map.data
        data = data / s_map.exposure_time.to(u.s).value * self.target.to(u.s).value
        return Map(data.astype(np.float32), s_map.meta)

class LoadMapEditor(Editor):

    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_map = Map(data)
            s_map.meta['timesys'] = 'tai'  # fix leap seconds
            return s_map, {'path': data}


class ScaleEditor(Editor):
    def __init__(self, arcspp):
        self.arcspp = arcspp
        super(ScaleEditor, self).__init__()

    def call(self, s_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            scale_factor = s_map.scale[0].value / self.arcspp
            new_dimensions = [int(s_map.data.shape[1] * scale_factor),
                              int(s_map.data.shape[0] * scale_factor)] * u.pixel
            s_map = s_map.resample(new_dimensions)

            return Map(s_map.data.astype(np.float32), s_map.meta)


class CropEditor(Editor):
    def __init__(self, start_x, end_x, start_y, end_y):
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y

    def call(self, data, **kwargs):

        crop = data[self.start_x: self.end_x, self.start_y:self.end_y]
        return crop


class ShiftMeanEditor(Editor):
    def call(self, data, **kwargs):
        mean = np.mean(data)
        data = (data - mean)
        data = np.clip(data, -1, 1)
        return data


class NormalizeRadiusEditor(Editor):
    def __init__(self, resolution, padding_factor=0.1, crop=True, **kwargs):
        self.padding_factor = padding_factor
        self.resolution = resolution
        self.crop = crop
        super(NormalizeRadiusEditor, self).__init__(**kwargs)

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # normalize solar radius
        r_obs_pix = (1 + self.padding_factor) * r_obs_pix
        scale_factor = self.resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
        s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=4)
        if self.crop:
            arcs_frame = (self.resolution / 2) * s_map.scale[0].value
            s_map = s_map.submap(bottom_left=SkyCoord(-arcs_frame * u.arcsec, -arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
                                 top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
            pad_x = s_map.data.shape[0] - self.resolution
            pad_y = s_map.data.shape[1] - self.resolution
            s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                                 top_right=[pad_x // 2 + self.resolution - 1, pad_y // 2 + self.resolution - 1] * u.pix)
        s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']
        return s_map



def get_local_correction_table():
    path = os.path.join(Path.home(), 'aiapy', 'correction_table.dat')
    if os.path.exists(path):
        return get_correction_table(path)
    os.makedirs(os.path.join(Path.home(), 'aiapy'), exist_ok=True)
    correction_table = get_correction_table()
    astropy.io.ascii.write(correction_table, path)
    return correction_table


def get_auto_calibration_table():
    table_path = os.path.join(Path.home(), '.iti', 'sdo_autocal_table.csv')
    os.makedirs(os.path.join(Path.home(), '.iti'), exist_ok=True)
    if not os.path.exists(table_path):
        request.urlretrieve('http://kanzelhohe.uni-graz.at/iti/sdo_autocal_table.csv', filename=table_path)
    return pd.read_csv(table_path, parse_dates=['DATE'], index_col=0)
