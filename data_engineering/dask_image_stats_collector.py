from collections import defaultdict

import dask
import dask.array as da
import numpy as np
import pandas as pd
import rasterio


def stats_for_numpy_images(filenames, use_test_data=False):
    def read_image(filename, use_test_data):
        if use_test_data:
            band_shape = (120, 120)
            return np.stack([np.full(band_shape, num) for num in [1, 2, 3]], axis=-1)
        return np.load(filename)

    delayed_read = dask.delayed(read_image, pure=True)
    lazy_images = [da.from_delayed(delayed_read(filename, use_test_data), dtype=np.uint16, shape=(120, 120, 3))
                   for filename in filenames]

    stack = da.stack(lazy_images, axis=0)
    stack = stack.reshape(-1, stack.shape[-1]).rechunk('auto')

    stats = defaultdict(dict)
    for stat_name, stat_func in [('mean', da.mean), ('std', da.std), ('min', da.min), ('max', da.max)]:
        stats[stat_name] = stat_func(stack, axis=0).compute()

    return pd.DataFrame(stats, index=['red', 'blue', 'green'])


def stats_for_tiff_images(filenames, use_test_data=False):
    def read_image(filename, band, use_test_data):
        filename = filename.format(band)
        if use_test_data:
            num = int(band)
            band_shape = (120, 120)
            return np.full(band_shape, num)
        band_ds = rasterio.open(filename)
        return np.array(band_ds.read(1))

    def images_for_band(band):
        delayed_read = dask.delayed(read_image, pure=True)
        lazy_images = [da.from_delayed(delayed_read(filename, band, use_test_data), dtype=np.uint16, shape=(120, 120))
                       for filename in filenames]

        stack = da.stack(lazy_images, axis=0).rechunk('auto')
        return stack.flatten()

    all_bands = da.stack([images_for_band("02"), images_for_band("03"), images_for_band("04")], axis=-1)

    stats = defaultdict(dict)
    for stat_name, stat_func in [('mean', da.mean), ('std', da.std), ('min', da.min), ('max', da.max)]:
        stats[stat_name] = stat_func(all_bands, axis=0).compute()

    return pd.DataFrame(stats, index=['red', 'blue', 'green'])