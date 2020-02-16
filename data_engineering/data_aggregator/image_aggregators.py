import os

import imageio
import numpy as np
from PIL import Image

from data_engineering.data_aggregator.parallelize import parallelize_task


def image_files_from_tif_to_npy(num_workers, npy_files_path, image_dir, image_prefixes):
    if not os.path.exists(npy_files_path):
        os.mkdir(npy_files_path)

    def image_to_npy(image_prefix):
        bands = [np.asarray(
            Image.open(f"{image_dir}/{image_prefix}/{image_prefix}_B{band}.tif"),
            dtype=np.uint16) for band in ["02", "03", "04"]]

        stacked_arr = np.stack(bands, axis=-1)
        np.save(f"{npy_files_path}/{image_prefix}", stacked_arr)

    def images_to_npy(image_prefixes):
        for image_prefix in image_prefixes:
            image_to_npy(image_prefix)

    parallelize_task(num_workers=num_workers, task=images_to_npy, iterator=image_prefixes)


def image_files_from_tif_to_augmented_png(png_files_path, image_dir, image_prefixes,
                                          image_suffix, augmentations):
    if not os.path.exists(png_files_path):
        os.mkdir(png_files_path)

    def image_to_png(image_prefix):
        bands = [np.asarray(
            Image.open(f"{image_dir}/{image_prefix}/{image_prefix}_B{band}.tif"),
            dtype=np.uint16) for band in ["02", "03", "04"]]

        stacked_arr = np.stack(bands, axis=-1)

        if augmentations is not None:
            augmented_arr = augmentations(image=stacked_arr)['image']
        else:
            augmented_arr = stacked_arr
        imageio.imwrite(im=augmented_arr, uri=f"{png_files_path}/{image_prefix}{image_suffix}.png",
                        format='PNG-FI')

    for image_prefix in image_prefixes:
        image_to_png(image_prefix)
