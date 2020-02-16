import logging
import os
import sys
import tarfile
import time
from queue import Queue

import imageio
import pandas as pd
from google.cloud import storage

from data_engineering.archive_etler.uploaders import upload_tiff_and_json_files, upload_png_files
from data_engineering.dask_image_stats_collector import stats_for_numpy_images, stats_for_tiff_images
from data_engineering.data_aggregator.image_aggregators import image_files_from_tif_to_npy, image_files_from_tif_to_augmented_png
from data_engineering.data_aggregator.metadata_aggregators import metadata_files_from_json_to_csv
from data_engineering.gcs_stream_downloader import GCSObjectStreamDownloader


def main():
    """
    Downloads tarfile from $GCS_BUCKET_NAME/$GCS_TARFILE_BLOB_NAME, extracts tarfile to $DISK_PATH, and then
    traverses files in $DISK_PATH/$UNCOMPRESSED_BLOB_PREFIX. If $SHOULD_UPLOAD_TIFF_AND_JSON_FILES,
    uploads tiff and json files to gcs.
    """
    imageio.plugins.freeimage.download()

    global_start = time.time()
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME", "big_earth")
    png_bucket_name: str = os.environ.get("GCS_PNG_BUCKET_NAME", "big_earth_us_central_1")
    tarfile_blob_name: str = os.environ.get("GCS_TARFILE_BLOB_NAME")
    uncompressed_blob_prefix: str = os.environ.get("UNCOMPRESSED_BLOB_PREFIX")
    should_upload_tiff_and_json_files: bool = os.environ.get("SHOULD_UPLOAD_TIFF_AND_JSON_FILES") == "True"
    should_aggregate_metadata: bool = os.environ.get("SHOULD_AGGREGATE_METADATA") == "True"
    should_aggregate_images_to_npy: bool = os.environ.get("SHOULD_AGGREGATE_IMAGES_TO_NPY") == "True"
    should_compute_band_stats: bool = os.environ.get("SHOULD_COMPUTE_BAND_STATS") == "True"
    should_aggregate_images_to_png: bool = os.environ.get("SHOULD_AGGREGATE_IMAGES_TO_PNG") == "True"
    should_upload_png_files: bool = os.environ.get("SHOULD_UPLOAD_PNG_FILES") == "True"
    disk_path: str = os.environ.get("DISK_PATH")

    gcs_client = storage.Client()
    logger = logging.Logger("archive_etler", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    tarfile_disk_path: str = disk_path + "/" + tarfile_blob_name

    if os.environ.get("SHOULD_DOWNLOAD_TARFILE") == "True":
        start = time.time()
        logger.info(f"Downloading BigEarth tarfile from bucket {bucket_name} and blob {tarfile_blob_name}, saving to "
                    f"{tarfile_disk_path}")

        for blob_name in ["patches_with_cloud_and_shadow.csv", "patches_with_seasonal_snow.csv"]:
            with GCSObjectStreamDownloader(client=gcs_client, bucket_name=bucket_name,
                                           blob_name=blob_name) as gcs_downloader:
                with open(disk_path + "/" + blob_name, 'wb') as fileobj:
                    chunk = gcs_downloader.read()
                    while chunk != b"":
                        fileobj.write(chunk)
                        chunk = gcs_downloader.read()

        with GCSObjectStreamDownloader(client=gcs_client, bucket_name=bucket_name,
                                       blob_name=tarfile_blob_name) as gcs_downloader:
            logger.info(f"tarfile_disk_path: {tarfile_disk_path}")
            with open(tarfile_disk_path, 'wb') as fileobj:
                chunk = gcs_downloader.read()
                while chunk != b"":
                    fileobj.write(chunk)
                    chunk = gcs_downloader.read()
        logger.info(
            f"Downloaded tarfile in {(time.time() - start) / 60} minutes.")

    extraction_path = tarfile_disk_path.replace(".gz", "").replace(".tar", "")
    logger.info(f"extraction_path: {extraction_path}")

    if os.environ.get("SHOULD_EXTRACT_TARFILE") == "True":
        start = time.time()
        with tarfile.open(tarfile_disk_path, 'r') as fileobj:
            fileobj.extractall(path=disk_path)

        # Remove the tarfile to save space
        os.remove(tarfile_disk_path)
        logger.info(
            f"tar extracted from {tarfile_disk_path} to {extraction_path} in {(time.time() - start) / 60} minutes.")

    if should_upload_tiff_and_json_files:
        bucket = gcs_client.bucket(bucket_name)
        # Don't use walk because filenames will have thousands of files. Iterate one by one instead
        filepaths_to_upload = Queue()

        stats = {
            "num_files_uploaded": 0,
            "num_folders_uploaded": 0,
            "checkpoint": 1000
        }

        upload_tiff_and_json_files(logger, filepaths_to_upload, bucket, stats, uncompressed_blob_prefix,
                                   extraction_path)

    num_workers = int(os.environ.get("NUM_WORKERS", 20))
    metadata_path = disk_path + "/metadata"
    if should_aggregate_metadata:
        start = time.time()
        
        metadata_df = metadata_files_from_json_to_csv(logger=logger, num_workers=num_workers,
                                                      csv_files_path=metadata_path,
                                                      cloud_and_snow_csv_dir=disk_path, json_dir=extraction_path)
        logger.info(f"Finished metadata aggregation in {(time.time() - start)} seconds.")
    else:
        metadata_df = pd.read_csv(metadata_path + '/metadata.csv')

    if should_aggregate_images_to_npy:
        start = time.time()
        logger.info(f"Starting npy image aggregation.")
        image_files_from_tif_to_npy(num_workers=num_workers, npy_files_path=disk_path + "/npy_image_files", image_dir=extraction_path,
                                    image_prefixes=metadata_df['image_prefix'].values)
        logger.info(f"Finished npy image aggregation in {(time.time() - start) / 60} minutes.")

    if should_compute_band_stats:
        start = time.time()
        logger.info(f"Starting band statistics computation.")
        # Use training data only to compute band stats
        train_set = pd.read_csv("/app/splits/train.csv", header=None)

        npy_files_path = disk_path + "/npy_image_files"
        npy_filenames = [npy_files_path + "/" + file for file in train_set[0].values]
        stats = stats_for_numpy_images(filenames=npy_filenames)
        stats_filepath = metadata_path + '/band_stats.csv'

        logger.info(f"stats: {stats}")
        stats.to_csv(stats_filepath)
        logger.info(f"Finished band statistics computation {(time.time() - start) / 60} minutes.")

    png_files_path = disk_path + "/png_image_files"
    if should_aggregate_images_to_png:
        start = time.time()
        logger.info(f"Starting png image aggregation.")
        image_files_from_tif_to_augmented_png(png_files_path=png_files_path,
                                               image_dir=extraction_path,
                                               image_prefixes=metadata_df['image_prefix'].values)
        logger.info(f"Finished png image aggregation in {(time.time() - start) / 60} minutes.")

    if should_upload_png_files:
        start = time.time()
        logger.info(f"Starting png image upload.")
        bucket = gcs_client.bucket(png_bucket_name)
        filepaths_to_upload = [png_files_path + "/" + filename for filename in os.listdir(png_files_path)]
        stats = {
            "num_files_uploaded": 0,
            "num_folders_uploaded": 0,
            "checkpoint": 1000
        }
        logger.info(f"{len(filepaths_to_upload)} files to upload.")
        upload_png_files(logger=logger, num_workers=num_workers, filepaths_to_upload=filepaths_to_upload,
                         bucket=bucket, stats=stats)
        logger.info(f"Finished image upload in {(time.time() - start)} seconds.")

    logger.info(f"Finished ETL in {(time.time() - global_start) / 60} minutes.")


if __name__ == "__main__":
    main()
