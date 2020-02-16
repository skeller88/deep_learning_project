import os
import time
from concurrent.futures import Future, as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Empty, Queue
from typing import List

from google.api_core.retry import Retry


def upload_png_files(logger, num_workers, filepaths_to_upload, bucket, stats):
    google_retry = Retry(deadline=480, maximum=240)

    filepaths_to_upload_queue = Queue()
    for filepath in filepaths_to_upload:
        blob_name: str = os.path.join("png_image_files", filepath.rsplit('/')[-1])
        filepaths_to_upload_queue.put((blob_name, filepath))

    def on_google_retry_error(ex: Exception):
        logger.error("Exception when uploading blob to google cloud.")
        logger.exception(ex)

    def google_cloud_uploader():
        start = time.time()
        while True:
            blob_name, filepath = filepaths_to_upload_queue.get(timeout=5)
            blob = bucket.blob(blob_name)
            content_type = "image/png"
            try:
                google_retry(blob.upload_from_filename(filepath, content_type=content_type),
                             on_error=on_google_retry_error)
            except Exception as ex:
                logger.error(f"Uncaught exception when uploading blob to google cloud.")
                logger.exception(ex)
                filepaths_to_upload_queue.put((blob.name, filepath))
                raise ex
            stats['num_files_uploaded'] += 1

            if stats['num_files_uploaded'] > stats['checkpoint']:
                elapsed = (time.time() - start) / 60
                logger.info(
                    f"Uploaded {stats['num_files_uploaded']} files in {elapsed} minutes, {stats['num_files_uploaded'] / elapsed} files per minute.")
                stats['checkpoint'] += 1000

    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        tasks: List[Future] = []
        for x in range(num_workers):
            tasks.append(executor.submit(google_cloud_uploader))
        logger.info(f"Started {len(tasks)} worker tasks.")

        logger.info("Starting traverse_directory")
        for task in as_completed(tasks):
            if task.exception() is not None:
                if type(task.exception()) == Empty:
                    logger.info("Child thread completed")
                else:
                    logger.error("Child thread failed")
                    logger.exception(task.exception())

    logger.info("Ending upload.")


def upload_tiff_and_json_files(logger, filepaths_to_upload, bucket, stats, uncompressed_blob_prefix, extraction_path):
    google_retry = Retry(deadline=480, maximum=240)

    def on_google_retry_error(ex: Exception):
        logger.error("Exception when uploading blob to google cloud.")
        logger.exception(ex)

    def google_cloud_uploader():
        start = time.time()

        blob_name, filepath, content_type = filepaths_to_upload.get(timeout=30)
        while True:
            blob = bucket.blob(blob_name)
            try:
                google_retry(blob.upload_from_filename(filepath, content_type=content_type),
                             on_error=on_google_retry_error)
            except Exception as ex:
                logger.error(f"Uncaught exception when uploading blob to google cloud.")
                logger.exception(ex)
                filepaths_to_upload.put((blob.name, filepath, content_type))
                raise ex
            stats['num_files_uploaded'] += 1

            if stats['num_files_uploaded'] > stats['checkpoint']:
                elapsed = (time.time() - start) / 60
                logger.info(
                    f"Uploaded {stats['num_files_uploaded']} files in {elapsed} minutes, {stats['num_files_uploaded'] / elapsed} files per minute.")
                stats['checkpoint'] += 1000
            blob_name, filepath, content_type = filepaths_to_upload.get(timeout=5)

    def traverse_directory():
        for subdir_name in os.listdir(extraction_path):
            subdir_path = extraction_path + "/" + subdir_name
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if not filename.startswith("._"):
                        split = filename.rsplit(".")
                        if split[-1] == "tif" and split[-2].endswith(("B02", "B03", "B04")):
                            content_type = "image/tiff"
                            # multiple tiff files per subdirectory
                            blob_name: str = os.path.join(uncompressed_blob_prefix, "tiff", subdir_name, filename)

                            filepath = subdir_path + "/" + filename
                            filepaths_to_upload.put((blob_name, filepath, content_type))
                        elif split[-1] == "json":
                            # one json file per subdirectory
                            blob_name: str = os.path.join(uncompressed_blob_prefix, "json_metadata", filename)

                            filepath = subdir_path + "/" + filename
                            filepaths_to_upload.put((blob_name, filepath, content_type))

    num_workers = int(os.environ.get("NUM_WORKERS", 3))
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        tasks: List[Future] = []
        for x in range(num_workers):
            tasks.append(executor.submit(google_cloud_uploader))
        tasks.append(executor.submit(traverse_directory))
        logger.info(f"Started {len(tasks)} worker tasks.")

        logger.info("Starting traverse_directory")
        for task in as_completed(tasks):
            if task.exception() is not None:
                if type(task.exception()) == Empty:
                    logger.info("Child thread completed")
                else:
                    logger.error("Child thread failed")
                    logger.exception(task.exception())

    logger.info("Ending job")