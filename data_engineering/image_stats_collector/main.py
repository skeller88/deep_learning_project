import logging
import os
import queue
import sys
import time
from concurrent.futures import Future, as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

import gcsfs
import imageio
from google.api_core.retry import Retry
from google.cloud import storage

gcs_client = storage.Client()
bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
disk_path: str = os.environ.get("DISK_PATH")
logger = logging.Logger(name='logger', level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def on_google_retry_error(ex: Exception):
    logger.error("Exception when uploading blob to google cloud.")
    logger.exception(ex)


fs = gcsfs.GCSFileSystem(project='big_earth')

google_retry = Retry(deadline=480, maximum=240)

image_paths = queue.Queue()

stats = {
    "pixel_sum": 0,
    "num_images": 0,
}


def get_image_sum_from_gcs():
    image_path = image_paths.get(timeout=30)
    r = google_retry(fs.cat(image_path), on_error=on_google_retry_error)
    img = imageio.core.asarray(imageio.imread(r, 'TIFF'))
    stats['pixel_sum'] += img.sum()
    stats['num_images'] += 1

    if stats['num_images'] % 100000 == 0:
        logger.info(f"Time elapsed: {(time.time() - start) / 60} seconds. stats: {stats}")


start = time.time()

# filenames = fs.ls("big_earth/raw_rgb/tiff")
filenames = fs.ls("big_earth/raw_test")
load_filenames = time.time()
for path in filenames:
    for band in ["B02", "B03", "B04"]:
        image_path = f"{path}{path.split('/')[-2]}_{band}.tif"
        image_paths.put(image_path)

logger.info(f"Read filenames in {load_filenames - start} seconds")

num_workers = int(os.environ.get("NUM_WORKERS", 3))
with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
    tasks: List[Future] = []
    for x in range(num_workers):
        tasks.append(executor.submit(get_image_sum_from_gcs))
    logger.info(f"Started {len(tasks)} worker tasks.")

    for task in as_completed(tasks):
        if task.exception() is not None:
            if type(task.exception()) == queue.Empty:
                logger.info("Child thread completed")
            else:
                logger.error("Child thread failed")
                logger.exception(task.exception())

    logger.info(f"stats: {stats}")
    logger.info(f"Ending job in {(time.time() - start) / 60} minutes")
