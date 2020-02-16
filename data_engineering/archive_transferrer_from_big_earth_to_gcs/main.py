import os
from pathlib import Path

import requests
from google.cloud import storage

from data_engineering.gcs_stream_uploader import GCSObjectStreamUploader


def main():
    """
    Download raw .tar.gz file from BigEarthNet and upload to Google Cloud Storage. Repeat for seasonal snow and
    seasonal cloud .csv files.
    :return:
    """
    gcs_client = storage.Client()
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
    blob_name: str = os.environ.get("GCS_BLOB_NAME")

    print(f"Uploading BigEarth to bucket {bucket_name} and blob {blob_name}")

    for url, blob_name in [
        ("http://bigearth.net/downloads/BigEarthNet-v1.0.tar.gz", blob_name),
        ("http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv",
         "patches_with_cloud_and_shadow.csv"),
        ("http://bigearth.net/static/documents/patches_with_snow.csv", "patches_with_seasonal_snow.csv"),
    ]:
        with GCSObjectStreamUploader(client=gcs_client, bucket_name=bucket_name, blob_name=blob_name) as gcs_uploader:
            with requests.get(url, stream=True) as response_stream:
                for chunk in response_stream.raw.stream(128*2000, decode_content=False):
                    gcs_uploader.write(chunk)


def local_file_test():
    gcs_client = storage.Client()
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
    blob_name: str = "test-blob"

    with GCSObjectStreamUploader(client=gcs_client, bucket_name=bucket_name, blob_name=blob_name) as gcs_uploader:
        for line in open(str(Path.home()) + '/Documents/big_earth_deep_learning_project/test_archive.tar.gz', 'rb'):
            gcs_uploader.write(line)


if __name__ == "__main__":
    print('running')
    main()
