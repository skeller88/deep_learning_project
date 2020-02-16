from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account


def main(bucket_name, blob_name):
    credentials = service_account.Credentials.from_service_account_file(
        "/Users/shanekeller/.gcs/credentials.json")
    gcs_client = storage.Client(credentials=credentials)
    _bucket = gcs_client.bucket(bucket_name)
    _blob = _bucket.blob(blob_name)

    with open(Path.home() / 'Documents/big_earth_springboard_project/test_downloaded.tif', 'wb') as fileobj:
        resp = _blob.download_as_string()
        fileobj.write(resp)


bucket_name: str = "big_earth"
blob_name: str = "raw/S2A_MSIL2A_20170717T113321_10_80/S2A_MSIL2A_20170717T113321_10_80_B01.tif"

main(bucket_name, blob_name)
