import contextlib
import time
from io import BytesIO

from google.auth.transport.requests import AuthorizedSession
from google.resumable_media import requests, common
from google.cloud import storage

class GCSObjectStreamDownloader(contextlib.AbstractContextManager, BytesIO):
    """
    From https://dev.to/sethmichaellarson/python-data-streaming-to-google-cloud-storage-with-resumable-uploads-458h
    """

    def __init__(self, client: storage.Client, bucket_name: str, blob_name: str, chunk_size: int = 200 * 256 * 1024):
        """
        :param client:
        :param bucket_name:
        :param blob_name:
        :param chunk_size: Must be a multiple of google/resumable_media/common.py#UPLOAD_CHUNK_SIZE
        """
        super().__init__()
        self.start_time = time.time()

        self._client = client
        self._bucket = self._client.bucket(bucket_name)
        self._blob = self._bucket.blob(blob_name)

        # stream (IO[bytes]): The stream (i.e. file-like object) to be uploaded during consume_next_chunk
        self._stream = b''

        # total_bytes (Optional[int]): The (expected) total number of bytes in the ``stream`` that are ready to be read.
        self._total_bytes: int = 0

        # chunk_size (int): The size of the chunk to be read from the ``stream``
        self._chunk_size: int = chunk_size
        # checkpoint every 1GB
        self._bytes_downloaded_checkpoint = 1e6

        self._transport = AuthorizedSession(credentials=self._client._credentials)
        self._request: requests.ChunkedDownload = None

    def __enter__(self):
        url: str = f"https://www.googleapis.com/download/storage/v1/b/{self._bucket.name}/o/{self._blob.name}?alt=media"
        self._request = requests.ChunkedDownload(
            media_url=url, chunk_size=self._chunk_size, stream=self
        )
        return self

    def write(self, data: bytes) -> int:
        """
        Called by google.resumable_media.requests.download.Download#_write_to_stream after the latest chunk is
        downloaded.
        :param data:
        :return:
        """
        data_len = len(data)
        self._total_bytes += data_len
        self._stream += data
        del data
        return data_len

    def read(self) -> bytes:
        """
        memoryview avoids loading a slice into memory.
        :return:
        """
        if not self._request.finished:
            try:
                # Calls google.resumable_media._download.consume_next_chunk, which gets the next chunk using
                # self._stream, self.bytes_downloaded
                self._request.consume_next_chunk(self._transport)
            except common.InvalidResponse as ex:
                print('InvalidResponse')
                raise ex

        bytes_to_read: int = min(self._chunk_size, self._total_bytes)
        memview = memoryview(self._stream)
        self._stream = memview[bytes_to_read:].tobytes()

        # Checkpoint download progress
        if self._request.bytes_downloaded > self._bytes_downloaded_checkpoint:
            minutes_elapsed = (time.time() - self.start_time) / 60
            print(f"Read {self._request.bytes_downloaded / 1e6} MB from stream in " \
                  f"{minutes_elapsed} minutes. {self._request.bytes_downloaded / 1e6 / minutes_elapsed} MB per minute.")
            self._bytes_downloaded_checkpoint += 1e6

        self._total_bytes -= bytes_to_read
        return memview[:bytes_to_read].tobytes()
