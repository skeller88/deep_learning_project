import contextlib
import time
from io import BytesIO

from google.auth.transport.requests import AuthorizedSession
from google.resumable_media import requests, common
from google.cloud import storage


class GCSObjectStreamUploader(contextlib.AbstractContextManager, BytesIO):
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

        # stream (IO[bytes]): The stream (i.e. file-like object) to be uploaded during transmit_next_chunk
        self._stream = b''

        # total_bytes (Optional[int]): The (expected) total number of bytes in the ``stream`` that are ready to be read.
        self._total_bytes: int = 0

        # chunk_size (int): The size of the chunk to be read from the ``stream``
        self._chunk_size: int = chunk_size
        self._bytes_read_from_stream: int = 0
        # checkpoint every 1GB
        self._bytes_read_from_stream_checkpoint = 1e6

        self._transport = AuthorizedSession(credentials=self._client._credentials)
        self._request: requests.ResumableUpload = None

    def __enter__(self):
        url: str = f"https://www.googleapis.com/upload/storage/v1/b/{self._bucket.name}/o?uploadType=resumable"
        self._request = requests.ResumableUpload(
            upload_url=url, chunk_size=self._chunk_size
        )
        self._request.initiate(
            transport=self._transport,
            content_type='application/octet-stream',
            stream=self,
            stream_final=False,
            metadata={'name': self._blob.name},
        )
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self._request.transmit_next_chunk(self._transport)

    def write(self, data: bytes) -> int:
        data_len = len(data)
        self._total_bytes += data_len
        self._stream += data
        del data
        while self._total_bytes >= self._chunk_size:
            try:
                # Calls google.resumable_media._upload.get_next_chunk, which gets the next chunk using
                # self._stream, self._chunk_size, self._total_bytes.
                self._request.transmit_next_chunk(self._transport)
            except common.InvalidResponse:
                self._request.recover(self._transport)
        return data_len

    def read(self, chunk_size: int) -> bytes:
        """
        Used by google.resumable_media._upload.get_next_chunk to slice the next chunk of bytes from the stream.

        memoryview avoids loading a slice into memory.
        :param chunk_size:
        :return:
        """
        bytes_to_read: int = min(chunk_size, self._total_bytes)
        memview = memoryview(self._stream)
        self._stream = memview[bytes_to_read:].tobytes()
        self._bytes_read_from_stream += bytes_to_read

        # Checkpoint upload progress
        if self._bytes_read_from_stream > self._bytes_read_from_stream_checkpoint:
            minutes_elapsed = (time.time() - self.start_time) / 60
            print(f"Read {self._bytes_read_from_stream / 1e6} MB from stream in " \
                  f"{minutes_elapsed} minutes. {self._bytes_read_from_stream / 1e6 / minutes_elapsed} MB per minute.")
            self._bytes_read_from_stream_checkpoint += 1e6

        self._total_bytes -= bytes_to_read
        return memview[:bytes_to_read].tobytes()

    def tell(self) -> int:
        """
        Used by google.resumable_media._upload.get_next_chunk to find the start_byte and end_byte of a chunk.
        :return:
        """
        return self._bytes_read_from_stream
