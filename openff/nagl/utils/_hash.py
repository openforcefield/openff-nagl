import enum
import hashlib
import json
import pathlib
from typing import Any, Dict

from ._types import Pathlike


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        if isinstance(obj, pathlib.Path):
            return str(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, enum.Enum):
            return obj.name
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def hash_file(path: Pathlike) -> str:
    """
    Compute the SHA256 hash of a file.
    """
    if not path:
        return ""

    path = pathlib.Path(path)
    sha256_hash = hashlib.sha256()
    with path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def hash_dict(obj: Dict[str, Any]) -> str:
    string = json.dumps(obj, sort_keys=True, cls=CustomJsonEncoder).encode()
    return hashlib.sha256(string).hexdigest()


def file_digest(fileobj, digest, _bufsize=2**18):
    """Hash the contents of a file-like object. Returns a digest object.

    *fileobj* must be a file-like object opened for reading in binary mode.
    It accepts file objects from open(), io.BytesIO(), and SocketIO objects.
    The function may bypass Python's I/O and use the file descriptor *fileno*
    directly.

    *digest* must either be a hash algorithm name as a *str*, a hash
    constructor, or a callable that returns a hash object.
    """
    # On Linux we could use AF_ALG sockets and sendfile() to archive zero-copy
    # hashing with hardware acceleration.
    digestobj = digest()

    if hasattr(fileobj, "getbuffer"):
        # io.BytesIO object, use zero-copy buffer
        digestobj.update(fileobj.getbuffer())
        return digestobj

    # Only binary files implement readinto().
    if not (
        hasattr(fileobj, "readinto")
        and hasattr(fileobj, "readable")
        and fileobj.readable()
    ):
        raise ValueError(
            f"'{fileobj!r}' is not a file-like object in binary reading mode."
        )

    # binary file, socket.SocketIO object
    # Note: socket I/O uses different syscalls than file I/O.
    buf = bytearray(_bufsize)  # Reusable buffer to reduce allocations.
    view = memoryview(buf)
    while True:
        size = fileobj.readinto(buf)
        if size == 0:
            break  # EOF
        digestobj.update(view[:size])

    return digestobj


def digest_file(file):
    with open(file, "rb") as f:
        try:
            h = hashlib.file_digest(f, "sha256")
        except AttributeError:
            h = file_digest(f, hashlib.sha256)
    return h.hexdigest()
