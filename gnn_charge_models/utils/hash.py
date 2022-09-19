import hashlib
import json
import pathlib
from typing import Any, Dict

from .types import Pathlike


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
    string = json.dumps(obj, sort_keys=True).encode()
    return hashlib.sha256(string).hexdigest()
