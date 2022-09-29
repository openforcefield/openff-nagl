import enum

import pathlib
from typing import Union

Pathlike = Union[str, pathlib.Path]


class HybridizationType(enum.Enum):
    OTHER = "other"
    SP = "sp"
    SP2 = "sp2"
    SP3 = "sp3"
    SP3D = "sp3d"
    SP3D2 = "sp3d2"