"Architectures for convolutional layers"

from ._base import GCNStackMeta
from ._sage import SAGEConvStack
from ._gin import GINConvStack


__all__ = [
    "GCNStackMeta",
    "SAGEConvStack",
    "GINConvStack"
]

# TODO: eventually migrate out DGL?