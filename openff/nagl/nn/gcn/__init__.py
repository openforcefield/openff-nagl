"Architectures for convolutional layers"

from .base import GCNStackMeta
from .sage import SAGEConvStack
from .gin import GINConvStack


__all__ = [
    "GCNStackMeta",
    "SAGEConvStack",
    "GINConvStack"
]