"Architectures for convolutional layers"

from ._base import BaseGCNStack, _GCNStackMeta
from ._gin import GINConvStack
from ._sage import SAGEConvStack

__all__ = ["BaseGCNStack", "SAGEConvStack", "GINConvStack"]

# TODO: eventually migrate out DGL?
