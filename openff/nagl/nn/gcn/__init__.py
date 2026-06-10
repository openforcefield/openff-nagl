"Architectures for convolutional layers"

from ._base import BaseGCNStack
from ._gin import GINConvStack
from ._sage import SAGEConvStack

__all__ = ["BaseGCNStack", "GINConvStack", "SAGEConvStack"]

# TODO: eventually migrate out DGL?
