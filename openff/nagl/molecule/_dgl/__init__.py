from typing import Union

from .batch import DGLMoleculeBatch
from .molecule import DGLMolecule

DGLMoleculeOrBatch = Union[DGLMolecule, DGLMoleculeBatch]
