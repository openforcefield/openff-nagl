# OpenFF NAGL

A playground for applying graph convolutional networks to molecules, with a focus on learning continuous "atom-type" embeddings and from these classical molecule force field parameters.

## Getting started

OpenFF recommends using Conda virtual environments for all scientific Python work. NAGL can be installed into a new Conda environment named `nagl` with the [`openff-nagl`] package:

```shell
mamba create -n nagl -c conda-forge openff-nagl
conda activate nagl
```

For more information on installing NAGL, see [](installation.md).

NAGL can then be imported from the [`openff.nagl`] module:

```python
import openff.nagl
```

Or executed from the command line:

```shell
openff-nagl --help
```

[`openff-nagl`]: https://anaconda.org/conda-forge/openff-nagl
[`openff.nagl`]: openff.nagl

(inference)=
## Inference with NAGL

NAGL GNN models are used via the [`openff.nagl.GNNModel`] class. A checkpoint file produced by NAGL can be loaded with the [`GNNModel.load()`] method:

```python
from openff.nagl import GNNModel

model = GNNModel.load("trained_model.pt")
```

Then, the properties the model is trained to predict can be computed with the [`GNNModel.compute_properties()`] method, which takes an OpenFF [`Molecule`] object:

```python
from openff.toolkit import Molecule

ethanol = Molecule.from_smiles("CCO")

model.compute_property(ethanol)
```

[`openff.nagl.GNNModel`]: openff.nagl.GNNModel
[`GNNModel.load()`]: openff.nagl.GNNModel.load
[`GNNModel.compute_properties()`]: openff.nagl.GNNModel.compute_properties
[`Molecule`]: openff.toolkit.topology.Molecule

:::{toctree}
---
hidden: true
---

Overview <self>
installation.md
theory.md
designing.md
training.md
examples.md
:::

:::{toctree}
---
hidden: true
caption: Developer's Guide
---

CHANGELOG.md
dev.md
toolkit_wrappers.md
:::

:::{toctree}
---
hidden: true
caption: CLI Reference
---

cli.md
:::


<!--
The autosummary directive renders to rST,
so we must use eval-rst here
-->
```{eval-rst}
.. raw:: html

    <div style="display: None">

.. autosummary::
   :recursive:
   :caption: API Reference
   :toctree: api/generated
   :nosignatures:

   openff.nagl

.. raw:: html

    </div>
```
