# OpenFF NAGL

A playground for applying graph convolutional networks to molecules, with a focus on learning continuous "atom-type" embeddings and from these classical molecule force field parameters.

## Installation

To build NAGL from source, we highly recommend using virtual environments. If possible, we strongly recommend that you use[Anaconda](https://docs.conda.io/en/latest/) as your package manager.

Ensure that you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Create a virtual environment and activate it:

```
conda create --name openff-nagl
conda activate openff-nagl
```

Install the development and documentation dependencies:

```
conda env update --name openff-nagl --file devtools/conda-envs/test_env.yaml
conda env update --name openff-nagl --file devtools/conda-envs/docs_env.yaml
```

Build this package from source:

```
pip install -e .
```

If you want to update your dependencies (which can be risky!), delete and rebuild the environment or run:

```
conda update --all
```

And when you are finished, you can exit the virtual environment with:

```
conda deactivate
```


:::{toctree}
---
hidden: true
---

Overview <self>
:::

<!-- 
:::{toctree}
---
hidden: true
caption: User Guide
---

::: 
-->

<!--
The autosummary directive renders to rST,
so we must use eval-rst here
-->
:::{eval-rst}
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
:::
