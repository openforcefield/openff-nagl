# OpenFF NAGL

A playground for applying graph convolutional networks to molecules, with a focus on learning continuous "atom-type" embeddings and from these classical molecule force field parameters.


:::{toctree}
---
hidden: true
---

Overview <self>
getting_started.md
training.md
theory.md
examples.md
toolkit_wrappers.md
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
