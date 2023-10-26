NAGL
==============================
[//]: # (Badges)

| **Latest release** | [![Last release tag](https://img.shields.io/github/release-pre/openforcefield/openff-nagl.svg)](https://github.com/openforcefield/openff-nagl/releases) ![GitHub commits since latest release (by date) for a branch](https://img.shields.io/github/commits-since/openforcefield/openff-nagl/latest)  [![Documentation Status](https://readthedocs.org/projects/openff-nagl/badge/?version=latest)](https://docs.openforcefield.org/projects/nagl/en/latest/?badge=latest)                                                                                                        |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**         | [![GH Actions Status](https://github.com/openforcefield/openff-nagl/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/openforcefield/openff-nagl/actions?query=branch%3Amain+workflow%3Agh-ci) [![codecov](https://codecov.io/gh/openforcefield/openff-nagl/branch/main/graph/badge.svg)](https://codecov.io/gh/openforcefield/openff-nagl/branch/main) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/openforcefield/openff-nagl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/openforcefield/openff-nagl/context:python) |
| **Community**      | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

A playground for applying graph convolutional networks to molecules, with a focus on learning continuous "atom-type" embeddings and from these classical molecule force field parameters.

**Note:** This project is still in development and liable to substantial API and other changes.

This framework is mostly based upon the [*End-to-End Differentiable Molecular Mechanics Force Field Construction*](https://arxiv.org/abs/2010.01196) 
preprint by Wang, Fass and Chodera.

NAGL is bound by a [Code of Conduct](https://github.com/openforcefield/openff-nagl/blob/main/CODE_OF_CONDUCT.md).

### [Documentation](https://docs.openforcefield.org/projects/nagl/en/latest/?badge=latest)

See our documentation for notes on installation, basic usage, and so on!

### Installation

To build NAGL from source,
we highly recommend using virtual environments.
If possible, we strongly recommend that you use
[Anaconda](https://docs.conda.io/en/latest/) as your package manager.


Ensure that you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Create a virtual environment and activate it:

```
conda create --name openff-nagl
conda activate openff-nagl
```

Install the development and documentation dependencies:

```
conda env update --name openff-nagl --file devtools/conda-envs/test_env.yaml
conda env update --name openff-nagl --file docs/requirements.yaml
```

Build this package from source:

```
pip install -e .
```

If you want to update your dependencies (which can be risky!), run:

```
conda update --all
```

And when you are finished, you can exit the virtual environment with:

```
conda deactivate
```

### Copyright

The NAGL source code is hosted at https://github.com/openforcefield/openff-nagl
and is available under the MIT license (see the file [LICENSE](https://github.com/openforcefield/openff-nagl/blob/main/LICENSE)). Some parts inherit from code distributed under other licenses, as detailed in [LICENSE-3RD-PARTY](https://github.com/openforcefield/openff-nagl/blob/main/LICENSE-3RD-PARTY)).

NAGL inherits from Simon Boothroyd's NAGL library at https://github.com/SimonBoothroyd/nagl.
