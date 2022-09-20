GNN Charge Models
==============================
[//]: # (Badges)

| **Latest release** | [![Last release tag](https://img.shields.io/github/release-pre/lilyminium/openff-nagl.svg)](https://github.com/lilyminium/openff-nagl/releases) ![GitHub commits since latest release (by date) for a branch](https://img.shields.io/github/commits-since/lilyminium/openff-nagl/latest)  [![Documentation Status](https://readthedocs.org/projects/openff/nagl/badge/?version=latest)](https://openff-nagl.readthedocs.io/en/latest/?badge=latest)                                                                                            |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**         | [![GH Actions Status](https://github.com/lilyminium/openff-nagl/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/lilyminium/openff-nagl/actions?query=branch%3Amain+workflow%3Agh-ci) [![codecov](https://codecov.io/gh/lilyminium/openff-nagl/branch/main/graph/badge.svg)](https://codecov.io/gh/lilyminium/openff-nagl/branch/main) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/lilyminium/openff-nagl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lilyminium/openff-nagl/context:python) |
| **Community**      | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)                                                                                                                                                                                                                                                                                                                                                                                                                                |

A short description of the project.

GNN Charge Models is bound by a [Code of Conduct](https://github.com/lilyminium/openff-nagl/blob/main/CODE_OF_CONDUCT.md).

### Installation

To build GNN Charge Models from source,
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

The GNN Charge Models source code is hosted at https://github.com/lilyminium/openff-nagl
and is available under the GNU General Public License, version 3 (see the file [LICENSE](https://github.com/lilyminium/openff-nagl/blob/main/LICENSE)).

Copyright (c) 2022, Lily Wang


#### Acknowledgements
 
Project based on the 
[MDAnalysis Cookiecutter](https://github.com/MDAnalysis/cookiecutter-mda) version 0.1.
Please cite [MDAnalysis](https://github.com/MDAnalysis/mdanalysis#citation) when using GNN Charge Models in published work.
