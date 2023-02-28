# Installation

OpenFF recommends using Conda virtual environments for all scientific Python work. OpenFF NAGL can be installed automatically from the open source [Conda Forge] channel; if you do not yet have Conda, we recommend installing the [MambaForge] distribution, which includes the faster Mamba package manager and is pre-configured to work with Conda Forge.

NAGL can be installed into a new Conda environment named `nagl` with the `openff-nagl` package:

```shell
mamba create -n nagl -c conda-forge openff-nagl
conda activate nagl
```

We recommend keeping environments minimal, and only installing packages you use together. Environments can be safely discarded when you no longer need them. This avoids dependency conflicts common to large Python environments. If you prefer, NAGL may be installed into the current environment:

```shell
mamba install -c conda-forge openff-nagl
```

Conda environments that use packages from Conda Forge alongside packages from the default Conda channels run the risk of breaking when an installation or update is attempted. This most commonly happens when a user forgets the `-c conda-forge` switch when installing a package or updating an environment. When this happens, Conda attempts to install or update from the default channels, and may replace shared dependencies of already installed packages with incompatible versions from the default channels.

For this reason, we recommend installing Conda via [MambaForge], which uses Conda Forge for all transactions and excludes packages from the default channels unless they are unavailable in Forge. If you are using a standard Conda installation, we recommend you at minimum configure Forge environments similarly:

```shell
# Remove the --env switch to apply these settings globally
conda activate nagl
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict 
```

In environments with this configuration, the `-c conda-forge` switch is unnecessary. Other channels, like `psi4` and `bioconda`, can still be used in the usual way.

More information on installing OpenFF packages can be found in the [OpenFF Toolkit documentation](openff.toolkit:installation).

[Conda Forge]: https://conda-forge.org/
[MambaForge]: https://github.com/conda-forge/miniforge#mambaforge


