# Getting started

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

## Inference with NAGL

