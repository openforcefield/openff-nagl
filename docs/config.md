# Training a GNN using config files

The configuration files defined in `openff.nagl.config` are expected
to be the main way a user will train a GNN. They are split into
three sections, which are all combined in a [`TrainingConfig`]. This document gives a broad outline of the config classes; please see examples for how to train a GNN.

## Model config

A model is defined using the classes in `openff.nagl.config.model`.
A model is expected to consist of a single [`ConvolutionModule`],
and any number of [`ReadoutModule`]s. The readout modules are
registered by name so multiple properties can be predicted from a single
GNN.

Moreover, the atom and bond features used to featurize a molecule
are defined in the model config.


## Data config

The datasets used for training, validation, and testing are defined here.
As this class is only used for training or testing a model, a [`DatasetConfig`]
must also define training targets and loss metrics.

## Optimizer config

Here is where the optimizer is configured for training the GNN.

## Training config

The model, data, and optimizer configs are combined in a [`TrainingConfig`] that is then used to create a [`TrainingGNNModel`] and [`DGLMoleculeDataModule`]
that can be passed to a Pytorch Lightning trainer.