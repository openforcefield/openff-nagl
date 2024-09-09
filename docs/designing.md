(designing_a_model)=

# Designing a GCN

Designing a GCN with NAGL primarily involves creating an instance of the [`ModelConfig`] class.

## In Python

A ModelConfig class can be created can be done straightforwardly in Python.

```python
from openff.nagl.features import atoms, bonds
from openff.nagl import GNNModel
from openff.nagl.nn import gcn
from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.nn import postprocess

from openff.nagl.config.model import (
    ModelConfig,
    ConvolutionModule, ReadoutModule,
    ConvolutionLayer, ForwardLayer,
)
```

First we can specify our desired features.
These should be instances of the feature classes.

```python
atom_features = (
    atoms.AtomicElement(["C", "H", "O", "N", "P", "S"]),
    atoms.AtomConnectivity(),
    ...
)

bond_features = (
    bonds.BondOrder(),
    ...
)
```

Next, we can design convolution and readout modules. For example:

```python
convolution_module = ConvolutionModule(
    architecture="SAGEConv",
    # construct 2 layers with dropout 0 (default),
    # hidden feature size 512, and ReLU activation function
    # these layers can also be individually specified,
    # but we just duplicate the layer 6 times for identical layers
    layers=[
        ConvolutionLayer(
            hidden_feature_size=512,
            activation_function="ReLU",
            aggregator_type="mean"
        )
    ] * 2,
)

# define our readout module/s
# multiple are allowed but let's focus on charges
readout_modules = {
    # key is the name of output property, any naming is allowed
    "charges": ReadoutModule(
        pooling="atoms",
        postprocess="compute_partial_charges",
        # 1 layers
        layers=[
            ForwardLayer(
                hidden_feature_size=512,
                activation_function="ReLU",
            )
        ],
    )
}
```

We can now bring it all together as a `ModelConfig` and create a `GNNModel`.

```python
model_config = ModelConfig(
    version="0.1",
    atom_features=atom_features,
    bond_features=bond_features,
    convolution=convolution_module,
    readouts=readout_modules,
)

model = GNNModel(model_config)
```

## From YAML

Or if you prefer, the same model architecture can be specified as a YAML file:

```yaml
version: '0.1'
convolution:
  architecture: SAGEConv
  layers:
    - hidden_feature_size: 512
      activation_function: ReLU
      dropout: 0
      aggregator_type: mean
    - hidden_feature_size: 512
      activation_function: ReLU
      dropout: 0
      aggregator_type: mean
readouts:
  charges:
    pooling: atoms
    postprocess: compute_partial_charges
    layers:
      - hidden_feature_size: 128
        activation_function: Sigmoid
        dropout: 0
atom_features:
  - name: atomic_element
    categories: ["C", "H", "O", "N", "P", "S"]
  - name: atom_connectivity
    categories: [1, 2, 3, 4, 5, 6]
  - name: atom_hybridization
  - name: atom_in_ring_of_size
    ring_size: 3
  - name: atom_in_ring_of_size
    ring_size: 4
  - name: atom_in_ring_of_size
    ring_size: 5
  - name: atom_in_ring_of_size
    ring_size: 6
bond_features:
  - name: bond_is_in_ring
```

And then loaded into a config using the [`ModelConfig.from_yaml()`] method:

```python
from openff.nagl import GNNModel
from openff.nagl.config import ModelConfig

model = GNNModel(ModelConfig.from_yaml("model.yaml"))
```

Here we'll go through each option, what it means, and where to find the available choices.

(model_features)=
## `atom_features` and `bond_features`

These arguments specify the featurization scheme for the model (see [](featurization_theory)). `atom_features` takes a tuple of features from the [`openff.nagl.features.atoms`] module, and `bond_features` a tuple of features from the [`openff.nagl.features.bonds`] module. Each feature is a class that must be instantiated, possibly with some arguments. Custom features may be implemented by subclassing [`AtomFeature`] or [`BondFeature`]; both share the interface of their base class [`Feature`].

[`openff.nagl.features.atoms`]: openff.nagl.features.atoms
[`openff.nagl.features.bonds`]: openff.nagl.features.bonds
[`AtomFeature`]: openff.nagl.features.atoms.AtomFeature
[`BondFeature`]: openff.nagl.features.bonds.BondFeature
[`Feature`]: openff.nagl.features.Feature

## `convolution_architecture`

The `convolution_architecture` argument specifies the structure of the convolution module. Available options are provided in the [`openff.nagl.config.model`] module. 

[`openff.nagl.nn.gcn`]: openff.nagl.nn.gcn

## Number of Features and Layers

Each module comprises a number of layers that must be individually specified.
For example, a [`ConvolutionModule`] consists of specified [`ConvolutionLayer`]s. A [`ReadoutModule`] consists of specified [`ForwardLayer`]s.

The "convolution" arguments define the update network in the convolution module, and the "readout" the network in the readout module (see [](convolution_theory) and [](readout_theory)). Read the `GNNModel` docstring carefully to determine which layers are considered hidden.

## `activation_function`

The `activation_function` argument defines the activation function used by the readout network (see [](nn_theory)). The activation function used by the convolution network is currently fixed to ReLU. Available activation functions are the variants (attributes) of the [`ActivationFunction`] class. When using the Python API rather than the YAML interface, other activation functions from [PyTorch](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) can be used.

[`ActivationFunction`]: openff.nagl.nn.activation.ActivationFunction

## `postprocess_layer`

`postprocess_layer` specifies a PyTorch [`Module`](torch.nn.Module) that performs post-processing in the readout module (see [](readout_theory)). Post-processing layers inherit from the [`PostprocessLayer`] class and are found in the [`openff.nagl.nn.postprocess`] module.

[`PostprocessLayer`]: openff.nagl.nn.postprocess.PostprocessLayer
[`openff.nagl.nn.postprocess`]: openff.nagl.nn.postprocess
