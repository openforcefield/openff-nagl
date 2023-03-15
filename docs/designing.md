# Designing a GCN

Designing a GCN with NAGL primarily involves creating an instance of the [`GNNModel`] class. This can be done straightforwardly in Python:

```python
from openff.nagl.features import atoms, bonds
from openff.nagl import GNNModel
from openff.nagl.nn import gcn
from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.nn import postprocess

atom_features = (
    atoms.AtomicElement(["C", "H", "O", "N", "P", "S"]),
    atoms.AtomConnectivity(),
    ...
)

bond_features = (
    bonds.BondOrder(),
    ...
)

model = GNNModel(
    atom_features=atom_features,
    bond_features=bond_features,
    convolution_architecture=gcn.SAGEConvStack,
    n_convolution_hidden_features=128,
    n_convolution_layers=3,
    n_readout_hidden_features=128,
    n_readout_layers=4,
    activation_function=ActivationFunction.ReLU,
    postprocess_layer=postprocess.ComputePartialCharges,
    readout_name=f"am1bcc-charges",
    learning_rate=0.001,
)
```

Or if you prefer, the same model architecture can be specified as a YAML file:

```yaml
convolution_architecture: SAGEConv
postprocess_layer: compute_partial_charges

activation_function: ReLU
learning_rate: 0.001
n_convolution_hidden_features: 128
n_convolution_layers: 3
n_readout_hidden_features: 128
n_readout_layers: 4

atom_features:
  - AtomicElement:
      categories: ["C", "H", "O", "N", "P", "S"]
  - AtomConnectivity
  - AtomAverageFormalCharge
  - AtomHybridization
  - AtomInRingOfSize: 3
  - AtomInRingOfSize: 4
  - AtomInRingOfSize: 5
  - AtomInRingOfSize: 6
bond_features:
  - BondOrder
  - BondInRingOfSize: 3
  - BondInRingOfSize: 4
  - BondInRingOfSize: 5
  - BondInRingOfSize: 6

```

And then loaded with the [`GNNModel.from_yaml_file()`] method:

```python
from openff.nagl import GNNModel

model = GNNModel.from_yaml_file("model.yml")
```

Here we'll go through each option, what it means, and where to find the available choices.

[`GNNModel`]: openff.nagl.GNNModel
[`GNNModel.from_yaml_file()`]: openff.nagl.GNNModel.from_yaml_file 

(model_features)=
## `atom_features` and `bond_features`

These arguments specify the featurization scheme for the model (see [](featurization_theory). `atom_features` takes a tuple of features from the [`openff.nagl.features.atoms`] module, and `bond_features` a tuple of features from the [`openff.nagl.features.bonds`] module. Each feature is a class that must be instantiated, possibly with some arguments.

[`openff.nagl.features.atoms`]: openff.nagl.features.atoms
[`openff.nagl.features.bonds`]: openff.nagl.features.bonds
[`Feature`]: openff.nagl.features.Feature

## `convolution_architecture`

The `convolution_architecture` argument specifies the structure of the convolution module. Available options are provided in the [`openff.nagl.nn.gcn`] module. 

[`openff.nagl.nn.gcn`]: openff.nagl.nn.gcn

## Number of Features and Layers

The size and shape of the neural networks in the convolution and readout modules are specified by four arguments:

- `n_convolution_hidden_features`
- `n_convolution_layers`
- `n_readout_hidden_features`
- `n_readout_layers`

The "convolution" arguments define the update network in the convolution module, and the "readout" the network in the readout module (see [](convolution_theory) and [](readout_theory)). Read the `GNNModel` docstring carefully to determine which layers are considered hidden.

## `activation_function`

The `activation_function` argument defines the activation function used by the readout network (see [](nn_theory)). The activation function used by the convolution network is currently fixed to ReLU. Available activation functions are the variants (attributes) of the [`ActivationFunction`] class. When using the Python API rather than the YAML interface, other activation functions from [PyTorch](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) can be used.

[`ActivationFunction`]: openff.nagl.nn.activation.ActivationFunction

## `postprocess_layer`

`postprocess_layer` specifies a PyTorch [`Module`](torch.nn.Module) that performs post-processing in the readout module (see [](readout_theory)). Post-processing layers inherit from the [`PostprocessLayer`] class and are found in the [`openff.nagl.nn.postprocess`] module.

[`PostprocessLayer`]: openff.nagl.nn.postprocess.PostprocessLayer
[`openff.nagl.nn.postprocess`]: openff.nagl.nn.postprocess
