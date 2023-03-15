# Designing a GCN

Designing a GCN with NAGL primarily involves creating an instance of the [`GNNModel`] class. This can be done straightforwardly in Python:

```python
from openff.nagl.features import atoms, bonds
from openff.nagl import GNNModel
from openff.nagl.nn.gcn import SAGEConvStack
from torch.nn import ReLU
from openff.nagl.nn.postprocess import ComputePartialCharges

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
    convolution_architecture=SAGEConvStack,
    n_convolution_hidden_features=128,
    n_convolution_layers=3,
    n_readout_hidden_features=128,
    n_readout_layers=4,
    activation_function=ReLU,
    postprocess_layer=ComputePartialCharges,
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

Here we'll go through each option, what it means, where to find the available choices, and some hints on how to write your own.

[`GNNModel`]: openff.nagl.GNNModel
[`GNNModel.from_yaml_file()`]: openff.nagl.GNNModel.from_yaml_file 

## `atom_features` and `bond_features`

These arguments specify the featurization scheme for the model. `atom_features` takes a tuple of features from the [`openff.nagl.features.atoms`] module, and `bond_features` a tuple of features from the [`openff.nagl.features.bonds`] module. These features inherit from the [`AtomFeature`] and [`BondFeature`] abstract base classes, respectively.

[`openff.nagl.features.atoms`]: openff.nagl.features.atoms
[`openff.nagl.features.bonds`]: openff.nagl.features.bonds
[`AtomFeature`]: openff.nagl.features.atoms.AtomFeature
[`BondFeature`]: openff.nagl.features.bonds.BondFeature

## `convolution_architecture`

The `convolution_architecture` argument specifies the structure of the convolution module. Available options are provided in the [`openff.nagl.nn.gcn`] module. 

[`openff.nagl.nn.gcn`]: openff.nagl.nn.gcn


    n_convolution_hidden_features=128,
    n_convolution_layers=3,
    n_readout_hidden_features=128,
    n_readout_layers=4,
    activation_function=ReLU,
    postprocess_layer=ComputePartialCharges,
    readout_name=f"am1bcc-charges",
    learning_rate=0.001,