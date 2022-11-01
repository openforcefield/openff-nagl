from typing import TYPE_CHECKING, Tuple

from openff.nagl.nn.modules.lightning import DGLMoleculeLightningModel

if TYPE_CHECKING:
    import torch
    from openff.toolkit.topology import Molecule as OFFMolecule

    from openff.nagl.features import AtomFeature, BondFeature


class GNNModel(DGLMoleculeLightningModel):
    @classmethod
    def from_yaml_file(cls, *paths, **kwargs):
        import yaml

        yaml_kwargs = {}
        for path in paths:
            with open(str(path), "r") as f:
                dct = yaml.load(f, Loader=yaml.FullLoader)
                dct = {k.replace("-", "_"): v for k, v in dct.items()}
                yaml_kwargs.update(dct)
        yaml_kwargs.update(kwargs)
        return cls(**yaml_kwargs)

    @property
    def n_atom_features(self):
        return sum(len(feature) for feature in self.atom_features)

    def __init__(
        self,
        convolution_architecture: str,
        n_convolution_hidden_features: int,
        n_convolution_layers: int,
        n_readout_hidden_features: int,
        n_readout_layers: int,
        activation_function: str,
        postprocess_layer: str,
        readout_name: str,
        learning_rate: float,
        atom_features: Tuple["AtomFeature", ...],
        bond_features: Tuple["BondFeature", ...],
    ):
        from openff.nagl.features import AtomFeature, BondFeature
        from openff.nagl.nn.activation import ActivationFunction
        from openff.nagl.nn.gcn import GCNStackMeta
        from openff.nagl.nn.modules.core import ConvolutionModule, ReadoutModule
        from openff.nagl.nn.modules.pooling import PoolAtomFeatures
        from openff.nagl.nn.modules.postprocess import PostprocessLayerMeta
        from openff.nagl.nn.sequential import SequentialLayers

        self.readout_name = readout_name

        convolution_architecture = GCNStackMeta._get_class(convolution_architecture)
        postprocess_layer = PostprocessLayerMeta._get_class(postprocess_layer)
        activation_function = ActivationFunction._get_class(activation_function)
        self.atom_features = self._validate_features(atom_features, AtomFeature)
        self.bond_features = self._validate_features(bond_features, BondFeature)

        hidden_conv = [n_convolution_hidden_features] * n_convolution_layers
        convolution_module = ConvolutionModule(
            architecture=convolution_architecture,
            n_input_features=self.n_atom_features,
            hidden_feature_sizes=hidden_conv,
        )

        hidden_readout = [n_readout_hidden_features] * n_readout_layers
        hidden_readout.append(postprocess_layer.n_features)
        readout_activation = [activation_function] * n_readout_layers
        readout_activation.append(ActivationFunction.Identity)
        readout_module = ReadoutModule(
            pooling_layer=PoolAtomFeatures(),
            readout_layers=SequentialLayers.with_layers(
                n_input_features=n_convolution_hidden_features,
                hidden_feature_sizes=hidden_readout,
                layer_activation_functions=readout_activation,
            ),
            postprocess_layer=postprocess_layer(),
        )

        super().__init__(
            convolution_module=convolution_module,
            readout_modules={self.readout_name: readout_module},
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

    def compute_property(self, molecule: "OFFMolecule") -> "torch.Tensor":
        from openff.nagl.dgl.molecule import DGLMolecule

        dglmol = DGLMolecule.from_openff(
            molecule,
            atom_features=self.atom_features,
            bond_features=self.bond_features,
        )
        return self.forward(dglmol)[self.readout_name]

    @staticmethod
    def _validate_features(features, feature_class):
        if isinstance(features, dict):
            features = list(features.items())
        all_v = []
        for item in features:
            if isinstance(item, dict):
                all_v.extend(list(item.items()))
            elif isinstance(item, (str, feature_class, type(feature_class))):
                all_v.append((item, {}))
            else:
                all_v.append(item)

        instantiated = []
        for klass, args in all_v:
            if isinstance(klass, feature_class):
                instantiated.append(klass)
            else:
                klass = type(feature_class)._get_class(klass)
                if not isinstance(args, dict):
                    item = klass._with_args(args)
                else:
                    item = klass(**args)
                instantiated.append(item)
        return instantiated
