# import dataclasses
# import os
# from typing import Any, Dict, Optional, Tuple, Union

# import pytorch_lightning as pl
# import rich
# import torch
# from openff.toolkit.topology.molecule import Molecule as OFFMolecule
# from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
# from pytorch_lightning.loggers import TensorBoardLogger
# from rich import pretty
# from rich.console import NewLine

# from openff.nagl.dgl.molecule import DGLMolecule
# from openff.nagl.features import FeatureArgs
# from openff.nagl.nn.activation import ActivationFunction
# from openff.nagl.nn.gcn import GCNStackMeta
# from openff.nagl.nn.modules.core import ConvolutionModule, ReadoutModule
# from openff.nagl.nn.modules.lightning import (
#     DGLMoleculeLightningDataModule,
#     DGLMoleculeLightningModel,
# )
# from openff.nagl.nn.modules.pooling import PoolAtomFeatures
# from openff.nagl.nn.modules.postprocess import PostprocessLayerMeta
# from openff.nagl.nn.sequential import SequentialLayers
# from openff.nagl.storage.record import ChargeMethod, WibergBondOrderMethod
# from openff.nagl.utils.types import Pathlike


# @dataclasses.dataclass
# class Trainer:
#     convolution_architecture: Union[str, GCNStackMeta]
#     n_convolution_hidden_features: int
#     n_convolution_layers: int
#     readout_name: str
#     n_readout_hidden_features: int
#     n_readout_layers: int
#     learning_rate: float
#     activation_function: Union[str, ActivationFunction]
#     postprocess_layer: Union[str, PostprocessLayerMeta]
#     atom_features: Tuple[Union[str, Dict[str, Any]]] = tuple()
#     bond_features: Tuple[Union[str, Dict[str, Any]]] = tuple()
#     partial_charge_method: Optional[Union[str, ChargeMethod]] = None
#     bond_order_method: Optional[Union[str, WibergBondOrderMethod]] = None
#     training_set_paths: Union[Pathlike, Tuple[Pathlike]] = tuple()
#     validation_set_paths: Union[Pathlike, Tuple[Pathlike]] = tuple()
#     test_set_paths: Union[Pathlike, Tuple[Pathlike]] = tuple()
#     output_directory: Pathlike = "."
#     output_data_path: Pathlike = "nagl-data-module.pkl"
#     training_batch_size: Optional[int] = None
#     validation_batch_size: Optional[int] = None
#     test_batch_size: Optional[int] = None
#     use_cached_data: bool = False
#     n_gpus: int = 0
#     n_epochs: int = 100
#     seed: Optional[int] = None

#     def __post_init__(self):
#         self._postprocess_inputs()
#         self.model = self._set_up_model()
#         self.data_module = self._set_up_data_module()

#     def _postprocess_inputs(self):
#         self.convolution_architecture = GCNStackMeta.get_gcn_class(
#             self.convolution_architecture
#         )
#         self.postprocess_layer = PostprocessLayerMeta.get_layer_class(
#             self.postprocess_layer
#         )
#         self.activation_function = ActivationFunction.get(
#             self.activation_function)
#         self.atom_features = [
#             FeatureArgs.from_input(feature, feature_type="atoms")
#             for feature in self.atom_features
#         ]
#         self.bond_features = [
#             FeatureArgs.from_input(feature, feature_type="bonds")
#             for feature in self.bond_features
#         ]
#         self.n_atom_features = sum(
#             len(feature) for feature in self.instantiate_atom_features()
#         )

#     def _set_up_data_module(self):
#         return DGLMoleculeLightningDataModule(
#             atom_features=self.instantiate_atom_features(),
#             bond_features=self.instantiate_bond_features(),
#             partial_charge_method=self.partial_charge_method,
#             bond_order_method=self.bond_order_method,
#             training_set_paths=self.training_set_paths,
#             training_batch_size=self.training_batch_size,
#             validation_set_paths=self.validation_set_paths,
#             validation_batch_size=self.validation_batch_size,
#             test_set_paths=self.test_set_paths,
#             test_batch_size=self.test_batch_size,
#             use_cached_data=self.use_cached_data,
#             output_path=self.output_data_path,
#         )

#     def _set_up_model(self):
#         convolution_module = self._set_up_convolution_module()
#         readout_module = self._set_up_readout_module()
#         model = DGLMoleculeLightningModel(
#             convolution_module=convolution_module,
#             readout_modules={self.readout_name: readout_module},
#             learning_rate=self.learning_rate,
#         )
#         return model

#     def _set_up_convolution_module(self):
#         hd = [self.n_convolution_hidden_features] * self.n_convolution_layers
#         return ConvolutionModule(
#             architecture=self.convolution_architecture,
#             n_input_features=self.n_atom_features,
#             hidden_feature_sizes=hd,
#         )

#     def _set_up_readout_module(self):
#         hd = [self.n_readout_hidden_features] * self.n_readout_layers
#         hd.append(2)
#         readout_activation = [self.activation_function] * self.n_readout_layers
#         readout_activation.append(ActivationFunction.Identity)
#         return ReadoutModule(
#             pooling_layer=PoolAtomFeatures(),
#             readout_layers=SequentialLayers.with_layers(
#                 n_input_features=self.n_convolution_hidden_features,
#                 hidden_feature_sizes=hd,
#                 layer_activation_functions=readout_activation,
#             ),
#             postprocess_layer=self.postprocess_layer(),
#         )

#     def compute_property(self, molecule: OFFMolecule) -> torch.Tensor:
#         dglmol = DGLMolecule.from_openff(
#             molecule,
#             atom_features=self.instantiate_atom_features(),
#             bond_features=self.instantiate_bond_features(),
#         )
#         return self.model.forward(dglmol)[self.readout_name]

#     def instantiate_atom_features(self):
#         return [feature() for feature in self.atom_features]

#     def instantiate_bond_features(self):
#         return [feature() for feature in self.bond_features]

#     def train(self):
#         os.makedirs(str(self.output_directory), exist_ok=True)

#         console = rich.get_console()
#         pretty.install(console)

#         console.print(NewLine())
#         console.rule("model")
#         console.print(NewLine())
#         console.print(self.model.hparams)
#         console.print(NewLine())
#         console.print(self.model)
#         console.print(NewLine())
#         console.print(NewLine())
#         console.rule("training")
#         console.print(NewLine())
#         console.print(f"Using {self.n_gpus} GPUs")

#         if self.seed is not None:
#             pl.seed_everything(self.seed)

#         logger = TensorBoardLogger(
#             self.output_directory,
#             name="default",
#         )

#         self.trainer = pl.Trainer(
#             gpus=self.n_gpus,
#             min_epochs=self.n_epochs,
#             max_epochs=self.n_epochs,
#             logger=logger,
#             callbacks=[
#                 ModelCheckpoint(save_top_k=3, monitor="val_loss"),
#                 TQDMProgressBar(),
#             ],
#         )

#         self.trainer.fit(self.model, datamodule=self.data_module)

#         if self.test_set_paths is not None:
#             self.trainer.test(self.model, self.data_module)
