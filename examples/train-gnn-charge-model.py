from typing import Tuple, Union, Dict, Any
import yaml
import os
import pickle

import rich
from pprint import pprint
from rich import pretty
from rich.console import NewLine
import click
from click_option_group import optgroup

import torch
import pytorch_lightning as pl
from openff.toolkit.topology import Molecule as OFFMolecule

from gnn_charge_models.dgl.molecule import DGLMolecule
from gnn_charge_models.features import FeatureArgs

from gnn_charge_models.nn.modules.lightning import DGLMoleculeLightningModel, DGLMoleculeLightningDataModule
from gnn_charge_models.nn.modules.core import ConvolutionModule, ReadoutModule
from gnn_charge_models.nn.modules.pooling import PoolAtomFeatures
from gnn_charge_models.nn.sequential import SequentialLayers
from gnn_charge_models.nn.modules.postprocess import ComputePartialCharges

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from gnn_charge_models.utils.hash import hash_file, hash_dict

ExistingFile = click.Path(exists=True, file_okay=True, dir_okay=False)


def required_option(*args, **kwargs):
    if "default" in kwargs:
        kwargs["show_default"] = kwargs.get("show_default", True)
    kwargs["required"] = kwargs.get("required", True)
    return optgroup.option(*args, **kwargs)


class PartialChargeModelV1(DGLMoleculeLightningModel):
    def __init__(
        self,
        n_gcn_hidden_features: int,
        n_gcn_layers: int,
        n_am1_hidden_features: int,
        n_am1_layers: int,
        learning_rate: float,
        partial_charge_method: str,
        atom_features: Tuple[Union[str, Dict[str, Any]]] = tuple(),
        bond_features: Tuple[Union[str, Dict[str, Any]]] = tuple(),
    ):
        self.n_gcn_hidden_features = n_gcn_hidden_features
        self.n_gcn_layers = n_gcn_layers
        self.n_am1_hidden_features = n_am1_hidden_features
        self.n_am1_layers = n_am1_layers
        self.learning_rate = learning_rate
        self.partial_charge_method = partial_charge_method
        self.readout_name = f"{partial_charge_method}-charges"

        self.atom_features = [
            FeatureArgs.from_input(feature, feature_type="atoms")
            for feature in atom_features
        ]
        self.bond_features = [
            FeatureArgs.from_input(feature, feature_type="bonds")
            for feature in bond_features
        ]

        self.n_atom_features = sum(
            len(feature)
            for feature in self.instantiate_atom_features()
        )

        # build modules
        convolution = ConvolutionModule(
            architecture="SAGEConv",
            n_input_features=self.n_atom_features,
            hidden_feature_sizes=[n_gcn_hidden_features] * n_gcn_layers,
        )

        readout_activation = ["ReLU"] * n_am1_layers + ["Linear"]
        readout_hidden_features = [n_am1_hidden_features] * n_am1_layers + [2]

        readout = ReadoutModule(
            pooling_layer=PoolAtomFeatures(),
            readout_layers=SequentialLayers.with_layers(
                n_input_features=n_gcn_hidden_features,
                hidden_feature_sizes=readout_hidden_features,
                layer_activation_functions=readout_activation,
            ),
            postprocess_layer=ComputePartialCharges(),
        )

        super().__init__(
            convolution_module=convolution,
            readout_modules={self.readout_name: readout},
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

    def compute_charges(self, molecule: OFFMolecule) -> torch.Tensor:
        dglmol = DGLMolecule.from_openff(
            molecule,
            atom_features=self.instantiate_atom_features(),
            bond_features=self.instantiate_bond_features(),
        )
        return self.forward(dglmol)[self.readout_name]

    def instantiate_atom_features(self):
        return [feature() for feature in self.atom_features]

    def instantiate_bond_features(self):
        return [feature() for feature in self.bond_features]


@click.group()
def cli():
    pass


@cli.command("train")
@optgroup.group("Data")
@required_option("--training-set-paths", type=ExistingFile, multiple=True)
@required_option("--training-batch-size", type=click.INT, default=512)
@required_option("--validation-set-paths", type=ExistingFile, multiple=True)
@required_option("--test-set-paths", type=ExistingFile, multiple=True)
@optgroup.group("Model")
@required_option("--partial-charge-method", type=click.STRING)
@required_option("--model-features-path", type=ExistingFile)
@required_option("--n-gcn-layers", type=click.INT, default=4)
@required_option("--n-gcn-hidden-features", type=click.INT, default=64)
@required_option("--n-am1-layers", type=click.INT, default=4)
@required_option("--n-am1-hidden-features", type=click.INT, default=64)
@optgroup.group("Optimizer")
@required_option("--n-epochs", type=click.INT, default=500)
@required_option("--learning-rate", type=click.FLOAT, default=1.0e-4)
@optgroup.group("Other")
@required_option("--seed", type=click.INT, required=False)
@required_option(
    "--output-directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="lightning-logs",
    required=False,
)
def train(
    training_set_paths,
    training_batch_size,
    validation_set_paths,
    test_set_paths,
    partial_charge_method,
    model_features_path,
    n_gcn_layers,
    n_gcn_hidden_features,
    n_am1_layers,
    n_am1_hidden_features,
    n_epochs,
    learning_rate,
    seed,
    output_directory
):
    cli_inputs = locals()

    os.makedirs(output_directory, exist_ok=True)

    console = rich.get_console()
    pretty.install(console)

    console.rule("CLI inputs")
    console.print(NewLine())
    pprint(cli_inputs)
    console.print(NewLine())

    if seed is not None:
        pl.seed_everything(seed)

    with open(model_features_path, "r") as f:
        contents = yaml.safe_load(f)
    atom_features = contents.get("atom_features", [])
    bond_features = contents.get("bond_features", [])

    model = PartialChargeModelV1(
        n_gcn_hidden_features=n_gcn_hidden_features,
        n_gcn_layers=n_gcn_layers,
        n_am1_hidden_features=n_am1_hidden_features,
        n_am1_layers=n_am1_layers,
        learning_rate=learning_rate,
        partial_charge_method=partial_charge_method,
        atom_features=atom_features,
        bond_features=bond_features,
    )

    with console.status("hashing inputs"):
        atom_features_ = [x.to_dict() for x in sorted(model.atom_features)]
        bond_features_ = [x.to_dict() for x in sorted(model.bond_features)]
        hashable_dict = dict(
            atom_feature_types=atom_features_,
            bond_feature_types=bond_features_,
            partial_charge_method=partial_charge_method,
            training_set_hash=[hash_file(f)
                               for f in sorted(training_set_paths)],
            validation_set_hash=[hash_file(f)
                                 for f in sorted(validation_set_paths)],
            test_set_hash=[hash_file(f) for f in sorted(test_set_paths)],
            training_batch_size=training_batch_size,
            validation_batch_size=None,
            test_batch_size=None,
        )
        cache_hash = hash_dict(hashable_dict)

    data_module = DGLMoleculeLightningDataModule(
        model.instantiate_atom_features(),
        model.instantiate_bond_features(),
        partial_charge_method=partial_charge_method,
        bond_order_method=None,
        training_set_paths=training_set_paths,
        training_batch_size=training_batch_size,
        validation_set_paths=validation_set_paths,
        validation_batch_size=None,
        test_set_paths=test_set_paths,
        test_batch_size=None,
        use_cached_data=True,
        output_path=f"nagl-data-module-{cache_hash}.pkl",
    )

    console.print(NewLine())
    console.rule("model")
    console.print(NewLine())
    console.print(model.hparams)
    console.print(NewLine())
    console.print(model)
    console.print(NewLine())

    console.print(NewLine())
    console.rule("training")
    console.print(NewLine())

    # Train the model
    n_gpus = 0 if not torch.cuda.is_available() else 1
    print(f"Using {n_gpus} GPUs")

    version_string = (
        f"tb-{training_batch_size}_"
        f"gcn-{n_gcn_layers}_"
        f"gcnfeat-{n_gcn_hidden_features}_"
        f"am1-{n_am1_layers}_"
        f"am1feat-{n_am1_hidden_features}_"
        f"lr-{learning_rate}"
    )

    os.makedirs(output_directory, exist_ok=True)
    logger = TensorBoardLogger(
        output_directory, name="default",
        version=version_string,
    )

    trainer = pl.Trainer(
        gpus=n_gpus,
        min_epochs=n_epochs,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(save_top_k=3, monitor="val_loss"),
            TQDMProgressBar(),
        ],
    )

    trainer.fit(model, datamodule=data_module)

    metrics_file = os.path.join(
        output_directory, "default", version_string, "metrics.pkl")

    with open(metrics_file, "wb") as f:
        pickle.dump((trainer.callback_metrics, trainer.logged_metrics), f)

    if test_set_paths is not None:
        trainer.test(model, data_module)
