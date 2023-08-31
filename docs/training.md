# Training a NAGL model

## Preparing a dataset

Preparing a dataset for training is just as important as designing the model itself. The dataset must, at a minimum:

1. Cover the entire space of molecules that you want to predict the properties of
2. Contain enough examples that the model can learn the desired function
3. Treat the property you want to model consistently

Once you have your overall dataset, you should divide it into **training**, **validation** and **test** datasets to evaluate the quality of fit of your model as it trains. It's important that all three buckets share coverage of the entire space:

- **training**: About 80% of your data. Model parameters are fit directly to these data.
- **validation**: About 10% of your data. Used to validate the model as it is iteratively fitted.
- **test**: About 10% of your data. Used to test the final model against data it has never "seen" before.

Datasets are constructed by producing OpenFF Toolkit [`Molecule`] objects with the appropriate charges, loading them into [`MoleculeRecord`] objects, and storing them in a SQLite database:

```python
from openff.nagl.storage.record import MoleculeRecord
from openff.nagl.storage import MoleculeStore
from openff.toolkit import Molecule

records = []
for smiles in [
    "C",
    "CC",
    "CCC",
    ...
]:
    # Create the Molecule
    molecule = Molecule.from_smiles(smiles)

    # Compute example partial charges
    molecule.generate_conformers(n_conformers=1)
    molecule.assign_partial_charges("am1bcc")

    # Record results
    records.append(
        MoleculeRecord.from_precomputed_openff(
            molecule,
            partial_charge_method="am1bcc"
        )
    )

# Save the dataset
MoleculeStore("dataset.sqlite").store(records)
```

[`Molecule`]: openff.toolkit.topology.Molecule
[`MoleculeRecord`]: openff.nagl.storage.record.MoleculeRecord

## Loading a dataset

Datasets are loaded for training with the [`DGLMoleculeLightningDataModule`] class. Objects of this class require the [featurization schema] used in NAGL models, as well as paths to the datasets and batch sizes for loading the data.

```python
from openff.nagl.nn.dataset import DGLMoleculeLightningDataModule

data_module = DGLMoleculeLightningDataModule(
    atom_features=atom_features,
    bond_features=bond_features,
    partial_charge_method="am1bcc",
    training_set_paths=["training_data.sqlite"],
    validation_set_paths=["validation_data.sqlite"],
    test_set_paths=["test_data.sqlite"],
    training_batch_size=1000,
    validation_batch_size=1000,
    test_batch_size=1000,
)
```

[`DGLMoleculeLightningDataModule`]: openff.nagl.nn.dataset.DGLMoleculeLightningDataModule
[featurization schema]: model_features

## Training

The PyTorch Lightning [`Trainer`] class greatly simplifies training a model. It's a simple procedure to construct the `Trainer`, fit the model to the data, test it, and save the fitted model:

```python
from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=200)

trainer.checkpoint_callback.monitor = "val_loss"

trainer.fit(
    model,
    datamodule=data_module,
)

trainer.test(model, data_module)

model.save("model.pt")
```

[`Trainer`]: pytorch_lightning:common/trainer
