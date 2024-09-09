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

Datasets can be constructed very flexibly by creating PyArrow-parseable tables.

```python
import pyarrow as pa
import pyarrow.parquet as pq
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
    record = {
        "mapped_smiles": molecule.to_smiles(mapped=True),
        "charges": molecule.partial_charges.m.tolist()
    }

# Save the dataset
table = pa.Table.from_pylist(records)
pq.write_table(table, "training-data.parquet")
```

[`Molecule`]: openff.toolkit.topology.Molecule

## Loading a dataset

Datasets are loaded for training with the [`DataConfig`] class. Objects of this class require training targets and loss functions to be defined,
as well as paths to the datasets and batch sizes for loading the data.

```python
from openff.nagl.config import DataConfig, DatasetConfig
from openff.nagl.training import ReadoutTarget

direct_charges_target = ReadoutTarget(
    # what we're using from the parquet table to evaluate loss
    target_label="charges",
    # the output of the GNN we use to evaluate loss
    prediction_label="charges",
    # how we want to evaluate loss, e.g. RMSE, MSE, ...
    metric="rmse",
    # how much to weight this target
    # helps with scaling in multi-target optimizations
    weight=1,
    denominator=1,
)

training_config = DatasetConfig(
    sources=["training-data.parquet"],
    targets=[direct_charges_target],
    batch_size=1000,
)

data_config = DataConfig(
    training=training_config,
    validation=...,
    test=...,
)
```



## Training

Combine the [`ModelConfig`] (see -- [](designing_a_model)), [`DataConfig`] and an [`OptimizerConfig`] class into a [`TrainingConfig`].

An [`OptimizerConfig`] is reasonably simple to define:

```python
from openff.nagl.config import OptimizerConfig

optimizer_config = OptimizerConfig(optimizer="Adam", learning_rate=0.001)
```

We can then combine these into a [`TrainingConfig`] and create a [`TrainingGNNModel`]

```
from openff.nagl.config import TrainingConfig
from openff.nagl.training.training import TrainingGNNModel

training_config = TrainingConfig(
    model=model_config,
    data=data_config,
    optimizer=optimizer_config
)

training_model = TrainingGNNModel(training_config)
```


Then use the PyTorch Lightning [`Trainer`] class to greatly simplify training a model. It's a simple procedure to construct the `Trainer`, fit the model to the data, test it, and save the fitted model:



```python
from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=200)
trainer.checkpoint_callback.monitor = "val_loss"

data_module = training_model.create_data_module()

trainer.fit(
    training_model,
    datamodule=data_module,
)

trainer.test(model, data_module)

model.save("model.pt")
```

[`Trainer`]: inv:pytorch_lightning#common/trainer