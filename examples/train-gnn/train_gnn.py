import os
import pathlib
import pickle
from typing import Tuple

import click


@click.command()
@click.option(
    "--model-config-file",
    help="The path to a YAML configuration file for the model.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=tuple(),
    multiple=True,
)
@click.option(
    "--data-cache-directory",
    help="Path to cached data",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--output-directory",
    help="The path to an output directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=".",
    show_default=True,
)
@click.option(
    "--n-epochs",
    help="Number of epochs",
    type=int,
    default=200,
    show_default=True,
)
@click.option(
    "--n-gpus",
    help="Number of gpus",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "--partial-charge-method",
    help="Method",
    type=str,
    default="am1",
    show_default=True,
)
def train_model(
    data_cache_directory: str,
    model_config_file: Tuple[str, ...] = tuple(),
    output_directory: str = ".",
    partial_charge_method: str = "am1",
    n_gpus: int = 1,
    n_epochs: int = 200,
):
    from pytorch_lightning.callbacks import ModelCheckpoint

    from openff.nagl._app.trainer import Trainer

    trainer = Trainer.from_yaml_file(
        *model_config_file,
        output_directory=output_directory,
        partial_charge_method=partial_charge_method,
        readout_name=f"{partial_charge_method}-charges",
        use_cached_data=True,
        data_cache_directory=os.path.abspath(data_cache_directory),
        n_gpus=n_gpus,
        n_epochs=n_epochs,
    )
    trainer_hash = trainer.to_simple_hash()

    print(f"Trainer hash: {trainer_hash}")

    log_config_file = os.path.join(output_directory, "config.yaml")

    trainer.to_yaml_file(log_config_file)
    print(f"Wrote configuration values to {log_config_file}")

    checkpoint_directory_ = (
        pathlib.Path(output_directory) / "checkpoints" / trainer_hash
    )
    checkpoint_directory_.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint = str(checkpoint_directory_ / "checkpoint")  # noqa
    if not os.path.exists(checkpoint):
        checkpoint = None

    output_ = pathlib.Path(output_directory) / trainer_hash
    output_.mkdir(parents=True, exist_ok=True)

    callbacks = [ModelCheckpoint(save_top_k=3, monitor="val_loss")]

    trainer.train(
        logger_name=trainer_hash, checkpoint_file=checkpoint, callbacks=callbacks
    )
    print("--- Best model ---")
    print(callbacks[0].best_model_path)
    print(callbacks[0].best_model_score)
    metrics_file = pathlib.Path(output_directory) / trainer_hash / "metrics.pkl"
    with open(str(metrics_file), "wb") as f:
        metrics = (trainer._trainer.callback_metrics, trainer._trainer.logged_metrics)
        pickle.dump(metrics, f)

    print(f"Wrote metrics to {str(metrics_file)}")


if __name__ == "__main__":
    train_model()
