import os
import pathlib
import pickle
from typing import Optional, Dict, Any, Tuple

import click
from click_option_group import optgroup

def train_model(
    config: Dict[str, Any] = {},
    output_directory: str = ".",
    checkpoint_directory: str = ".",
    checkpoint_dir: str = None,
    metrics: Dict[str, str] = {},
    model_config_files: Tuple[str, ...] = tuple(),
    runtime_kwargs: Dict[str, Any] = {},
):
    from openff.nagl.app.trainer import Trainer

    from ray.tune.integration.pytorch_lightning import (
        TuneReportCallback,
        TuneReportCheckpointCallback
    )

    trainer = Trainer.from_yaml_file(
        *model_config_files,
        output_directory=output_directory,
        **config,
        **runtime_kwargs
    )
    trainer_hash = trainer.to_simple_hash()

    print(f"Trainer hash: {trainer_hash}")
    
    log_config_file = os.path.join(
        output_directory,
        f"config-{trainer_hash}.yaml"
    )

    trainer.to_yaml_file(log_config_file)
    print(f"Wrote configuration values to {log_config_file}")

    checkpoint_directory_ = pathlib.Path(checkpoint_directory) / trainer_hash
    checkpoint_directory_.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint = str(checkpoint_directory_ / "checkpoint")
    if not os.path.exists(checkpoint):
        checkpoint = None

    tune_report = TuneReportCallback(metrics, on="validation_end")
    callbacks = [tune_report]
    if checkpoint_file:
        checkpointer = TuneReportCheckpointCallback(
            metrics=metrics,
            filename=checkpoint_file,
            on="validation_end",
        )
        callbacks.append(checkpointer)

    trainer.train(callbacks=callbacks, logger_name=trainer_hash, checkpoint_file=checkpoint)



@click.command()
@click.option(
    "--model-config-file",
    help="The path to a YAML configuration file for the model.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=tuple(),
    multiple=True,
)
@click.option(
    "--output-directory",
    help="The path to an output directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=".",
    show_default=True,
)
@click.option(
    "--output-config-file",
    help="Output best parameters",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--data-cache-directory",
    help="The path to a cached data path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    required=True,
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
    "--n-total-trials",
    help="Total number of trials",
    type=int,
    default=250,
    show_default=True,
)
@click.option(
    "--partial-charge-method",
    help="Method",
    type=str,
    default="am1",
    show_default=True,
)
@click.option(
    "--suffix",
    help="name suffix",
    type=str,
    default="",
    show_default=True,
)
def tune_model(
    data_cache_directory: str,
    n_epochs: int = 200,
    n_gpus: int = 1,
    n_total_trials: int = 300,
    output_directory: str = ".",
    output_config_file: str = "output.yaml",
    model_name: str = "graph",
    model_config_file: Tuple[str, ...] = tuple(),
    partial_charge_method: str = "am1",
    suffix: str = "",
):
    import yaml

    import ray
    from ray import air
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler

    metrics = {
        "loss": "val_loss",
    }

    config = {
        "n_convolution_layers": tune.choice([3, 4, 5]),
        "n_convolution_hidden_features": tune.choice([64, 128, 256]),
        "n_readout_layers": tune.choice([0, 1, 2]),
        "n_readout_hidden_features": tune.choice([64, 128, 256]),
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5]),
        "activation_function": tune.choice(["ReLU", "Sigmoid", "Tanh"]),
    }

    ray.init(num_cpus=1)

    scheduler = ASHAScheduler(
        max_t=n_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=list(metrics.keys()) + ["training_iteration"]
    )

    training_output_directory = pathlib.Path(output_directory) / model_name
    training_output_directory.mkdir(exist_ok=True, parents=True)
    training_checkpoint_directory = training_output_directory / "checkpoints"
    training_checkpoint_directory.mkdir(exist_ok=True, parents=True)

    model_config_file = [
        str(pathlib.Path(x).resolve())
        for x in model_config_file
    ]

    training_function = tune.with_parameters(
        train_model,
        output_directory=str(training_output_directory.resolve()),
        checkpoint_directory=str(training_checkpoint_directory.resolve()),
        metrics=metrics,
        runtime_kwargs=dict(
            n_epochs=n_epochs,
            n_gpus=n_gpus,
            partial_charge_method=partial_charge_method,
            readout_name=f"{partial_charge_method}-charges",
            data_cache_directory=os.path.abspath(data_cache_directory),
            use_cached_data=True,
        ),
        model_config_files=model_config_file
    )

    tune_config = tune.TuneConfig(
        metric="loss",
        mode="min",
        scheduler=scheduler,
        num_samples=n_total_trials,
    )
    
    run_config = air.RunConfig(
        name=f"tune_{model_name}_model_hyperparameters",
        progress_reporter=reporter,
        local_dir="./ray_results",
    )

    tuner = tune.Tuner(
        tune.with_resources(
            training_function,
            resources=dict(cpu=1, gpu=n_gpus)
        ),
        tune_config=tune_config,
        run_config=run_config,
        param_space=config
    )
    results = tuner.fit()
    results_file = f"{model_name}{suffix}-{partial_charge_method}_search-results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {results_file}")

    print("# results: ", len(results))


    best = results.get_best_result()
    print("Best hyperparameters: ", best.config)
    print("Best metrics: ", best.metrics)

    with open(output_config_file, "w") as f:
        yaml.dump(best.config, f)
    print(f"Wrote to {output_config_file}")

    for result in results:
        if result.error:
            print("Error in trial: ", result.error)
        else:
            print("Trial loss: ", result.metrics["loss"])


if __name__ == "__main__":
    tune_model()
