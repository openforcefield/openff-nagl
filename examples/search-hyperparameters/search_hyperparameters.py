import os
import pathlib
import math
import pickle
from typing import Dict, Any, Tuple

import click

def train_model(
    config: Dict[str, Any] = {},
    output_directory: str = ".",
    checkpoint_dir: str = None,
    metrics: Dict[str, str] = {},
    model_config_files: Tuple[str, ...] = tuple(),
    runtime_kwargs: Dict[str, Any] = {},
):
    from openff.nagl.app.trainer import Trainer
    from ray.tune.integration.pytorch_lightning import TuneReportCallback

    tune_report = TuneReportCallback(metrics, on="validation_end")
    callbacks = [tune_report]

    trainer = Trainer.from_yaml_file(
        *model_config_files,
        output_directory=output_directory,
        **config,
        **runtime_kwargs,
    )

    trainer_hash = trainer.to_simple_hash()
    print(f"Trainer hash: {trainer_hash}")
    
    log_config_file = os.path.join(
        output_directory,
        f"config-{trainer_hash}.yaml"
    )

    trainer.to_yaml_file(log_config_file)
    print(f"Wrote configuration values to {log_config_file}")

    trainer.train(callbacks=callbacks, logger_name=trainer_hash)



@click.command()
@click.option(
    "--model-config-file",
    "model_config_files",
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
    "--n-convolution-layers",
    type=str,
    default="4 5 6",
    show_default=True,
)
@click.option(
    "--n-convolution-hidden-features",
    type=str,
    default="128 256 512",
    show_default=True,
)
@click.option(
    "--n-readout-layers",
    type=str,
    default="1 2",
    show_default=True,
)
@click.option(
    "--n-readout-hidden-features",
    type=str,
    default="64 128 256",
    show_default=True,
)
@click.option(
    "--learning-rate",
    type=str,
    default="5e-2 1e-3 5e-3",
    show_default=True,
)
@click.option(
    "--activation-function",
    type=str,
    default="ReLU Sigmoid",
    show_default=True,
)
@click.option(
    "--convolution-architecture",
    type=str,
    default="SAGEConv GINConv",
    show_default=True,
)
@click.option(
    "--postprocess-layer",
    type=str,
    default="compute_partial_charges",
    show_default=True,
)
def tune_model(
    data_cache_directory: str,
    n_convolution_layers: str,
    n_convolution_hidden_features: str,
    n_readout_layers: str,
    n_readout_hidden_features: str,
    learning_rate: str,
    activation_function: str,
    convolution_architecture: str,
    n_epochs: int = 200,
    n_gpus: int = 1,
    n_total_trials: int = 300,
    output_directory: str = ".",
    output_config_file: str = "output.yaml",
    model_name: str = "graph",
    model_config_files: Tuple[str, ...] = tuple(),
    partial_charge_method: str = "am1",
    postprocess_layer: str = "compute_partial_charges"
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

    CONFIG_TYPES = {
        "n_convolution_layers": (n_convolution_layers, int),
        "n_convolution_hidden_features": (n_convolution_hidden_features, int),
        "n_readout_layers": (n_readout_layers, int),
        "n_readout_hidden_features": (n_readout_hidden_features, int),
        "learning_rate": (learning_rate, float),
        "activation_function": (activation_function, str),
        "convolution_architecture": (convolution_architecture, str)
    }

    config = {
        k: tune.choice(list(map(type_, input_.split())))
        for k, (input_, type_)
        in CONFIG_TYPES.items()
    }

    print("--- Evaluating hyperparameters ---")
    print(config)

    n = math.prod([len(v[0].split()) for v in CONFIG_TYPES.values()])
    print(f"Total potential combinations: {n_total_trials}/{n}" )


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

    output_directory = pathlib.Path(output_directory)
    training_output_directory = output_directory / model_name
    training_output_directory.mkdir(exist_ok=True, parents=True)

    model_config_files = [
        str(pathlib.Path(x).resolve())
        for x in model_config_files
    ]

    training_function = tune.with_parameters(
        train_model,
        output_directory=str(training_output_directory.resolve()),
        metrics=metrics,
        runtime_kwargs=dict(
            n_epochs=n_epochs,
            n_gpus=n_gpus,
            partial_charge_method=partial_charge_method,
            readout_name=f"{partial_charge_method}-charges",
            data_cache_directory=os.path.abspath(data_cache_directory),
            use_cached_data=True,
            postprocess_layer=postprocess_layer,
        ),
        model_config_files=model_config_files
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
    results_file = str(training_output_directory / "search-results.pkl")
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
