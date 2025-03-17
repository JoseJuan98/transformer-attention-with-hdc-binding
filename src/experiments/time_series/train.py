# -*- coding: utf-8 -*-
"""Training script for the Transformer model."""
# Standard imports
import multiprocessing

# Third party imports
import lightning
import torch
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

# First party imports
from experiments import ExperimentConfig
from experiments.time_series.dataset import get_ucr_datasets
from models import EncoderOnlyTransformerTSClassifier, SinusoidalPositionalEncoding
from utils import Config, get_logger, msg_task, plot_csv_logger_metrics


def train():
    """Trains the model."""

    task = "time_series_classification"
    dataset_name = "ArticularyWordRecognition"
    model_name = "transformer_encoder_only"
    run_path = Config.model_dir / "runs" / task / dataset_name

    logger = get_logger(
        name="main", log_filename=f"{task}/{dataset_name.lower().replace(' ', '_')}/{model_name.lower()}.log"
    )

    if not torch.cuda.is_available():
        raise Exception("CUDA not available. No GPU found.")

    msg_task(msg="Loading data", logger=logger)
    train_dataset, test_dataset, max_len, num_classes, num_channels = get_ucr_datasets(
        dsid=dataset_name,
        extract_path=Config.data_dir / task,
        logger=logger,
        plot_first_row=True,
        plot_path=Config.plot_dir / task / f"{dataset_name}_sample.png",
    )

    # Configuration
    experiment_cfg = ExperimentConfig(
        num_epochs=100,
        input_size=1000,
        context_lenght=max_len,
        d_model=num_channels,
        num_heads=8,
        d_ff=256,
        num_layers=3,
        dropout=0.1,
        batch_size=32,
        learning_rate=1e-3,
        device="cuda",
        model_path=f"{task}/{dataset_name.lower().replace(' ', '_')}/{model_name.lower()}.pth",
        description=f"{model_name.replace('_', ' ').title()} for {task.replace('_', ' ').title()} for {dataset_name.replace('_', ' ').title()} dataset",
        num_classes=num_classes,
        dataset=dataset_name,
        experiment_name=model_name.replace("_", " ").title(),
        precision="16-mixed",
    )

    logger.info(f"Saving experiment configuration to {run_path}")
    experiment_cfg.dump(path=run_path / "experiment_config.json")

    # Create data loaders
    # TODO: balance the number of workers depending on the dataset size and pin memory
    cpu_count = multiprocessing.cpu_count()
    num_workers = cpu_count - 2 if cpu_count > 4 else 1

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=experiment_cfg.batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=experiment_cfg.batch_size, shuffle=False, num_workers=num_workers
    )

    # Create the model
    model = EncoderOnlyTransformerTSClassifier(
        num_layers=experiment_cfg.num_layers,
        d_model=experiment_cfg.d_model,
        num_heads=experiment_cfg.num_heads,
        d_ff=experiment_cfg.d_ff,
        embed_dim=experiment_cfg.context_lenght,
        batch_size=experiment_cfg.batch_size,
        positional_encoding=SinusoidalPositionalEncoding(
            d_model=experiment_cfg.d_model, max_len=experiment_cfg.context_lenght
        ),
        num_classes=experiment_cfg.num_classes,
        dropout=experiment_cfg.dropout,
        learning_rate=experiment_cfg.learning_rate,
    )

    msg_task(msg=f"Experiment {model_name.replace("_", " ").title()}", logger=logger)

    logger.info(f"Experiment Configuration:\n\n{experiment_cfg.pretty_str()}\n\n")

    logger.info("Trainer Configuration:\n")

    # Create a trainer
    trainer = lightning.Trainer(
        default_root_dir=Config.root_dir,
        max_epochs=experiment_cfg.num_epochs,
        accelerator="auto",
        devices="auto",
        precision=experiment_cfg.precision,
        logger=[
            CSVLogger(save_dir=Config.log_dir / task / dataset_name, name=f"metrics_{model_name}"),
            TensorBoardLogger(save_dir=run_path, name=f"board_{model_name}_logs"),
            logger,
        ],
        log_every_n_steps=1,
        callbacks=[ModelSummary(max_depth=-1)],
        # measures all the key methods across Callbacks, DataModules and the LightningModule in the training loop.
        profiler="simple",
        # If True, runs 1 batch of train, test and val to find any bugs. Also, it can be specified the number of
        # batches to run as an integer
        fast_dev_run=True,
    )

    # Train the model
    if experiment_cfg.num_epochs > 0:
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    # Test the model
    trainer.test(model, dataloaders=test_dataloader)

    # Save the model
    model_path = Config.model_dir / experiment_cfg.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    msg_task(f"Model saved to {model_path}", logger=logger)

    # plot metrics if there was training
    if experiment_cfg.num_epochs > 0:
        plot_csv_logger_metrics(
            csv_dir=trainer.logger.log_dir,
            experiment=model_name,
            logger=logger,
            plots_path=Config.plot_dir / task / dataset_name,
        )


if __name__ == "__main__":
    train()
