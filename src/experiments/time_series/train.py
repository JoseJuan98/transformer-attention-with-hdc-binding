# -*- coding: utf-8 -*-
"""Training script for the Transformer model."""

# Third party imports
import lightning
import torch
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
from tsai.data.core import TSDatasets
from tsai.data.external import get_UCR_data
from tsai.data.preprocessing import TSStandardize

# First party imports
from experiments import ExperimentConfig
from models import EncoderOnlyTransformerTSClassifier, SinusoidalPositionalEncoding
from utils import Config, get_logger, msg_task, plot_csv_logger_metrics


def train():
    """Trains the model."""
    # Load data
    task = "time_series_classification"
    dataset_name = "ArticularyWordRecognition"
    model_name = "transformer_encoder_only"
    X, y, X_test, y_test = get_UCR_data(dataset_name, split_data=True, return_type="tsai")

    # Create directories for the experiment
    Config.create_dirs()

    # Preprocess data
    X, y = TSDatasets(X, y)
    X_test, y_test = TSDatasets(X_test, y_test)

    # Standardize data
    X = TSStandardize().fit_transform(X)
    X_test = TSStandardize().fit_transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X).float()
    y_train_tensor = torch.from_numpy(y).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    if not torch.cuda.is_available():
        raise Exception("CUDA not available. No GPU found.")

    # Configuration
    experiment_cfg = ExperimentConfig(
        input_size=1000,  # Not used, placeholder
        max_len=X_train_tensor.shape[1],  # Sequence length
        d_model=128,
        num_heads=8,
        d_ff=256,
        num_layers=3,
        dropout=0.1,
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=100,  # Reduced for demonstration
        device="cuda",
        model_path=f"{task}/{dataset_name.lower().replace(' ', '_')}/{model_name.lower()}.pth",
        description=f"{model_name.replace('_', ' ').title()} for {task.replace('_', ' ').title()} for {dataset_name.replace('_', ' ').title()} dataset",
        num_classes=len(set(y)),
        dataset=dataset_name,
        experiment_name=model_name.replace("_", " ").title(),
        precision="16-mixed",
    )

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=experiment_cfg.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=experiment_cfg.batch_size, shuffle=False)

    # Create the model
    model = EncoderOnlyTransformerTSClassifier(
        num_layers=experiment_cfg.num_layers,
        d_model=experiment_cfg.d_model,
        num_heads=experiment_cfg.num_heads,
        d_ff=experiment_cfg.d_ff,
        input_size=X_train_tensor.shape[2],  # Input size (number of features)
        max_len=experiment_cfg.max_len,
        positional_encoding=SinusoidalPositionalEncoding(
            d_model=experiment_cfg.d_model, max_len=experiment_cfg.max_len
        ),
        num_classes=experiment_cfg.num_classes,
        dropout=experiment_cfg.dropout,
        learning_rate=experiment_cfg.learning_rate,
    )

    # Create a logger
    logger = get_logger()
    msg_task("Starting training", logger=logger)

    # Create a trainer
    trainer = lightning.Trainer(
        default_root_dir=Config.root_dir,
        max_epochs=experiment_cfg.num_epochs,
        accelerator="auto",
        devices="auto",
        precision=experiment_cfg.precision,
        logger=[
            CSVLogger(save_dir=Config.log_dir / task / dataset_name, name=f"metrics_{model_name}"),
            TensorBoardLogger(save_dir=Config.log_dir / task / dataset_name, name=f"board_{model_name}_logs"),
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
