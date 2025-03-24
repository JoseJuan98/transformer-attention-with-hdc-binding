# -*- coding: utf-8 -*-
"""This module is responsible for running the experiments."""
#
# # Standard imports
# import multiprocessing
# import pathlib
#
# # Third party imports
# import lightning
# import torch
# from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
# from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
# from lightning.pytorch.profilers import SimpleProfiler
# from torch.utils.data import DataLoader
#
# # First party imports
# from experiments import ModelConfig
# from experiments.time_series.dataset import get_ucr_datasets
# from models import EncoderOnlyTransformerTSClassifier, PocketAlgorithm, TimeSeriesSinusoidalPositionalEmbedding
# from utils import Config, get_logger, msg_task, save_csv_logger_metrics_plot
#
# # Local imports
# from .experiment_config import ExperimentConfig
#
#
# class ExperimentRunner:
#     """This class is responsible for running the experiments."""
#
#     def __init__(self, experiment_cfg: ExperimentConfig):
#         """Initializes the ExperimentRunner class."""
#         self.experiment_cfg = experiment_cfg
#
#     def run(self):
#         """Run the experiment."""
#         print("Experiment completed.")
#
#     @staticmethod
#     def _train_model_for_dataset(
#         task: str, dataset_name: str, model_name: str, run_version: str  # TODO: , model: lightning.LightningModule
#     ) -> None:
#         """Trains a model for a dataset."""
#         run_path = Config.model_dir / "runs" / task / model_name
#         test_only = False  # Set to True to only test the model
#
#         log_file = Config.log_dir / f"{task}/{model_name}/{dataset_name}/{run_version}/main.log"
#         logger = get_logger(name="lightning.pytorch.core", log_filename=log_file, propagate=False)
#
#         if not torch.cuda.is_available():
#             raise Exception("CUDA not available. No GPU found.")
#
#         msg_task(msg="Loading data", logger=logger)
#         train_dataset, test_dataset, max_len, num_classes, num_channels = get_ucr_datasets(
#             dsid=dataset_name,
#             extract_path=Config.data_dir / task,
#             logger=logger,
#             # TODO:
#             plot_first_row=False,
#             plot_path=Config.plot_dir / task / f"{dataset_name}_sample.png",
#         )
#
#         # --- Configuration ---
#         experiment_cfg = ModelConfig(
#             num_epochs=30,
#             input_size=num_channels,  # Number of variates (channels)
#             context_length=max_len,  # Sequence length
#             d_model=128,
#             num_heads=8,
#             d_ff=128,
#             num_layers=4,
#             dropout=0.1,
#             batch_size=64,
#             learning_rate=1e-3,
#             device="cuda",
#             model_relative_path=f"runs/{task}/{model_name.lower()}/{dataset_name}/{run_version}/model.pth",
#             description=f"{model_name.replace('_', ' ').title()} for {task.replace('_', ' ').title()} for {dataset_name.replace('_', ' ').title()} dataset",
#             num_classes=num_classes,
#             dataset=dataset_name,
#             experiment_name=model_name.replace("_", " ").title(),
#             precision="16-mixed",
#         )
#
#         logger.info(f"Saving experiment configuration to {run_path}/{run_version}")
#         experiment_cfg.dump(path=run_path / dataset_name / run_version / "experiment_config.json")
#
#         # --- Data Loaders ---
#         cpu_count = multiprocessing.cpu_count()
#         num_workers = cpu_count - 2 if cpu_count > 4 else 1
#
#         train_dataloader = DataLoader(
#             dataset=train_dataset,
#             batch_size=experiment_cfg.batch_size,
#             shuffle=True,
#             num_workers=num_workers,
#         )
#         test_dataloader = DataLoader(
#             dataset=test_dataset,
#             batch_size=experiment_cfg.batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#         )
#
#         # --- Model ---
#         model = EncoderOnlyTransformerTSClassifier(
#             num_layers=experiment_cfg.num_layers,
#             d_model=experiment_cfg.d_model,
#             num_heads=experiment_cfg.num_heads,
#             d_ff=experiment_cfg.d_ff,
#             input_size=experiment_cfg.input_size,  # Use input_size
#             context_length=experiment_cfg.context_length,  # Use context_length
#             positional_encoding=TimeSeriesSinusoidalPositionalEmbedding(
#                 embedding_dim=experiment_cfg.d_model, num_positions=experiment_cfg.context_length
#             ),
#             num_classes=experiment_cfg.num_classes,
#             dropout=experiment_cfg.dropout,
#             learning_rate=experiment_cfg.learning_rate,
#             scaling="mean",
#             mask_input=True,
#             loss_fn=torch.nn.CrossEntropyLoss() if experiment_cfg.num_classes > 2 else torch.nn.BCELoss(),
#             # TODO: remove profiler for systematic runs
#             # TODO: move all configurations outside to make it composible
#             # Torch Profiler: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
#             torch_profiling=torch.profiler.profile(
#                 schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#                 on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=(run_path / run_version).as_posix()),
#                 record_shapes=True,
#                 with_stack=True,
#             ),
#         )
#
#         msg_task(msg=f"Experiment {model_name.replace("_", " ").title()}", logger=logger)
#         logger.info(f"Experiment Configuration:\n\n{experiment_cfg.pretty_str()}\n\n")
#
#         # --- Callbacks ---
#         early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
#         pocket_algorithm = PocketAlgorithm(
#             monitor="val_acc",
#             mode="max",
#             ckpt_filepath=Config.model_dir / pathlib.Path(experiment_cfg.model_relative_path).with_suffix(".ckpt"),
#             model_file_path=Config.model_dir / experiment_cfg.model_relative_path,
#         )
#
#         # --- Trainer ---
#         trainer = lightning.Trainer(
#             default_root_dir=Config.root_dir,
#             max_epochs=experiment_cfg.num_epochs,
#             accelerator="auto",
#             devices="auto",
#             precision=experiment_cfg.precision,
#             logger=[
#                 CSVLogger(save_dir=Config.log_dir / task / model_name, name=dataset_name, version=run_version),
#                 TensorBoardLogger(save_dir=run_path, name=dataset_name, version=run_version),
#             ],
#             log_every_n_steps=1,
#             callbacks=[ModelSummary(max_depth=-1), early_stopping, pocket_algorithm],
#             # TODO: remove profiler for systematic runs
#             # measures all the key methods across Callbacks, DataModules and the LightningModule in the training loop.
#             profiler=SimpleProfiler(filename="simple_profiler"),
#             # If True, runs 1 batch of train, test and val to find any bugs. Also, it can be specified the number of
#             # batches to run as an integer
#             fast_dev_run=False if experiment_cfg.num_epochs > 0 else True,
#         )
#
#         # --- Train and Test ---
#         msg_task(msg="Train and Test", logger=logger)
#         logger.info(
#             f"Training {model_name} for {dataset_name} in {run_version} for {experiment_cfg.num_epochs} epochs..."
#         )
#         if experiment_cfg.num_epochs > 0 and not test_only:
#             trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
#         logger.info(f"{model_name.title()} training finished!")
#         trainer.test(model, dataloaders=test_dataloader)
#
#         # --- Save Model ---
#         # No need to save here, PocketAlgorithm saves the model_relative_path
#         # model_path = Config.model_dir / experiment_cfg.model_path
#         # model_path.parent.mkdir(parents=True, exist_ok=True)
#         # torch.save(model.state_dict(), model_path)
#         # logger.info(f"\nModel saved to {model_path}")
#
#         # --- Plot Metrics ---
#         if experiment_cfg.num_epochs > 0:
#             print(f"Log dir: {trainer.log_dir}")
#             save_csv_logger_metrics_plot(
#                 csv_dir=trainer.log_dir,
#                 experiment=f"{model_name.replace("_", " ").title()} for {dataset_name.replace('_', ' ').title()} in {run_version}",
#                 logger=logger,
#                 plots_path=Config.plot_dir / task / model_name / f"epoch_metrics_{dataset_name}_{run_version}.png",
#             )
