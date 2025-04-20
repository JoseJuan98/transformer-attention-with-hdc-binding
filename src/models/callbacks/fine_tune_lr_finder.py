# -*- coding: utf-8 -*-
"""FineTuneLearningRateFinder callback for Lightning Trainer."""

# Third party imports
import lightning
from lightning.pytorch.callbacks import LearningRateFinder


class FineTuneLearningRateFinder(LearningRateFinder):
    """The ``FineTuneLearningRateFinder`` callback enables the user to do a range test of good initial learning rates.

    It reduces the amount of guesswork in picking a good starting learning rate.

    Args:
        min_lr: Minimum learning rate to investigate
        max_lr: Maximum learning rate to investigate
        num_training_steps: Number of learning rates to test
        mode: Search strategy to update learning rate after each batch:

            - ``'exponential'`` (default): Increases the learning rate exponentially.
            - ``'linear'``: Increases the learning rate linearly.

        early_stop_threshold: Threshold for stopping the search. If the
            loss at any point is larger than early_stop_threshold*best_loss
            then the search is stopped. To disable, set to None. Default is 4.
        update_attr: Whether to update the learning rate attribute or not.
        attr_name: Name of the attribute which stores the learning rate. The names 'learning_rate' or 'lr' get
            automatically detected. Otherwise, set the name here.
    """

    def __init__(self, milestones, *args, **kwargs):
        super(FineTuneLearningRateFinder, self).__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        """Do nothing."""
        return

    def on_train_epoch_start(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """On train epoch start, check if we need to run the LR finder."""
        if (
            trainer.current_epoch in self.milestones
            or trainer.current_epoch == trainer.max_epochs - 1
            or trainer.current_epoch == 0
        ):
            self.lr_find(trainer=trainer, pl_module=pl_module)
