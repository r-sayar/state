import lightning.pytorch as pl
import logging

logger = logging.getLogger(__name__)


class ScheduledFinetuningCallback(pl.Callback):
    """
    A callback to handle a two-phase finetuning schedule.

    Phase 1: Freeze all model weights except for specified modules and train for a set number of steps.
    Phase 2: Unfreeze the entire model and continue training.
    """

    def __init__(self, finetune_steps: int, modules_to_unfreeze: list[str]):
        """
        Args:
            finetune_steps: The number of steps for the initial finetuning phase.
            modules_to_unfreeze: A list of module names (e.g., ['pert_encoder']) to keep trainable
                                 during Phase 1.
        """
        super().__init__()
        self.finetune_steps = finetune_steps
        self.modules_to_unfreeze = modules_to_unfreeze
        self._phase1_completed = False
        self._phase2_completed = False

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx,
    ):
        """Called at the beginning of each training batch."""
        current_step = trainer.global_step

        # Phase 1: Freeze everything except the specified modules at the very beginning
        if current_step == 0 and not self._phase1_completed:
            logger.info(
                f"ScheduledFinetuning: Starting Phase 1. Freezing model except for {self.modules_to_unfreeze} "
                f"for {self.finetune_steps} steps."
            )
            self.freeze_and_unfreeze(pl_module)
            self._phase1_completed = True

        # Phase 2: Unfreeze everything after the specified number of steps
        if (
            current_step >= self.finetune_steps
            and self._phase1_completed
            and not self._phase2_completed
        ):
            logger.info(
                f"ScheduledFinetuning: Starting Phase 2. Unfreezing all model parts at step {current_step}."
            )
            self.unfreeze_all(pl_module)
            self._phase2_completed = True

    def freeze_and_unfreeze(self, model: pl.LightningModule):
        """Freeze all parameters, then unfreeze only the specified modules."""
        # First, freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Then, unfreeze the specified modules
        for name, module in model.named_modules():
            if name in self.modules_to_unfreeze:
                logger.info(f"  - Unfreezing module: {name}")
                for param in module.parameters():
                    param.requires_grad = True

    def unfreeze_all(self, model: pl.LightningModule):
        """Unfreeze all parameters in the model."""
        for param in model.parameters():
            param.requires_grad = True 
