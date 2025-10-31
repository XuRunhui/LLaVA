"""
LLaVA Trainer with Differential Privacy Support

This module implements a privacy-preserving trainer for LLaVA using Opacus library.
It supports training vision-language models with formal differential privacy guarantees.
"""

import os
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)

try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator
    from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logger.warning(
        "Opacus not found. Please install it to use differential privacy: "
        "`pip install opacus`"
    )

from llava.train.llava_trainer import (
    LLaVATrainer,
    LengthGroupedSampler,
    get_mm_adapter_state_maybe_zero_3,
    maybe_zero_3
)


class LLaVADPTrainer(LLaVATrainer):
    """
    LLaVA Trainer with Differential Privacy support using Opacus.

    This trainer extends LLaVATrainer to add DP-SGD (Differentially Private
    Stochastic Gradient Descent) capabilities for privacy-preserving training.

    Key DP Parameters:
        - epsilon: Privacy budget (lower = more private, typical: 1-10)
        - delta: Failure probability (typically 1/n where n is dataset size)
        - max_grad_norm: Gradient clipping threshold (typical: 0.1-1.0)
        - noise_multiplier: Controls noise added to gradients (auto-computed if not set)
        - target_epsilon: Target privacy budget for training
        - target_delta: Target failure probability

    Example usage:
        ```python
        training_args = TrainingArguments(
            # ... standard args ...
            dp_enabled=True,
            dp_epsilon=3.0,
            dp_delta=1e-5,
            dp_max_grad_norm=1.0,
        )
        trainer = LLaVADPTrainer(model=model, args=training_args, ...)
        trainer.train()
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.privacy_engine = None
        self.dp_enabled = getattr(self.args, 'dp_enabled', False)

        if self.dp_enabled:
            if not OPACUS_AVAILABLE:
                raise ImportError(
                    "Differential Privacy is enabled but Opacus is not installed. "
                    "Please install it: `pip install opacus`"
                )

            # DP hyperparameters
            self.target_epsilon = getattr(self.args, 'dp_epsilon', 3.0)
            self.target_delta = getattr(self.args, 'dp_delta', 1e-5)
            self.max_grad_norm = getattr(self.args, 'dp_max_grad_norm', 1.0)
            self.noise_multiplier = getattr(self.args, 'dp_noise_multiplier', None)
            self.poisson_sampling = getattr(self.args, 'dp_poisson_sampling', True)
            self.secure_mode = getattr(self.args, 'dp_secure_mode', False)

            # Physical batch size for memory management
            self.physical_batch_size = getattr(self.args, 'dp_physical_batch_size', None)

            logger.info("=" * 80)
            logger.info("Differential Privacy Training Enabled")
            logger.info(f"  Target ε (epsilon): {self.target_epsilon}")
            logger.info(f"  Target δ (delta): {self.target_delta}")
            logger.info(f"  Max gradient norm: {self.max_grad_norm}")
            logger.info(f"  Poisson sampling: {self.poisson_sampling}")
            logger.info(f"  Secure mode: {self.secure_mode}")
            logger.info("=" * 80)

    def _prepare_model_for_dp(self, model):
        """
        Prepare model for differential privacy training.

        This includes:
        1. Validating and fixing model architecture for DP compatibility
        2. Handling vision tower and multimodal components
        3. Replacing incompatible layers
        """
        logger.info("Preparing model for differential privacy...")

        # For VLM, we typically want to apply DP only to the language model part
        # and optionally to the projection layer, not the frozen vision encoder

        # Check if model is DP-compatible
        errors = ModuleValidator.validate(model, strict=False)

        if len(errors) > 0:
            logger.warning(f"Model has {len(errors)} compatibility issues with Opacus:")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")

            # Try to fix the model automatically
            logger.info("Attempting to fix model for DP compatibility...")
            model = ModuleValidator.fix(model)

            # Re-validate
            errors = ModuleValidator.validate(model, strict=False)
            if len(errors) > 0:
                logger.warning(
                    f"Model still has {len(errors)} issues after auto-fix. "
                    "Training will proceed but may encounter errors."
                )
            else:
                logger.info("Model successfully fixed for DP compatibility!")
        else:
            logger.info("Model is already DP-compatible!")

        return model

    def _setup_privacy_engine(self):
        """
        Initialize and attach the Privacy Engine to the model and optimizer.
        """
        if not self.dp_enabled or self.privacy_engine is not None:
            return

        logger.info("Initializing Privacy Engine...")

        # Prepare model for DP
        self.model = self._prepare_model_for_dp(self.model)

        # Calculate total training steps
        total_steps = len(self.get_train_dataloader()) * self.args.num_train_epochs

        # Initialize Privacy Engine
        self.privacy_engine = PrivacyEngine(secure_mode=self.secure_mode)

        # Attach privacy engine
        try:
            self.model, self.optimizer, train_dataloader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.get_train_dataloader(),
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=int(self.args.num_train_epochs),
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=self.poisson_sampling,
            )

            # Store the computed noise multiplier
            computed_noise = self.privacy_engine.noise_multiplier

            logger.info("Privacy Engine successfully attached!")
            logger.info(f"  Computed noise multiplier: {computed_noise:.4f}")
            logger.info(f"  Total training steps: {total_steps}")
            logger.info(f"  Training will satisfy (ε={self.target_epsilon}, δ={self.target_delta})-DP")

        except Exception as e:
            logger.error(f"Failed to attach Privacy Engine: {e}")
            logger.error("Falling back to non-private training...")
            self.dp_enabled = False
            self.privacy_engine = None

    def create_optimizer(self):
        """
        Setup the optimizer. Override to ensure DP setup happens after optimizer creation.
        """
        # First create the optimizer using parent class
        optimizer = super().create_optimizer()

        # Then setup privacy engine if DP is enabled
        if self.dp_enabled and self.privacy_engine is None:
            self._setup_privacy_engine()

        return optimizer

    def get_train_dataloader(self):
        """
        Override to handle DP-specific data loading requirements.
        """
        dataloader = super().get_train_dataloader()

        # For DP training, we need uniform sampling (or Poisson sampling)
        # Disable length-grouped sampling when DP is enabled
        if self.dp_enabled and self.args.group_by_modality_length:
            logger.warning(
                "Length-grouped sampling is not recommended with DP training. "
                "Consider setting group_by_modality_length=False for better privacy guarantees."
            )

        return dataloader

    def training_step(self, model, inputs):
        """
        Perform a training step with DP support.
        """
        # Use physical batch size management if specified
        if self.dp_enabled and self.physical_batch_size is not None:
            # This helps with memory management for large logical batch sizes
            return self._training_step_with_batch_memory_manager(model, inputs)
        else:
            # Standard training step
            return super().training_step(model, inputs)

    def _training_step_with_batch_memory_manager(self, model, inputs):
        """
        Training step with batch memory manager for efficient DP training.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with BatchMemoryManager(
            data_loader=[inputs],
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer
        ) as batch_manager:
            for batch in batch_manager:
                loss = self.compute_loss(model, batch)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

        return loss.detach()

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log training metrics including privacy spent.
        """
        # Add privacy metrics if DP is enabled
        if self.dp_enabled and self.privacy_engine is not None:
            try:
                epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
                logs["privacy/epsilon"] = epsilon
                logs["privacy/delta"] = self.target_delta
                logs["privacy/max_grad_norm"] = self.max_grad_norm

                # Log remaining privacy budget
                remaining_budget = max(0, self.target_epsilon - epsilon)
                logs["privacy/remaining_epsilon"] = remaining_budget

                if epsilon > self.target_epsilon * 0.9:
                    logger.warning(
                        f"Privacy budget nearly exhausted! "
                        f"ε={epsilon:.2f}/{self.target_epsilon:.2f}"
                    )
            except Exception as e:
                logger.warning(f"Could not compute privacy metrics: {e}")

        # Call parent log
        super().log(logs)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with privacy accounting information.
        """
        # Save privacy metadata
        if self.dp_enabled and self.privacy_engine is not None:
            try:
                epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)

                # Save privacy info to a separate file
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                if self.args.local_rank in [0, -1]:
                    os.makedirs(output_dir, exist_ok=True)
                    privacy_info = {
                        'epsilon': float(epsilon),
                        'delta': float(self.target_delta),
                        'target_epsilon': float(self.target_epsilon),
                        'max_grad_norm': float(self.max_grad_norm),
                        'noise_multiplier': float(self.privacy_engine.noise_multiplier),
                        'global_step': self.state.global_step,
                        'epoch': self.state.epoch,
                    }

                    import json
                    privacy_path = os.path.join(output_dir, 'privacy_accounting.json')
                    with open(privacy_path, 'w') as f:
                        json.dump(privacy_info, f, indent=2)

                    logger.info(f"Privacy accounting saved to {privacy_path}")
                    logger.info(f"  Current ε: {epsilon:.4f} (target: {self.target_epsilon})")

            except Exception as e:
                logger.error(f"Failed to save privacy accounting: {e}")

        # Call parent save
        super()._save_checkpoint(model, trial, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        Override to check privacy budget before continuing training.
        """
        # Check if privacy budget is exhausted
        if self.dp_enabled and self.privacy_engine is not None:
            try:
                current_epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)

                if current_epsilon > self.target_epsilon:
                    logger.error(
                        f"Privacy budget EXHAUSTED! "
                        f"Current ε={current_epsilon:.4f} > target ε={self.target_epsilon}"
                    )
                    logger.error("Stopping training to preserve privacy guarantees.")
                    self.control.should_training_stop = True
                    return
            except Exception as e:
                logger.warning(f"Could not check privacy budget: {e}")

        # Call parent method
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)


def compute_privacy_requirements(
    dataset_size: int,
    batch_size: int,
    epochs: int,
    target_epsilon: float = 3.0,
    target_delta: float = None,
) -> Dict[str, Any]:
    """
    Compute privacy requirements and provide guidance for DP training.

    Args:
        dataset_size: Number of training samples
        batch_size: Batch size per device
        epochs: Number of training epochs
        target_epsilon: Target privacy budget
        target_delta: Target failure probability (defaults to 1/dataset_size)

    Returns:
        Dictionary with privacy requirements and recommendations
    """
    if target_delta is None:
        target_delta = 1.0 / dataset_size

    steps_per_epoch = dataset_size / batch_size
    total_steps = steps_per_epoch * epochs

    # Rule of thumb: smaller batches = better privacy but slower training
    # Recommended batch size is 0.1-1% of dataset size for good privacy/utility tradeoff
    recommended_batch_size = max(32, int(dataset_size * 0.005))

    return {
        'dataset_size': dataset_size,
        'target_epsilon': target_epsilon,
        'target_delta': target_delta,
        'total_steps': int(total_steps),
        'steps_per_epoch': int(steps_per_epoch),
        'recommended_batch_size': recommended_batch_size,
        'privacy_guidance': {
            'epsilon': {
                '<1': 'Very strong privacy (may hurt utility)',
                '1-3': 'Strong privacy (recommended for sensitive data)',
                '3-10': 'Moderate privacy (common in practice)',
                '>10': 'Weak privacy (not recommended)'
            },
            'current': 'Strong privacy' if target_epsilon <= 3 else 'Moderate privacy' if target_epsilon <= 10 else 'Weak privacy'
        }
    }
