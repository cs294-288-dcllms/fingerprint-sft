#!/usr/bin/env python3
"""SkyRL OPD entrypoint with exact online ADFP teacher-logit scoring."""

from __future__ import annotations

from scripts.skyrl_adfp_teacher_patch import install_adfp_teacher_patch

install_adfp_teacher_patch()

import sys

import ray
import torch
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import register_advantage_estimator
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray


class ADFPOnPolicyDistillationTrainer(RayPPOTrainer):
    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        loss_mask = data["loss_mask"]
        teacher_action_log_probs = data["base_action_log_probs"]
        action_log_probs = data["action_log_probs"]
        data["rewards"] = -(
            action_log_probs - teacher_action_log_probs
        ) * loss_mask
        return data


@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


class ADFPOnPolicyDistillationExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return ADFPOnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    ADFPOnPolicyDistillationExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
