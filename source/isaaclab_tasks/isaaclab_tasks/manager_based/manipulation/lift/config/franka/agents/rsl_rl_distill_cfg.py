# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
)

@configclass
class DistillStudentRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "distill_student"
    empirical_normalization = False

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu"
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-4,
        gradient_length=48,
        max_grad_norm=1.0,
    )

    load_run = "2025-08-06_14-57-09"
    load_checkpoint = "model_2999.pt"