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
    max_iterations = 3000
    save_interval = 50
    experiment_name = "franka_lift"
    empirical_normalization = True

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="adaptive",
        student_hidden_dims=[1024, 512, 256], # 512, 256, 128 for resnet18
        teacher_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-4,
        gradient_length=48,
        max_grad_norm=1.0,
    )

    load_run = "ExpertTeacher3000"
    load_checkpoint = "model_2999.pt"