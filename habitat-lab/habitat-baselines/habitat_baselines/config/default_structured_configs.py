#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING

cs = ConfigStore.instance()


@dataclass
class HabitatBaselinesBaseConfig:
    pass


@dataclass
class WBConfig(HabitatBaselinesBaseConfig):
    """Weights and Biases config"""

    # The name of the project on W&B.
    project_name: str = ""
    # Logging entity (like your username or team name)
    entity: str = ""
    # The group ID to assign to the run. Optional to specify.
    group: str = ""
    # The run name to assign to the run. If not specified,
    # W&B will randomly assign a name.
    run_name: str = ""


@dataclass
class EvalConfig(HabitatBaselinesBaseConfig):
    # The split to evaluate on
    split: str = "val"
    use_ckpt_config: bool = True
    should_load_ckpt: bool = True
    # The number of time to run each episode through evaluation.
    # Only works when evaluating on all episodes.
    evals_per_ep: int = 1
    video_option: List[str] = field(
        # available options are "disk" and "tensorboard"
        default_factory=list
    )


@dataclass
class PreemptionConfig(HabitatBaselinesBaseConfig):
    # Append the slurm job ID to the resume state filename if running
    # a slurm job. This is useful when you want to have things from a different
    # job but the same checkpoint dir not resume.
    append_slurm_job_id: bool = False
    # Number of gradient updates between saving the resume state
    save_resume_state_interval: int = 100
    # Save resume states only when running with slurm
    # This is nice if you don't want debug jobs to resume
    save_state_batch_only: bool = False


@dataclass
class ActionDistributionConfig(HabitatBaselinesBaseConfig):
    use_log_std: bool = True
    use_softplus: bool = False
    std_init: float = MISSING
    log_std_init: float = 0.0
    # If True, the std will be a parameter not conditioned on state
    use_std_param: bool = False
    # If True, the std will be clamped to the specified min and max std values
    clamp_std: bool = True
    min_std: float = 1e-6
    max_std: int = 1
    min_log_std: int = -5
    max_log_std: int = 2
    # For continuous action distributions (including gaussian):
    action_activation: str = "tanh"  # ['tanh', '']
    scheduled_std: bool = False


@dataclass
class ObsTransformConfig(HabitatBaselinesBaseConfig):
    type: str = MISSING


@dataclass
class CenterCropperConfig(ObsTransformConfig):
    type: str = "CenterCropper"
    height: int = 256
    width: int = 256
    channels_last: bool = True
    trans_keys: Tuple[str, ...] = (
        "rgb",
        "depth",
        "semantic",
    )


cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.center_cropper",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="center_cropper_base",
    node=CenterCropperConfig,
)


@dataclass
class ResizeShortestEdgeConfig(ObsTransformConfig):
    type: str = "ResizeShortestEdge"
    size: int = 256
    channels_last: bool = True
    trans_keys: Tuple[str, ...] = (
        "rgb",
        "depth",
        "semantic",
    )
    semantic_key: str = "semantic"


cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.resize_shortest_edge",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="resize_shortest_edge_base",
    node=ResizeShortestEdgeConfig,
)


@dataclass
class Cube2EqConfig(ObsTransformConfig):
    type: str = "CubeMap2Equirect"
    height: int = 256
    width: int = 512
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )


cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.cube_2_eq",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="cube_2_eq_base",
    node=Cube2EqConfig,
)


@dataclass
class Cube2FishConfig(ObsTransformConfig):
    type: str = "CubeMap2Fisheye"
    height: int = 256
    width: int = 256
    fov: int = 180
    params: Tuple[float, ...] = (0.2, 0.2, 0.2)
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )

@dataclass
class PolicyFinetuneConfig:
    enabled: bool = False
    lr: float = 1.5e-5
    start_actor_warmup_at: int = 750
    start_actor_update_at: int = 1500
    start_critic_warmup_at: int = 500
    start_critic_update_at: int = 1000
    
cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.cube_2_fish",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="cube_2_fish_base",
    node=Cube2FishConfig,
)


@dataclass
class AddVirtualKeysConfig(ObsTransformConfig):
    type: str = "AddVirtualKeys"
    virtual_keys: Dict[str, int] = field(default_factory=dict)


cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.add_virtual_keys",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="add_virtual_keys_base",
    node=AddVirtualKeysConfig,
)


@dataclass
class Eq2CubeConfig(ObsTransformConfig):
    type: str = "Equirect2CubeMap"
    height: int = 256
    width: int = 256
    sensor_uuids: List[str] = field(
        default_factory=lambda: [
            "BACK",
            "DOWN",
            "FRONT",
            "LEFT",
            "RIGHT",
            "UP",
        ]
    )


cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.eq_2_cube",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="eq_2_cube_base",
    node=Eq2CubeConfig,
)

@dataclass
class TransformerOVONConfig(HabitatBaselinesBaseConfig):
    model_name: str = "llama"
    n_layers: int = 4
    n_heads: int = 8
    n_hidden: int = 512
    n_mlp_hidden: int = 1024
    max_context_length: int = 100
    max_position_embeddings: int = 500
    shuffle_pos_id_for_update: bool = True

@dataclass
class TransformerConfig(HabitatBaselinesBaseConfig):
    model_name: str = "llamarl"
    n_layers: int = 24
    n_heads: int = 16
    n_hidden: int = 2048
    n_mlp_hidden: int = 8192
    kv_size: int = 128
    activation: str = "gelu_new"
    depth_dropout_p: float = 0.0
    inter_episodes_attention: bool = False
    reset_position_index: bool = True
    add_sequence_idx_embed: bool = False
    sequence_embed_type: str = "learnable"
    position_embed_type: str = "rope"
    # from [STABILIZING TRANSFORMERS FOR REINFORCEMENT LEARNING](https://arxiv.org/pdf/1910.06764.pdf)
    gated_residual: bool = False
    # The length of history prepended to the input batch
    context_len: int = 0
    # Force tokens to attend to at most `context_len` tokens
    banded_attention: bool = False
    # Don't process time steps of episodes that didn't start in the batch
    orphan_steps_attention: bool = True
    # Whether to include the context tokens in the loss or not
    add_context_loss: bool = False
    max_position_embeddings: int = 2048

    add_sink_tokens: bool = False

    add_sink_kv: bool = False
    mul_factor_for_sink_attn: bool = True
    is_sink_v_trainable: bool = True
    is_sink_k_trainable: bool = True

    num_sink_tokens: int = 1

    mem_len: int = -1


@dataclass
class VC1Config(HabitatBaselinesBaseConfig):
    avg_pool_size: int = 2
    is_2d_output: bool = False

@dataclass
class TrainingPrecisionConfig(HabitatBaselinesBaseConfig):
    visual_encoder: str = "float32"
    heads: str = "float32"
    others: str = "float32"

@dataclass
class HierarchicalPolicy(HabitatBaselinesBaseConfig):
    high_level_policy: Dict[str, Any] = MISSING
    defined_skills: Dict[str, Any] = field(default_factory=dict)
    use_skills: Dict[str, str] = field(default_factory=dict)


@dataclass
class PolicyConfig(HabitatBaselinesBaseConfig):
    name: str = "PointNavResNetPolicy"
    action_distribution_type: str = "categorical"  # or 'gaussian'
    # If the list is empty, all keys will be included.
    # For gaussian action distribution:
    action_dist: ActionDistributionConfig = ActionDistributionConfig()
    obs_transforms: Dict[str, ObsTransformConfig] = field(default_factory=dict)
    hierarchical_policy: HierarchicalPolicy = MISSING

@dataclass
class CustomPolicyConfig(PolicyConfig):
    transformer_config: TransformerConfig = field(
        default_factory=lambda: TransformerConfig()
    )
    vc1_config: VC1Config = VC1Config()
    training_precision_config: TrainingPrecisionConfig = TrainingPrecisionConfig()

@dataclass
class CustomOVONPolicyConfig(PolicyConfig):
    transformer_config: TransformerOVONConfig = field(
        default_factory=lambda: TransformerOVONConfig()
    )
    vc1_config: VC1Config = VC1Config()
    training_precision_config: TrainingPrecisionConfig = TrainingPrecisionConfig()

    finetune: PolicyFinetuneConfig = PolicyFinetuneConfig()


    name: str = "OVONPolicy"
    backbone: str = "siglip"
    use_augmentations: bool = True
    augmentations_name: str = "jitter+shift"
    use_augmentations_test_time: bool = True
    randomize_augmentations_over_envs: bool = False
    rgb_image_size: int = 224
    resnet_baseplanes: int = 32
    avgpooled_image: bool = False
    drop_path_rate: float = 0.0
    freeze_backbone: bool = True
    pretrained_encoder: Optional[str] = None

    clip_model: str = "RN50"
    add_clip_linear_projection: bool = False
    depth_ckpt: str = ""
    late_fusion: bool = False
    fusion_type: str = "cross_attention"
    attn_heads: int = 3
    use_vis_query: bool = True
    use_residual: bool = True
    residual_vision: bool = True
    rgb_only: bool = True
    use_prev_action: bool = True
    use_odom: bool = False

    unfreeze_xattn: bool = False
    


@dataclass
class PPOConfig(HabitatBaselinesBaseConfig):
    """Proximal policy optimization config"""

    clip_param: float = 0.2
    ppo_epoch: int = 4
    num_mini_batch: int = 2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 2.5e-4
    eps: float = 1e-5
    max_grad_norm: float = 0.5
    num_steps: int = 5
    use_gae: bool = True
    use_linear_lr_decay: bool = False
    use_linear_clip_decay: bool = False
    gamma: float = 0.99
    tau: float = 0.95
    reward_window_size: int = 50
    use_normalized_advantage: bool = False
    hidden_size: int = 512
    entropy_target_factor: float = 0.0
    use_adaptive_entropy_pen: bool = False
    use_clipped_value_loss: bool = True
    # Use double buffered sampling, typically helps
    # when environment time is similar or larger than
    # policy inference time during rollout generation
    # Not that this does not change the memory requirements
    use_double_buffered_sampler: bool = False

@dataclass
class CustomPPOConfig(PPOConfig):
    updates_per_rollout: int = 1
    full_updates_per_rollout: int = 1
    percent_envs_update: Optional[float] = None
    slice_in_partial_update: bool = False
    update_stale_kv: bool = False
    update_stale_values: bool = False
    update_stale_action_probs: bool = False
    shuffle_old_episodes: bool = False
    shift_scene_every: int = 0
    shift_scene_staggered: bool = True
    force_env_reset_every: int = -1

    context_len: int = 32
    skipgrad: bool = False
    skipgrad_factor1: float = 0.1
    skipgrad_factor2: int = 2

    # Not used at the moment
    optimizer_name: str = "adam"
    adamw_weight_decay: float = 0.01

    lr_scheduler: str = ""
    warmup: bool = False
    warmup_total_iters: int = 300
    warmup_start_factor: float = 0.3
    warmup_end_factor: float = 1

    initial_lr: float = 1e-7
    lr_scheduler_restart_step: int = 5_000_000
    lrsched_T_0: int = 2500
    lrsched_T_mult: int = 1
    lrsched_T_max: int = 2500
    lrsched_eta_min: float = 0

    grad_accum_mini_batches: int = 1
    storage_low_precision: bool = False
    ignore_old_obs_grad: bool = False
    gradient_checkpointing: bool = False

    acting_context: Optional[int] = None

    shortest_path_follower: bool = False
    init_checkpoint: str = ""
    append_global_avg_pool: bool = False

    num_warmup_steps_per_env: int = 1000

@dataclass
class VERConfig(HabitatBaselinesBaseConfig):
    """Variable experience rollout config"""

    variable_experience: bool = True
    num_inference_workers: int = 2
    overlap_rollouts_and_learn: bool = False


@dataclass
class AuxLossConfig(HabitatBaselinesBaseConfig):
    pass


@dataclass
class CPCALossConfig(AuxLossConfig):
    """Action-conditional contrastive predictive coding loss"""

    k: int = 20
    time_subsample: int = 6
    future_subsample: int = 2
    loss_scale: float = 0.1

@dataclass
class SegmLossConfig(AuxLossConfig):
    loss_scale: float = 1.0

@dataclass
class DDPPOConfig(HabitatBaselinesBaseConfig):
    """Decentralized distributed proximal policy optimization config"""

    sync_frac: float = 0.6
    distrib_backend: str = "GLOO"
    rnn_type: str = "GRU"
    num_recurrent_layers: int = 1
    backbone: str = "resnet18"
    # Visual encoder backbone
    pretrained_weights: str = "data/ddppo-models/gibson-2plus-resnet50.pth"
    # Initialize with pretrained weights
    pretrained: bool = False
    # Loads just the visual encoder backbone weights
    pretrained_encoder: bool = False
    # Whether the visual encoder backbone will be trained
    train_encoder: bool = True
    # Whether to reset the critic linear layer
    reset_critic: bool = True
    # Forces distributed mode for testing
    force_distributed: bool = False


@dataclass
class RLConfig(HabitatBaselinesBaseConfig):
    """Reinforcement learning config"""

    preemption: PreemptionConfig = PreemptionConfig()
    policy: PolicyConfig = PolicyConfig()
    ppo: PPOConfig = PPOConfig()
    ddppo: DDPPOConfig = DDPPOConfig()
    ver: VERConfig = VERConfig()
    auxiliary_losses: Dict[str, AuxLossConfig] = field(default_factory=dict)

@dataclass
class HrlDefinedSkillConfig(HabitatBaselinesBaseConfig):
    """
    Defines a low-level skill to be used in the hierarchical policy.
    """

    skill_name: str = MISSING
    name: str = "PointNavResNetPolicy"
    action_distribution_type: str = "gaussian"
    load_ckpt_file: str = ""
    max_skill_steps: int = 200
    # If true, the stop action will be called if the skill times out.
    force_end_on_timeout: bool = True
    # Overrides the config file of a neural network skill rather than loading
    # the config file from the checkpoint file.
    force_config_file: str = ""
    at_resting_threshold: float = 0.15
    # If true, this will apply the post-conditions of the skill after it
    # terminates.
    apply_postconds: bool = False

    # If true, do not call grip_actions automatically when calling high level skills.
    # Do not check either if an arm action necessarily exists.
    ignore_grip: bool = False
    obs_skill_inputs: List[str] = field(default_factory=list)
    obs_skill_input_dim: int = 3
    start_zone_radius: float = 0.3
    # For the oracle navigation skill
    action_name: str = "base_velocity"
    stop_thresh: float = 0.001
    # For the reset_arm_skill
    reset_joint_state: List[float] = MISSING
    # The set of PDDL action names (as defined in the PDDL domain file) that
    # map to this skill. If not specified,the name of the skill must match the
    # PDDL action name.
    pddl_action_names: Optional[List[str]] = None
    turn_power_x: float = 0.0
    turn_power_y: float = 0.0
    # Additional skill data to be passed to the skill. Included so extending to
    # new skills doesn't require adding new Hydra dataclass configs.
    skill_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HierarchicalPolicyConfig(HabitatBaselinesBaseConfig):
    high_level_policy: Dict[str, Any] = MISSING
    # Names of the skills to not load.
    ignore_skills: List[str] = field(default_factory=list)
    defined_skills: Dict[str, HrlDefinedSkillConfig] = field(
        default_factory=dict
    )
    use_skills: Dict[str, str] = field(default_factory=dict)


@dataclass
class CustomRLConfig(RLConfig):
    ppo: CustomPPOConfig = CustomPPOConfig()
    policy: Dict[str, Any] = field(
        default_factory=lambda: {"main_agent": CustomPolicyConfig()}
    )

@dataclass
class CustomOVON_RLConfig(RLConfig):
    ppo: CustomPPOConfig = CustomPPOConfig()
    # policy: Dict[str, Any] = field(
    #     default_factory=lambda: {"depth_ckpt": 'a.ckpt'}
    # )

    #policy: CustomOVONPolicyConfig = CustomOVONPolicyConfig()
    
    # policy: Dict[str, Any] = field(
    #     default_factory=lambda: {"main_agent": CustomOVONPolicyConfig()}
    # )

    policy: Dict[str, Any] = field(
        default_factory=lambda: {
            "main_agent": CustomOVONPolicyConfig(),
            "depth_ckpt": 'a.ckpt',
            **CustomOVONPolicyConfig().__dict__  # Include policy config at top level
        }
    )

@dataclass
class ORBSLAMConfig(HabitatBaselinesBaseConfig):
    """ORB-SLAM config"""

    slam_vocab_path: str = "habitat_baselines/slambased/data/ORBvoc.txt"
    slam_settings_path: str = (
        "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
    )
    map_cell_size: float = 0.1
    map_size: int = 40
    # camera_height = (
    #     get_task_config().habitat.simulator.depth_sensor.position[1]
    # )
    camera_height: float = II("habitat.simulator.depth_sensor.position[1]")
    beta: int = 100
    # h_obstacle_min = 0.3 * _C.orbslam2.camera_height
    h_obstacle_min: float = 0.3 * 1.25
    # h_obstacle_max = 1.0 * _C.orbslam2.camera_height
    h_obstacle_max = 1.0 * 1.25
    d_obstacle_min: float = 0.1
    d_obstacle_max: float = 4.0
    preprocess_map: bool = True
    # Note: hydra does not support basic operators in interpolations of numbers
    # see https://github.com/omry/omegaconf/issues/91 for more details
    # min_pts_in_obstacle = (
    #     get_task_config().habitat.simulator.depth_sensor.width / 2.0
    # )
    # Workaround for the operation above:
    # (640 is the default habitat depth sensor width)
    min_pts_in_obstacle: float = 640 / 2.0
    angle_th: float = math.radians(15)  # float(np.deg2rad(15))
    dist_reached_th: float = 0.15
    next_waypoint_th: float = 0.5
    num_actions: int = 3
    dist_to_stop: float = 0.05
    planner_max_steps: int = 500
    # depth_denorm = (
    #     get_task_config().habitat.simulator.depth_sensor.max_depth
    # )
    depth_denorm: float = II("habitat.simulator.depth_sensor.max_depth")


@dataclass
class ProfilingConfig(HabitatBaselinesBaseConfig):
    capture_start_step: int = -1
    num_steps_to_capture: int = -1

@dataclass
class VectorEnvFactoryConfig(HabitatBaselinesBaseConfig):
    """
    `_target_` points to the `VectorEnvFactory` to setup the vectorized
    environment. Defaults to the Habitat vectorized environment setup.
    """

    _target_: str = "habitat_baselines.common.HabitatVectorEnvFactory"

@dataclass
class EvaluatorConfig(HabitatBaselinesBaseConfig):
    """
    `_target_` points to the `Evaluator` class to instantiate to evaluate the
    policy during evaluation mode.
    """

    _target_: str = (
        "habitat_baselines.rl.ppo.habitat_evaluator.HabitatEvaluator"
    )
@dataclass
class HydraCallbackConfig(HabitatBaselinesBaseConfig):
    """
    Generic callback option for Hydra. Used to create the `_target_` class or
    call the `_target_` method.
    """

    _target_: Optional[str] = None

@dataclass
class HabitatBaselinesConfig(HabitatBaselinesBaseConfig):
    # task config can be a list of configs like "A.yaml,B.yaml"
    # base_task_config_path: str = (
    #     "habitat-lab/habitat/config/task/pointnav.yaml"
    # )
    # cmd_trailing_opts: List[str] = field(default_factory=list)
    trainer_name: str = "ppo"
    torch_gpu_id: int = 0
    video_render_views: List[str] = field(default_factory=list)
    tensorboard_dir: str = "tb"
    writer_type: str = "tb"
    video_dir: str = "video_dir"
    video_fps: int = 10
    test_episode_count: int = -1
    # path to ckpt or path to ckpts dir
    eval_ckpt_path_dir: str = "data/checkpoints"
    num_environments: int = 16
    num_processes: int = -1  # deprecated
    checkpoint_folder: str = "data/checkpoints"
    num_updates: int = 10000
    num_checkpoints: int = 10
    # Number of model updates between checkpoints
    checkpoint_interval: int = -1
    total_num_steps: float = -1.0
    log_interval: int = 10
    log_file: str = "train.log"
    force_blind_policy: bool = False
    verbose: bool = True
    eval_keys_to_include_in_name: List[str] = field(default_factory=list)
    # For our use case, the CPU side things are mainly memory copies
    # and nothing of substantive compute. PyTorch has been making
    # more and more memory copies parallel, but that just ends up
    # slowing those down dramatically and reducing our perf.
    # This forces it to be single threaded.  The default
    # value is left as false as it's different from how
    # PyTorch normally behaves, but all configs we provide
    # set it to true and yours likely should too
    force_torch_single_threaded: bool = False
    # Weights and Biases config
    wb: WBConfig = WBConfig()
    # When resuming training or evaluating, will use the original
    # training config if load_resume_state_config is True
    load_resume_state_config: bool = True
    eval: EvalConfig = EvalConfig()
    profiling: ProfilingConfig = ProfilingConfig()

@dataclass
class CustomHabitatBaselinesRLConfig(HabitatBaselinesConfig):
    reset_envs_after_update: bool = False
    call_after_update_env: bool = False
    separate_envs_and_policy: bool = False
    separate_rollout_and_policy: bool = False
    rollout_on_cpu: bool = False
    eval_data_dir: str = MISSING
    rl: CustomRLConfig = CustomRLConfig()

@dataclass
class AgentAccessMgrConfig(HabitatBaselinesBaseConfig):
    type: str = "SingleAgentAccessMgr"
    ###############################
    # Population play configuration
    num_agent_types: int = 1
    num_active_agents_per_type: List[int] = field(default_factory=lambda: [1])
    num_pool_agents_per_type: List[int] = field(default_factory=lambda: [1])
    agent_sample_interval: int = 20
    force_partner_sample_idx: int = -1
    # A value of -1 means not configured.
    behavior_latent_dim: int = -1
    # Configuration option for evaluating BDP. If True, then include all
    # behavior agent IDs in the batch. If False, then we will randomly sample IDs.
    force_all_agents: bool = False
    discrim_reward_weight: float = 1.0
    allow_self_play: bool = False
    self_play_batched: bool = False
    # If specified, this will load the policies for the type 1 population from
    # the checkpoint file at the start of training. Used to independently train
    # the type 1 population, and then train a separate against this population.
    load_type1_pop_ckpts: Optional[List[str]] = None
    ###############################

@dataclass
class CustomHabitatOVONBaselinesRLConfig(HabitatBaselinesConfig):
    reset_envs_after_update: bool = False
    call_after_update_env: bool = False
    separate_envs_and_policy: bool = False
    separate_rollout_and_policy: bool = False
    rollout_on_cpu: bool = False
    eval_data_dir: str = MISSING
    video_render_views: str = 'rgb'
    rl: CustomOVON_RLConfig = CustomOVON_RLConfig()

@dataclass
class HabitatBaselinesRLConfig(HabitatBaselinesConfig):
    rl: RLConfig = RLConfig()


@dataclass
class HabitatBaselinesILConfig(HabitatBaselinesConfig):
    il: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HabitatBaselinesORBSLAMConfig(HabitatBaselinesConfig):
    orbslam2: ORBSLAMConfig = ORBSLAMConfig()


@dataclass
class HabitatBaselinesSPAConfig(HabitatBaselinesConfig):
    sense_plan_act: Any = MISSING


# Register configs to config store
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_OVON_rl_config_base",
    node=CustomHabitatOVONBaselinesRLConfig(),
)

cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl_config_base",
    node=HabitatBaselinesRLConfig(),
)
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_orbslam2_config_base",
    node=HabitatBaselinesORBSLAMConfig,
)
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_il_config_base",
    node=HabitatBaselinesILConfig,
)
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_spa_config_base",
    node=HabitatBaselinesSPAConfig,
)
cs.store(
    group="habitat_baselines/rl/policy", name="policy_base", node=PolicyConfig
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.cpca",
    group="habitat_baselines/rl/auxiliary_losses",
    name="cpca",
    node=CPCALossConfig,
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.segm",
    group="habitat_baselines/rl/auxiliary_losses",
    name="segm",
    node=SegmLossConfig,
)

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class HabitatBaselinesConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://habitat_baselines/config/",
        )
