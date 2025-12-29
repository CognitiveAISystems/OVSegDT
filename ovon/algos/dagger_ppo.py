#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import inspect
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import LagrangeInequalityCoefficient, inference_mode, linear_decay, get_num_actions
from torch import Tensor
from ovon.obs_transformers.relabel_teacher_actions import RelabelTeacherActions
import math

EPS_PPO = 1e-5


class DAgger_PPO(nn.Module):
    entropy_coef: Union[float, LagrangeInequalityCoefficient]

    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config):
        config = {k.lower(): v for k, v in config.items()}
        param_dict = dict(actor_critic=actor_critic)
        sig = inspect.signature(cls.__init__)
        for p in sig.parameters.values():
            if p.name == "self" or p.name in param_dict:
                continue

            assert p.name in config, "{} parameter '{}' not in config".format(
                cls.__name__, p.name
            )

            param_dict[p.name] = config[p.name]

        return cls(**param_dict)

    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
    ) -> None:
        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.device = next(actor_critic.parameters()).device

        if (
            use_adaptive_entropy_pen
            and hasattr(self.actor_critic, "num_actions")
            and getattr(self.actor_critic, "action_distribution_type", None)
            == "gaussian"
        ):
            num_actions = self.actor_critic.num_actions

            self.entropy_coef = LagrangeInequalityCoefficient(
                -float(entropy_target_factor) * num_actions,
                init_alpha=entropy_coef,
                alpha_max=1.0,
                alpha_min=1e-4,
                greater_than=True,
            ).to(device=self.device)

        self.use_normalized_advantage = use_normalized_advantage

        state_encoder_params = []
        blacklisted_layers = [
            "visual_encoder",
            "action_distribution",
            "critic",
            "visual_fc",
        ]
        if actor_critic.unfreeze_xattn:
            blacklisted_layers.remove("cross_attention")

        whitelisted_names = []
        for name, param in actor_critic.named_parameters():
            is_blacklisted = False
            for layer in blacklisted_layers:
                if layer in name:
                    is_blacklisted = True
                    break
            if not is_blacklisted:
                state_encoder_params.append(param)
                whitelisted_names.append(name)

        params = [
            {
                "params": list(
                    filter(
                        lambda p: p.requires_grad,
                        actor_critic.critic.parameters(),
                    )
                ),
                "lr": lr,
                "eps": eps,
            },
            {
                "params": state_encoder_params,
                "lr": 0.0,
                "eps": eps,
            },
            {
                "params": list(actor_critic.action_distribution.parameters()),
                "lr": 0.0,
                "eps": eps,
            },
        ]

        if len(params) > 0:
            optim_cls = optim.Adam
            optim_kwargs = dict(
                params=params,
                lr=lr,
                eps=eps,
            )
            signature = inspect.signature(optim_cls.__init__)
            if "foreach" in signature.parameters:
                optim_kwargs["foreach"] = True
            else:
                try:
                    import torch.optim._multi_tensor
                except ImportError:
                    pass
                else:
                    optim_cls = torch.optim._multi_tensor.Adam

            self.optimizer = optim_cls(**optim_kwargs)
        else:
            self.optimizer = None

        self.non_ac_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("actor_critic.")
        ]
        
        # ------------------------------------------------------------------
        # Entropy-based adaptive DAGGER ↔ PPO switching (replaces frame-based)
        # ------------------------------------------------------------------
        # If the policy is highly uncertain (high entropy), rely more on
        # behaviour cloning; when it becomes confident (low entropy), rely on
        # PPO.  A small exponential moving average (EMA) of the policy entropy
        # is used to make the signal smooth.
        self.entropy_low = 0.35   # Below this → full PPO
        self.entropy_high = 0.75  # Above this → full BC
        self.entropy_ema_decay = 0.95
        self.entropy_ema: Optional[float] = None
        self.ppo_ratio = 0.0  # will be updated every learn step

        # Q-function head for diagnostics only (does NOT affect learning).
        # It will be lazily instantiated the first time we need it so that we
        # know the feature dimensionality.
        self.q_head: Optional[nn.Module] = None

        # Coefficient for auxiliary Q-value loss
        self.q_loss_coef = 1.0

        # Store lr/eps for later addition of q_head param group
        self._lr = lr
        self._eps = eps

        # Tracking counters for logging
        self.ppo_usage_count: float = 0.0
        self.dagger_usage_count: float = 0.0
        self.total_samples: float = 0.0

        # Running frame counter (kept for compatibility with external callers)
        self.total_frames: int = 0

        # Number of discrete actions (for auxiliary Q head)
        try:
            self.num_actions = get_num_actions(self.actor_critic.action_space)
        except Exception:
            # Fallback: if policy exposes dim_actions or similar
            self.num_actions = getattr(self.actor_critic, "dim_actions", 1)

        
        feat_dim = 512
        hidden_dim = max(128, feat_dim // 2)
        self.q_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_actions),
        ).to(self.device)
        if self.optimizer is not None:
            # Append q_head parameters to first param group to keep
            # the total number of groups constant (scheduler expects 3).
            self.optimizer.param_groups[0]["params"].extend(self.q_head.parameters())

        self.segm_step = 0

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict[prefix + 'entropy_ema'] = self.entropy_ema
        state_dict[prefix + 'ppo_ratio'] = self.ppo_ratio
        state_dict[prefix + 'segm_step'] = self.segm_step
        state_dict[prefix + 'total_frames'] = self.total_frames
        state_dict[prefix + 'ppo_usage_count'] = self.ppo_usage_count
        state_dict[prefix + 'dagger_usage_count'] = self.dagger_usage_count
        state_dict[prefix + 'total_samples'] = self.total_samples
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.entropy_ema = state_dict.pop('entropy_ema', None)
        self.ppo_ratio = state_dict.pop('ppo_ratio', 0.0)
        self.segm_step = state_dict.pop('segm_step', 0)
        self.total_frames = state_dict.pop('total_frames', 0)
        self.ppo_usage_count = state_dict.pop('ppo_usage_count', 0.0)
        self.dagger_usage_count = state_dict.pop('dagger_usage_count', 0.0)
        self.total_samples = state_dict.pop('total_samples', 0.0)

        # Handle q_head instantiation if it exists in state_dict but not in self
        if self.q_head is None and 'q_head.0.weight' in state_dict:
            weight = state_dict['q_head.0.weight']
            feat_dim = weight.shape[1]
            hidden_dim = weight.shape[0]
            
            self.q_head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_actions),
            ).to(self.device)
            
            if self.optimizer is not None:
                # Add parameters to the optimizer so they are updated/loaded correctly
                self.optimizer.param_groups[0]["params"].extend(self.q_head.parameters())

        super().load_state_dict(state_dict, strict=strict)

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"]  # type: ignore
            - rollouts.buffers["value_preds"]
        )
        if not self.use_normalized_advantage:
            return advantages

        var, mean = self._compute_var_mean(advantages[torch.isfinite(advantages)])

        advantages -= mean

        return advantages.mul_(torch.rsqrt(var + EPS_PPO))

    @staticmethod
    def _compute_var_mean(x):
        return torch.var_mean(x)

    def _set_grads_to_none(self):
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                p.grad = None

    def _update_ppo_ratio(self):
        """Entropy-based adaptive gate: sets self.ppo_ratio ∈ [0,1].
        High entropy ⇒ 0 (behaviour cloning), low entropy ⇒ 1 (PPO)."""

        # Still warming-up: stay with behaviour cloning until we have an EMA.
        if self.entropy_ema is None:
            self.ppo_ratio = 0.0
            return

        if self.entropy_ema >= self.entropy_high:
            self.ppo_ratio = 0.0
        elif self.entropy_ema <= self.entropy_low:
            self.ppo_ratio = 1.0
        else:
            # Linear interpolation between the two bounds
            self.ppo_ratio = (
                (self.entropy_high - self.entropy_ema)
                / (self.entropy_high - self.entropy_low)
            )


    def update(
        self,
        rollouts: RolloutStorage,
        current_frames: Optional[int] = None,
    ) -> Dict[str, float]:

        self.segm_step += 1

        # Update frame count if provided
        if current_frames is not None:
            self.total_frames = current_frames
        
        # Update PPO to DAgger ratio
        self._update_ppo_ratio()

        advantages = self.get_advantages(rollouts)

        learner_metrics = collections.defaultdict(list)

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        # Store learner_metrics as instance variable to share with after_step()
        self._learner_metrics = learner_metrics

        # Reset usage tracking for this update
        self.ppo_usage_count = 0
        self.dagger_usage_count = 0
        self.total_samples = 0

        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for _bid, batch in enumerate(data_generator):
                self._set_grads_to_none()

                # Get policy actions from rollout
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    aux_loss_res,
                ) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["prev_segm_masks"],
                    batch["masks"],
                    batch["actions"],
                    batch["rnn_build_seq_info"],
                )

                # Get the teacher actions from observations
                teacher_actions = batch["observations"][
                                        RelabelTeacherActions.TEACHER_LABEL
                                    ].type(torch.long)

                # Evaluate teacher actions once so that we also obtain the critic value
                (
                    values_teacher,
                    action_log_probs_teacher,
                    _dist_entropy_teacher,  # not needed for loss
                    _,
                    _,
                ) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["prev_segm_masks"],
                    batch["masks"],
                    teacher_actions,
                    batch["rnn_build_seq_info"],
                )

                # Get batch size
                batch_size = batch["actions"].shape[0]
                
                # Entropy–based probabilistic gate
                prob_ppo = torch.full((batch_size, 1), self.ppo_ratio, device=self.device)
                
                # Complementary probability for DAgger
                do_bc = 1.0 - prob_ppo
                
                # Binary mask used only for some logging quantities
                do_bc_bin = (do_bc > 0.5).float()

                ################################## logging ##################################
                # Get the current policy's action distribution for comparison
                with torch.no_grad():
                    features, _, _ = self.actor_critic.net(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["prev_segm_masks"],
                        batch["masks"],
                        batch["rnn_build_seq_info"],
                    )
                    distribution = self.actor_critic.action_distribution(features)
                    policy_actions = distribution.mode()
                    
                    # Compare with teacher actions
                    if policy_actions.shape != teacher_actions.shape:
                        if len(policy_actions.shape) == len(teacher_actions.shape) - 1:
                            policy_actions = policy_actions.unsqueeze(-1)
                        elif len(policy_actions.shape) - 1 == len(teacher_actions.shape):
                            policy_actions = policy_actions.squeeze(-1)
                    
                    # Measure action agreement
                    action_match = (policy_actions == teacher_actions).float()
                    learner_metrics["action_agreement"].append(action_match.mean().item())
                    
                    # Log how often BC is applied despite actions already matching
                    agreement_stats = (action_match + do_bc_bin).floor()  # 2 if both, 1 if either, 0 if neither
                    learner_metrics["bc_with_matching_actions"].append(
                        (agreement_stats > 1.5).float().sum() / (do_bc_bin.sum() + 1e-8)
                    )
                
                # Lazily create q_head once we know feature dimension
                if self.q_head is None:
                    feat_dim = features.shape[-1]
                    hidden_dim = max(128, feat_dim // 2)
                    self.q_head = nn.Sequential(
                        nn.Linear(feat_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, self.num_actions),
                    ).to(self.device)
                    if self.optimizer is not None:
                        # Append q_head parameters to first param group to keep
                        # the total number of groups constant (scheduler expects 3).
                        self.optimizer.param_groups[0]["params"].extend(self.q_head.parameters())

                # Compute value deltas for diagnostics
                value_deltas = values_teacher - values
                value_improvement = value_deltas * do_bc  # Positive values where teacher is better
                
                # Record these statistics for debugging
                record_min_mean_max(values_teacher, "value_teacher")
                record_min_mean_max(value_deltas, "value_delta")
                learner_metrics["mean_value_improvement"].append(value_improvement.sum() / (do_bc.sum() + 1e-8))
                learner_metrics["improvement_ratio"].append(
                    (value_deltas > 0).float().mean().item()  # Fraction where teacher is better
                )
                ################################## done logging ##################################

                # Calculate PPO action loss for current policy action
                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = batch["advantages"] * ratio
                surr2 = batch["advantages"] * (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                )
                action_loss_ppo = -torch.min(surr1, surr2)

                # Behaviour-cloning loss (negative log-likelihood of teacher action)
                action_loss_dagger = -action_log_probs_teacher

                # Blend the two policy losses according to probabilistic gate
                action_loss = (prob_ppo * action_loss_ppo + do_bc * action_loss_dagger)
                print("Action loss shape", action_loss.shape)

                # -----  VALUE LOSS  -----
                values = values.float()
                orig_values = values

                if self.use_clipped_value_loss:
                    delta = values.detach() - batch["value_preds"]
                    value_pred_clipped = batch["value_preds"] + delta.clamp(
                        -self.clip_param, self.clip_param
                    )

                    values = torch.where(
                        delta.abs() < self.clip_param,
                        values,
                        value_pred_clipped,
                    )

                # NOTE: do NOT mask the critic loss – we want it to learn everywhere
                value_loss = 0.5 * F.mse_loss(
                    values, batch["returns"], reduction="mean"
                )

                # Apply importance sampling weights if VER is used
                if "is_coeffs" in batch:
                    assert isinstance(batch["is_coeffs"], torch.Tensor)
                    ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)
                    print("VER coefs", torch.max(batch["is_coeffs"]), torch.mean(batch["is_coeffs"]), torch.min(batch["is_coeffs"]))

                    def mean_fn(t):
                        return torch.mean(ver_is_coeffs * t)
                else:
                    mean_fn = torch.mean

                # Compute Q-value loss for executed action (TD target = returns)
                if self.q_head is not None:
                    features_detached = features.detach()
                    q_values_train = self.q_head(features_detached)
                    # Logging Q-values (no grad):
                    with torch.no_grad():
                        q_policy_log   = q_values_train.gather(1, policy_actions.long().view(-1,1)).squeeze(-1)
                        q_teacher_log  = q_values_train.gather(1, teacher_actions.long().view(-1,1)).squeeze(-1)
                        record_min_mean_max(q_policy_log,  "q_policy")
                        record_min_mean_max(q_teacher_log, "q_teacher")

                    selected_q = q_values_train.gather(1, batch["actions"].long().view(-1,1)).squeeze(-1)
                    q_loss_sample = 0.5 * (selected_q - batch["returns"].detach().squeeze(-1)) ** 2
                    q_loss = mean_fn(q_loss_sample)
                else:
                    q_loss = torch.tensor(0.0, device=self.device)

                action_loss, value_loss, dist_entropy = map(
                    mean_fn,
                    (action_loss, value_loss, dist_entropy),
                )

                # ------------------------------------------------
                # Update EMA of policy entropy (for adaptive gating)
                # ------------------------------------------------
                with torch.no_grad():
                    current_entropy = dist_entropy.detach().item()
                    print("current_entropy", current_entropy)
                    if self.entropy_ema is None:
                        self.entropy_ema = current_entropy
                    else:
                        self.entropy_ema = (
                            self.entropy_ema_decay * self.entropy_ema
                            + (1.0 - self.entropy_ema_decay) * current_entropy
                        )
                    # Re-compute PPO ratio with the updated EMA so that later
                    # mini-batches in this update see the refined value.
                    self._update_ppo_ratio()

                # ----------------  TOTAL LOSS  -----------------
                # Base losses: value and policy (BC ↔ PPO depending on phase)
                all_losses = [
                    self.value_loss_coef * value_loss,
                    action_loss,
                ]
                
                if isinstance(self.entropy_coef, float):
                    all_losses.append(-self.entropy_coef * dist_entropy)
                else:
                    all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

                all_losses.append(self.q_loss_coef * q_loss)
                segm_updates_enabled = self.actor_critic.net.segm_update_config.enabled
                segm_updates_frequency = self.actor_critic.net.segm_update_config.frequency

                if segm_updates_enabled and self.segm_step == segm_updates_frequency:
                    segmentation_loss = self.actor_critic.net.segment(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["rnn_build_seq_info"],
                    )
                    all_losses.append(segmentation_loss['loss'])
                    with inference_mode():
                        learner_metrics["aux_segmentation"].append(segmentation_loss['loss'])
                        learner_metrics["aux_dice"].append(segmentation_loss['dice'])
                        learner_metrics["aux_bce"].append(segmentation_loss['bce'])
                    self.segm_step = 0
                else:
                    all_losses.extend(torch.mean(do_bc*v["loss"]) for v in aux_loss_res.values())
                    for v in aux_loss_res.values():
                        print("Before:", torch.mean(v["loss"]), "after:", torch.mean(do_bc*v["loss"]))

                total_loss = torch.stack(all_losses).sum()

                # ------------------------------------------------
                # Back-prop and optimisation step (unchanged)
                total_loss = self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)
                grad_norm = self.before_step()
                self.optimizer.step()
                self.after_step()

                # ----------------  LOGGING  -------------------
                with inference_mode():
                    ppo_samples    = prob_ppo.sum().item()
                    dagger_samples = do_bc.sum().item()
                    self.ppo_usage_count    += ppo_samples
                    self.dagger_usage_count += dagger_samples
                    self.total_samples      += prob_ppo.numel()

                    ppo_ratio = ppo_samples / (prob_ppo.numel() + 1e-8)
                    learner_metrics["ppo_usage_ratio"].append(ppo_ratio)
                    learner_metrics["dagger_usage_ratio"].append(1.0 - ppo_ratio)
                    # Log the current gating ratio and EMA entropy
                    learner_metrics["ema_entropy"].append(self.entropy_ema)

                    if "is_coeffs" in batch:
                        record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
                    record_min_mean_max(orig_values, "value_pred")
                    record_min_mean_max(ratio, "prob_ratio")

                    learner_metrics["value_loss"].append(value_loss)
                    learner_metrics["action_loss"].append(action_loss)
                    learner_metrics["action_loss_ppo"].append(
                        (action_loss_ppo * prob_ppo).sum() / (ppo_samples + 1e-8)
                    )
                    learner_metrics["action_loss_dagger"].append(
                        (action_loss_dagger * do_bc).sum() / (dagger_samples + 1e-8)
                    )
                    learner_metrics["dist_entropy"].append(dist_entropy)
                    learner_metrics["q_loss"].append(q_loss)
                    if epoch == (self.ppo_epoch - 1):
                        learner_metrics["ppo_fraction_clipped"].append(
                            (ratio > (1.0 + self.clip_param)).float().mean()
                            + (ratio < (1.0 - self.clip_param)).float().mean()
                        )

                    learner_metrics["grad_norm"].append(grad_norm)
                    if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                        learner_metrics["entropy_coef"].append(
                            self.entropy_coef().detach()
                        )

                    for name, res in aux_loss_res.items():
                        for k, v in res.items():
                            learner_metrics[f"aux_{name}_{k}"].append(v.detach())

                    if "is_stale" in batch:
                        assert isinstance(batch["is_stale"], torch.Tensor)
                        learner_metrics["fraction_stale"].append(
                            batch["is_stale"].float().mean()
                        )

                    if isinstance(rollouts, VERRolloutStorage):
                        assert isinstance(batch["policy_version"], torch.Tensor)
                        record_min_mean_max(
                            (
                                rollouts.current_policy_version
                                - batch["policy_version"]
                            ).float(),
                            "policy_version_difference",
                        )

                    learner_metrics["action_loss_dagger_ce"].append(
                        (-action_log_probs_teacher).mean()
                    )

            profiling_wrapper.range_pop()  # PPO.update epoch

        # Add the overall PPO/DAGGER ratio to metrics
        if self.total_samples > 0:
            overall_ppo_ratio = self.ppo_usage_count / self.total_samples
            learner_metrics["overall_ppo_ratio"].append(overall_ppo_ratio)
            learner_metrics["overall_dagger_ratio"].append(1.0 - overall_ppo_ratio)
            print(f"PPO/DAGGER usage: {overall_ppo_ratio:.3f}/{1.0 - overall_ppo_ratio:.3f} (Total samples: {self.total_samples}, Frame count: {self.total_frames})")

        self._set_grads_to_none()

        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }

    def _evaluate_actions(self, *args, **kwargs):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(*args, **kwargs)

    def before_backward(self, loss: Tensor) -> Tensor:
        return loss

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> torch.Tensor:
        handles = []
        if torch.distributed.is_initialized():
            for p in self.non_ac_params:
                if p.grad is not None:
                    p.grad.data.detach().div_(torch.distributed.get_world_size())
                    handles.append(
                        torch.distributed.all_reduce(
                            p.grad.data.detach(), async_op=True
                        )
                    )

        grad_norm = nn.utils.clip_grad_norm_(
            self.actor_critic.policy_parameters(),
            self.max_grad_norm,
        )

        for v in self.actor_critic.aux_loss_parameters().values():
            nn.utils.clip_grad_norm_(v, self.max_grad_norm)

        [h.wait() for h in handles]

        return grad_norm

    def after_step(self) -> None:
        if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
            self.entropy_coef.project_into_bounds()


class DDPPO_custom(DecentralizedDistributedMixin, DAgger_PPO):
    pass
