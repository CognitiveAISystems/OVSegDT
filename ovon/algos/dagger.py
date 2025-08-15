#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.optim as optim
from gym import spaces
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.config.default_structured_configs import PolicyConfig
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet  # noqa: F401.
from habitat_baselines.rl.ppo import PPO, Policy
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import (
    CategoricalNet,
    GaussianNet,
    LagrangeInequalityCoefficient,
    inference_mode,
)
from omegaconf import DictConfig
from torch import nn

from ovon.obs_transformers.relabel_teacher_actions import RelabelTeacherActions

EPS_PPO = 1e-5


class DAgger(PPO):
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
        nn.Module.__init__(self)

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.inflection_weight = 2.11

        if hasattr(self.actor_critic, "net"):
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

        params = list(filter(lambda p: p.requires_grad, self.parameters()))

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

        self.segm_step = 0

    def update(self, rollouts: RolloutStorage) -> Dict[str, float]:
        learner_metrics = collections.defaultdict(list)
        self.segm_step += 1
        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("DAgger.update epoch")
            data_generator = rollouts.recurrent_generator(None, self.num_mini_batch)

            for _bid, batch in enumerate(data_generator):
                self._set_grads_to_none()
                # TODO: see if casting from torch.uint8 to long is necessary
                teacher_actions = batch["observations"][
                    RelabelTeacherActions.TEACHER_LABEL
                ].type(torch.long)
                log_probs, aux_loss_res = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["prev_segm_masks"],
                    batch["masks"],
                    teacher_actions,
                    batch["rnn_build_seq_info"],
                )
                if "inflection" in batch["observations"]:
                    # Wherever inflections_batch is 1, change it to self.inflection
                    # weight, otherwise change the value (which should be 0) to 1
                    inflection_weights = torch.where(
                        batch["observations"]["inflection"] == 1,
                        torch.ones_like(batch["observations"]["inflection"])
                        * self.inflection_weight,
                        torch.ones_like(batch["observations"]["inflection"]),
                    )
                    loss = -(
                        (log_probs * inflection_weights).sum(0)
                        / inflection_weights.sum(0)
                    ).mean()
                else:
                    loss = -log_probs.mean()

                if "is_coeffs" in batch:
                    assert isinstance(batch["is_coeffs"], torch.Tensor)
                    ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)

                    def mean_fn(t):
                        return torch.mean(ver_is_coeffs * t)

                else:
                    mean_fn = torch.mean

                total_loss = mean_fn(loss).sum()

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
                    total_loss += segmentation_loss['loss']
                    with inference_mode():
                        learner_metrics["aux_segmentation"].append(segmentation_loss['loss'])
                        learner_metrics["aux_dice"].append(segmentation_loss['dice'])
                        learner_metrics["aux_bce"].append(segmentation_loss['bce'])
                    self.segm_step = 0
                else:
                    if self.actor_critic.net.segm_loss_enabled:
                        total_loss += aux_loss_res['loss']
                        with inference_mode():
                            learner_metrics["aux_segmentation"].append(aux_loss_res['loss'])
                            learner_metrics["aux_dice"].append(aux_loss_res['dice'])
                            learner_metrics["aux_bce"].append(aux_loss_res['bce'])
                total_loss = self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                grad_norm = self.before_step()
                self.optimizer.step()
                self.after_step()

                with inference_mode():
                    if "is_coeffs" in batch:
                        record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
                    learner_metrics["loss"].append(loss)
                    learner_metrics["grad_norm"].append(grad_norm)

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

            profiling_wrapper.range_pop()  # PPO.update epoch

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
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of
        calling that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(*args, **kwargs)


class DDPDAgger(DecentralizedDistributedMixin, DAgger):
    pass


class DAggerPolicyMixin:
    """Avoids computing value or action_log_probs, which are RL-only, and
    .evaluate_actions() will be overridden to produce the correct gradients."""

    action_distribution: Union[CategoricalNet, GaussianNet]
    critic: nn.Module
    net: nn.Module
    action_distribution_type: str

    def __init__(self, teacher_forcing: bool, *args, **kwargs):
        self.teacher_forcing = teacher_forcing
        super().__init__(*args, **kwargs)

    @property
    def policy_components(self):
        """Same except critic weights are not included."""
        return self.net, self.action_distribution

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        prev_segm_masks,
        masks,
        deterministic=False,
    ):
        if not hasattr(self, "action_distribution"):
            # For the dummy policy class that has no neural net
            return super().act(  # type: ignore
                observations, rnn_hidden_states, prev_actions, masks
            )

        if self.teacher_forcing and "teacher_label" in observations:
            return self.cheat(observations, rnn_hidden_states)

        """Skips computing values and action_log_probs, which are RL-only."""
        features, rnn_hidden_states, aux_loss_states = self.net(
            observations, rnn_hidden_states, prev_actions, prev_segm_masks, masks
        )
        distribution = self.action_distribution(features)

        with torch.no_grad():
            if deterministic:
                if self.action_distribution_type == "categorical":
                    action = distribution.mode()
                elif self.action_distribution_type == "gaussian":
                    action = distribution.mean
                else:
                    raise NotImplementedError(
                        "Distribution type {} is not supported".format(
                            self.action_distribution_type
                        )
                    )
            else:
                action = distribution.sample()
        n = action.shape[0]
        value = torch.zeros(n, 1, device=action.device)
        action_log_probs = torch.zeros(n, 1, device=action.device)
        return value, action, action_log_probs, rnn_hidden_states, aux_loss_states['prev_segm_masks']

    @staticmethod
    def cheat(observations, rnn_hidden_states):
        action = observations["teacher_label"].long()

        num_envs = observations["teacher_label"].shape[0]
        device = observations["teacher_label"].device

        action_log_probs = torch.zeros(num_envs, 1).to(device)
        value = torch.zeros(num_envs, 1).to(device)

        return value, action, action_log_probs, rnn_hidden_states

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        prev_segm_masks,
        masks,
        action,
        rnn_build_seq_info,
    ):
        """Given a state and action, computes the policy's action distribution for that
        state and then returns the log probability of the given action under this
        distribution."""
        features, _, aux_loss_state = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            prev_segm_masks,
            masks,
            rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        log_probs = distribution.log_probs(action)

        if self.net.segm_loss_enabled:
            aux_loss_res = self.net.segm_loss(aux_loss_state, {'observations': observations})
        else:
            aux_loss_res = None
        return log_probs, aux_loss_res

    def load_state_dict(self, state_dict):

        prefix = "net."
        u, v = self.net.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            },
            strict=False
        )
        prefix = "action_distribution."
        self.action_distribution.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            },
            strict=True
        )
        prefix = "critic."
        self.critic.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            },
            strict=True
        )

        return u


@baseline_registry.register_policy
class DAggerPolicy(Policy):
    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        original_cls = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.original_name
        )
        # fmt: off
        class MixedPolicy(DAggerPolicyMixin, original_cls): pass  # noqa
        # fmt: on
        return MixedPolicy.from_config(
            config,
            observation_space,
            action_space,
            teacher_forcing=config.habitat_baselines.rl.policy.teacher_forcing,
            **kwargs,
        )



@dataclass
class DAggerPolicyConfig(PolicyConfig):
    original_name: str = ""
    teacher_forcing: bool = False
