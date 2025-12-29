#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.action_embedding import ActionEmbedding
from habitat_baselines.rl.ppo.policy import Net

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        pos_weight: float
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    if True:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        weight = targets * pos_weight + (1 - targets)  # Higher weight for 1s
        loss = weight*loss
    else:
        loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        weight = targets * pos_weight + (1 - targets)  # Higher weight for 1s
        loss = weight*loss
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


@baseline_registry.register_auxiliary_loss(name="segm")
class SegmentationAuxiliary(nn.Module):
    r"""Implements Segmentation Loss
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        net: Net,
        loss_scale: float = 0.1,
    ):

        super().__init__()
        self.loss_scale = loss_scale
        self.num_points = 12544
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

    def forward(self, aux_loss_state, batch):

        gt_segmentation = batch['observations']['semantic'].permute(0, 3, 1, 2).float()
        #print(gt_segmentation.shape)
        # Compute sum along H and W dimensions to check for empty masks
        #mask_sums = gt_segmentation.sum(dim=(2, 3))
        
        # Get indices of non-empty masks (sum > 0)
        #non_empty_indices = mask_sums.squeeze(1) > 0

        #print(non_empty_indices)

        #gt_segmentation = gt_segmentation[non_empty_indices]
        #print(gt_segmentation.shape)
        #exit()
        
        #gt_segmentation = F.interpolate(gt_segmentation, size=(64, 64), mode='nearest')
        num_masks = int(gt_segmentation.shape[0])

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = aux_loss_state['reconstructed_image']
        #src_masks = src_masks[non_empty_indices]
        target_masks = gt_segmentation

        #print(src_masks.shape, target_masks.shape)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        dice =  dice_loss_jit(point_logits, point_labels, num_masks)
        bce = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, aux_loss_state['pos_weight'])

        loss = 1.0 * (dice + bce)
        #print(loss, dice, bce, torch.sum(gt_segmentation), torch.max(src_masks), torch.min(src_masks), src_masks.shape)
        del src_masks
        del target_masks
        return dict(
            loss=loss,
            non_zero_pixel_count=torch.sum(gt_segmentation),
            dice=dice,
            bce=bce,
        )

    def forward_new(self, aux_loss_state, batch):
        gt_segmentation = batch['observations']['semantic'].permute(0, 3, 1, 2).float()
        #print(gt_segmentation.shape)
        # Compute sum along H and W dimensions to check for empty masks
        mask_sums = gt_segmentation.sum(dim=(2, 3))
        
        # Get indices of non-empty masks (sum > 0)
        non_empty_indices = mask_sums.squeeze(1) > 0

        #if torch.sum(non_empty_indices) == 0:
        #    non_empty_indices = mask_sums.squeeze(1) == 0
        
        # Get indices of empty masks (sum == 0)
        empty_indices = mask_sums.squeeze(1) == 0

        new_non_empty_indices = torch.zeros_like(non_empty_indices, dtype=torch.bool)

        selected_indices = torch.randperm(empty_indices.numel())[:(int(torch.sum(non_empty_indices)) + 1)]

        new_non_empty_indices[selected_indices] = True
        new_non_empty_indices[non_empty_indices] = True
        
        # Randomly select 5% of the empty masks
        #random_empty_indices = torch.randint(empty_indices

        # non_empty_indices[torch.randint(0, empty_indices.size(0), (int(empty_indices.size(0)*0.05),))] = 1
        
        # non_empty_indices = non_empty_indices 

        #print(non_empty_indices)

        # non_empty_indices = torch.randint(0, real_image.size(0), (64,))
        #non_empty_indices[torch.randint(0, real_image.size(0), (64,))] = 1
        #if non_empty_indices.numel() > 64:
        #    selected_indices = torch.randperm(non_empty_indices.numel())[:64]  # Randomly pick 64
        #    new_non_empty_indices = torch.zeros_like(non_empty_indices, dtype=torch.bool)  # Create all False
        #    new_non_empty_indices[selected_indices] = True  # Set only the selected ones to True
        #else:
        #    new_non_empty_indices = non_empty_indices
        gt_segmentation = gt_segmentation[new_non_empty_indices]

        #print(non_empty_indices)

        #gt_segmentation = gt_segmentation[aux_loss_state["non_empty_indices"]]
        #print(gt_segmentation.shape)
        #exit()
        
        #gt_segmentation = F.interpolate(gt_segmentation, size=(64, 64), mode='nearest')
        num_masks = int(gt_segmentation.shape[0])

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = aux_loss_state['reconstructed_image']
        src_masks = src_masks[new_non_empty_indices]
        target_masks = gt_segmentation

        #print(src_masks.shape, target_masks.shape)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        dice =  dice_loss_jit(point_logits, point_labels, num_masks)
        bce = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, aux_loss_state['pos_weight'])

        vqgan = False
        if vqgan:
            g_loss = (0.25*(dice + bce + aux_loss_state['vq_loss']) + aux_loss_state['orthogonal_loss'])
            print(g_loss, dice, bce, aux_loss_state['vq_loss'], aux_loss_state['orthogonal_loss'], torch.max(src_masks), torch.min(src_masks), src_masks.shape)
        else:
            g_loss = dice + bce

            #print(g_loss, dice, bce, torch.max(src_masks), torch.min(src_masks), src_masks.shape)
        
        return dict(
            loss=g_loss,
            dice=dice,
            bce=bce,
            #vq_loss=aux_loss_state['vq_loss'],
            #d_loss=aux_loss_state['d_loss'],
            max_segm_pred=torch.max(src_masks),
            min_segm_pred=torch.min(src_masks)
        )

    def reconstruction_loss(self, real, fake):
        return F.mse_loss(fake, real)
        