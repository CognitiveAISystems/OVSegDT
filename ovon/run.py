#!/usr/bin/env python3

import argparse
import glob
import os
import os.path as osp

import torch
from habitat import get_config
from habitat.config import read_write
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from omegaconf import OmegaConf  # keep this import for print debugging

from ovon.config import ClipObjectGoalSensorConfig, HabitatConfigPlugin


def register_plugins():
    register_hydra_plugin(HabitatConfigPlugin)


def main():
    """Builds upon the habitat_baselines.run.main() function to add more flags
    for convenience."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run-type",
        "-r",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        "-e",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Saves files to $JUNK directory and ignores resume state.",
    )
    parser.add_argument(
        "--single-env",
        "-s",
        action="store_true",
        help="Sets num_environments=1.",
    )
    parser.add_argument(
        "--debug-datapath",
        "-p",
        action="store_true",
        help="Uses faster-to-load $OVON_DEBUG_DATAPATH episode dataset for debugging.",
    )
    parser.add_argument(
        "--blind",
        "-b",
        action="store_true",
        help="If set, no cameras will be used.",
    )
    parser.add_argument(
        "--checkpoint-config",
        "-c",
        action="store_true",
        help=(
            "If set, checkpoint's config will be used, but overrides WILL be "
            "applied. Does nothing when training; meant for using ckpt config + "
            "overrides for eval."
        ),
    )
    parser.add_argument(
        "--text-goals",
        "-t",
        action="store_true",
        help="If set, only CLIP text goals will be used for evaluation.",
    )
    parser.add_argument(
        "--eval-analysis",
        "-v",
        action="store_true",
        help="If set, add semantic sensor for evaluation.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # Register custom hydra plugin
    register_plugins()

    config = get_config(args.exp_config, args.opts)

    if args.run_type == "eval" and args.checkpoint_config:
        config = merge_config(config, args.opts)

    with read_write(config):
        edit_config(config, args)

    if args.run_type == "eval" and os.environ.get("OVON_LIMIT_EVAL", "0") == "1":
        # Move the checkpoint file and the script if the checkpoint wasn't
        # trained for long enough or if it was trained for too long
        ckpt_path = config.habitat_baselines.eval_ckpt_path_dir
        assert osp.isfile(ckpt_path)
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        step_id = ckpt_dict["extra_state"]["step"]
        if not 1e8 < step_id < 5e8:
            # Create an overflow folder if it doesn't exist
            overflow_dir = osp.join(osp.dirname(ckpt_path), "overflow")
            try:
                os.makedirs(overflow_dir, exist_ok=True)
            except Exception as e:
                print(f"Could not create overflow directory {overflow_dir}: {e}")
            assert osp.isdir(overflow_dir)
            # Move the checkpoint file
            try:
                os.rename(
                    ckpt_path,
                    osp.join(overflow_dir, osp.basename(ckpt_path)),
                )
                print("Moved checkpoint file to overflow directory")
            except FileNotFoundError:
                print(
                    f"Checkpoint file {ckpt_path} not found! Skipping checkpoint move."
                )
            # End the script
            return

    execute_exp(config, args.run_type)


def merge_config(config, opts):
    """There might be a better way to do this with Hydra... do I know it? No.
    1. Locate a checkpoint using the config's eval checkpoint path
    2. Load that checkpoint's config to replicate training config
    3. Save this config to a temporary file
    4. Use the path to the temporary file and the given override opts

    This is the only way to add overrides in eval that also use whatever
    overrides were used in training.
    """
    # 1. Locate a checkpoint using the config
    checkpoint_path = config.habitat_baselines.eval_ckpt_path_dir
    if osp.isdir(checkpoint_path):
        ckpt_files = glob.glob(osp.join(checkpoint_path, "*.pth"))
        assert len(ckpt_files) > 0, f"No checkpoints found in {checkpoint_path}!"
        checkpoint_path = ckpt_files[0]
    elif not osp.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} is not a file!")

    # 2. Load the config from the checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = ckpt["config"]

    # 3. Save the given config to a temporary file
    randstr = str(torch.randint(0, 100000, (1,)).item())
    tmp_config_path = f"/tmp/ovon_config_{randstr}.yaml"
    OmegaConf.save(ckpt_config, tmp_config_path)

    # 4. Use the path to the temporary file as the config path and use the
    # given opts to override the config
    config = get_config(tmp_config_path, opts)
    os.remove(tmp_config_path)

    # Set load_resume_state_config to False so we don't load the checkpoint's
    # config again and lose the overrides
    with read_write(config):
        config.habitat_baselines.load_resume_state_config = False
        config.habitat.dataset.data_path = "data/datasets/ovon/hm3d/v1/val_seen/val_seen.json.gz"

    return config


def edit_config(config, args):
    if args.debug:
        assert osp.isdir(os.environ["JUNK"]), (
            "Environment variable directory $JUNK does not exist "
            f"(Current value: {os.environ['JUNK']})"
        )

        # Remove resume state in junk if training, so we don't resume from it
        resume_state_path = osp.join(os.environ["JUNK"], ".habitat-resume-state.pth")
        if args.run_type == "train" and osp.isfile(resume_state_path):
            print(
                "Removing junk resume state file:",
                osp.abspath(resume_state_path),
            )
            os.remove(resume_state_path)

        config.habitat_baselines.tensorboard_dir = os.environ["JUNK"]
        config.habitat_baselines.video_dir = os.environ["JUNK"]
        config.habitat_baselines.checkpoint_folder = os.environ["JUNK"]
        config.habitat_baselines.log_file = osp.join(os.environ["JUNK"], "junk.log")
        config.habitat_baselines.load_resume_state_config = False

    if args.debug_datapath:

        if args.run_type == "train":
            config.habitat.dataset.content_scenes = [
                "1S7LAXRdDqK", "741Fdj7NLF9", "DsEJeNPcZtE", "HfMobPm86Xn", "NPHxDe6VeCc", "U3oQjwTuMX8", 
                "XiJhRLvpKpX", "ceJTwFNjqCt", "h6nwVLpAKQz", "nGhNxKrgBPb", "u5atqC7vRCY", "zUG6FL9TYeR",
                "1UnKg1rAb8A", "77mMEyxhs44", "E1NrAhMoqvB", "HkseAnWCgqk", "NtnvZSMK3en", "URjpCob8MGw", 
                "YHmAkqgwe2p", "dQrLTxHvLXU", "hWDDQnSDMXb", "nS8T59Aw3sf", "u9rPN5cHWBg", "zepmXAdrpjR",
                "1xGrZPxG1Hz", "8B43pG641ff", "ENiCjXWB6aQ", "HxmXPBbFCkH", "PE6kVEtrxtj", "UuwwmrTsfBN", 
                "YJDUB7hWg9h", "erXNfWVjqZ8", "iKFn6fzyRqs", "oEPjPNSPmzL", "v7DzfFFEpsD",
                "226REUyJh2K", "8wJuSPJ9FXG", "FRQ75PjD278", "JNiWU5TZLtt", "PPTLa8SkUfo", "VSxVP19Cdyw", 
                "YMNvYDhK8mB", "fK2vEV32Lag", "iLDo95ZbDJq", "oPj9qMxrDEa", "vDfkYo5VqEQ",
                "2Pc8W48bu21", "92vYG1q49FY", "FnDDfrBZPhh", "Jfyvj3xn2aJ", "QN2dRqwd84J", "VoVGtfYrpuQ", 
                "YY8rqV6L6rf", "fRZhp6vWGw7", "iePHCSf119p", "oStKKWkQ1id", "vLpv2VX547B",
                "3CBBjsNkhqW", "9h5JJxM6E5S", "GGBvSFddQgs", "JptJPosx1Z6", "QVAA6zecMHu", "W16Bm4ysK8v", 
                "YmWinf3mhb5", "fxbzYAGkrtm", "iigzG1rtanx", "oahi4u45xMf", "w8GiikYuFRk",
                "3XYAD64HpDr", "ACZZiU6BXLz", "GPyDUnjwZQy", "KjZrPggnHm8", "R9fYpvCUkV7", "W9YAR9qcuvN", 
                "Z2DQddYp1fn", "g7hUFVNac26", "ixTj1aTMup2", "ooq3SnvC79d", "wPLokgvCnuk",
                "4vwGX7U38Ux", "CQWES1bawee", "GTV2Y73Sn5t", "L5QEsaVqwrY", "RTV2n6fXB2w", "WhNyDTnd9g5", 
                "ZNanfzgCdm3", "g8Xrdbe9fir", "j2EJhFEQGCL", "pcpn6mFqFCg", "wsAYBFtQaL7",
                "5Kw4nGdqYtS", "CthA7sQNTPK", "GsQBY83r3hb", "LVgQNuK8vtv", "RaYrxWt5pR1", "Wo6kuutE9i7", 
                "aRKASs4e8j1", "gQ3xxshDiCz", "j6fHrce9pHR", "qZ4B7U6XE5Y", "xAHnY3QzFUN",
                "5biL7VEkByM", "DBBESbk4Y3k", "GtM3JtRvvvR", "LcAd9dhvVwh", "S7uMvxjBVZq", "X6Pct1msZv5", 
                "b3WpMbPFB6q", "gQgtJ9Stk5s", "kA2nG18hCAr", "qgZhhx1MpTi", "xWvSkKiWQpC",
                "6HRFAUDqpTb", "DNWbUAJYsPy", "H8rQCnvBgo6", "MVVzj944atG", "SgkmkWjjmDJ", "XVSZJAtHKdi", 
                "bB6nKqfsb1z", "ggNAcMh8JPT", "kJxT5qssH4H", "qk9eeNeR4vw", "xgLmjqzoAzF",
                "6YtDG3FhNvx", "DoSbsoo4EAg", "HZ2iMMBsBQ9", "NEVASPhcrxR", "TSJmdttd2GV", "XYyR54sxe6b", 
                "bHKTDQFJxTw", "gjhYih4upQ9", "mt9H8KcxRKD", "qz3829g1Lzf", "yHLr6bvWsVm",
                "6imZUJGRUq4", "DqJKU7YU7dA", "HeSYRw7eMtG", "NGyoyh91xXJ", "TYDavTf8oyy", "XfUxBGTFQQb", 
                "bdp1XNEdvmW", "gmuS7Wgsbrx", "nACV8wLu1u5", "sX9xad6ULKc", "yX5efd48dLf"
            ]
            # config.habitat.dataset.content_scenes = ["gmuS7Wgsbrx", "bdp1XNEdvmW", "sX9xad6ULKc", "yX5efd48dLf"]
            # config.habitat.simulator.habitat_sim_v0.gpu_device_id = 1
        else:
            config.habitat.dataset.content_scenes = [
                '4ok3usBNeis', 
                'DYehNKdT76V', 'Nfvxx8J5NCo', 'bCPU9suPUw9', 'mL8ThkuaVTM', 'svBbv1Pavdk',
                '5cdEh9F2hJL', 'Dd4bFSTQ8gi', 'QaLdnwvtxbs', 'bxsVRursffK', 'mv2HUxq3B53', 'wcojb4TFT35',
                '6s7QHgap2fW', 'GLAQ4DNUx5U', 'TEEsavR23oF', 'cvZr5TUy5C5', 'p53SfW6mjZe', 'y9hTuugGdiq',
                '7MXmsvcQjpJ', 'HY1NcmCgn3n', 'VBzV5z6i1WS', 'eF36g7L6Z9M', 'q3zU7Yy5E5s', 'yr17PDCnDDW',
                'BAbdmeyTvMZ', 'LT9Jq6dN3Ea', 'XB4GS9ShBRE', 'h1zeeAwLh9Z', 'q5QZSEeHe5g', 'ziup5kvtCCR',
                'CrMo8WxCyVb', 'MHPLjHsuG27', 'a8BtkwhxdRV', 'k1cupFYWXJ6', 'qyAac8rV8Zk', 'zt1RVoi7PcG'
            ]
            #config.habitat.dataset.content_scenes = ['4ok3usBNeis']



    if args.single_env:
        config.habitat_baselines.num_environments = 1

    # Remove all cameras if running blind (e.g., evaluating frontier explorer)
    if args.blind:
        for k in ["depth_sensor", "rgb_sensor"]:
            if k in config.habitat.simulator.agents.main_agent.sim_sensors:
                config.habitat.simulator.agents.main_agent.sim_sensors.pop(k)
        from habitat.config.default_structured_configs import (
            HabitatSimDepthSensorConfig,
        )

        # Camera required to load in a scene; use dummy 1x1 depth camera
        config.habitat.simulator.agents.main_agent.sim_sensors.update(
            {"depth_sensor": HabitatSimDepthSensorConfig(height=1, width=1)}
        )
        if hasattr(config.habitat_baselines.rl.policy, "obs_transforms"):
            config.habitat_baselines.rl.policy.obs_transforms = {}

    if args.eval_analysis:
        from habitat.config.default_structured_configs import (
            HabitatSimSemanticSensorConfig,
        )

        config.habitat.simulator.agents.main_agent.sim_sensors.update(
            {"semantic_sensor": HabitatSimSemanticSensorConfig(height=640, width=360)}
        )

    if (
        args.text_goals
        and args.run_type == "eval"
        and hasattr(config.habitat.task.lab_sensors, "clip_imagegoal_sensor")
    ):
        config.habitat.task.lab_sensors.pop("clip_imagegoal_sensor")
        if hasattr(config.habitat.task.lab_sensors, "clip_goal_selector_sensor"):
            config.habitat.task.lab_sensors.pop("clip_goal_selector_sensor")
        if not hasattr(config.habitat.task.lab_sensors, "clip_objectgoal_sensor"):
            config.habitat.task.lab_sensors.update(
                {"clip_objectgoal_sensor": ClipObjectGoalSensorConfig()}
            )

    if args.run_type == "train":
        for measure_name in ["frontier_exploration_map", "top_down_map"]:
            if hasattr(config.habitat.task.measurements, measure_name):
                print(
                    f"[run.py]: Removing {measure_name} measurement from "
                    "config to speed up training."
                )
                config.habitat.task.measurements.pop(measure_name)
    elif len(config.habitat_baselines.eval.video_option) == 0:
        # For eval, remove the objnav_explorer teacher
        if hasattr(config.habitat.task.lab_sensors, "objnav_explorer"):
            print("[run.py]: Removing objnav_explorer sensor from config for eval.")
            config.habitat.task.lab_sensors.pop("objnav_explorer")

        # Remove relabeler too
        if hasattr(
            config.habitat_baselines.rl.policy.obs_transforms, "relabel_teacher_actions"
        ):
            print("[run.py]: Removing relabel_teacher_actions from config for eval.")
            config.habitat_baselines.rl.policy.obs_transforms.pop(
                "relabel_teacher_actions"
            )


if __name__ == "__main__":
    main()
