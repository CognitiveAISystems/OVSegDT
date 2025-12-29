import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import HabitatSimActions

@registry.register_action_space_configuration(name="v0noisy")
class HabitatSimV0NoisyActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        # Default to LoCoBot parameters if not specified, but they should be in config
        if hasattr(self.config, "action_space_config_arguments") and hasattr(self.config.action_space_config_arguments, "NOISE_MODEL"):
            noisemodel_config = self.config.action_space_config_arguments.NOISE_MODEL
        else:
            # Fallback or error? Let's try to be safe
            class MockConfig:
                robot = "LoCoBot"
                CONTROLLER = "Proportional"
                NOISE_MULTIPLIER = 1.0
            noisemodel_config = MockConfig()

        return {
            HabitatSimActions.stop: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.move_forward: habitat_sim.ActionSpec(
                "pyrobot_noisy_move_forward",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.forward_step_size,
                    robot=noisemodel_config.robot,
                    controller=noisemodel_config.CONTROLLER,
                    noise_multiplier=noisemodel_config.NOISE_MULTIPLIER,
                ),
            ),
            HabitatSimActions.turn_left: habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_left",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.turn_angle,
                    robot=noisemodel_config.robot,
                    controller=noisemodel_config.CONTROLLER,
                    noise_multiplier=noisemodel_config.NOISE_MULTIPLIER,
                ),
            ),
            HabitatSimActions.turn_right: habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_right",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.turn_angle,
                    robot=noisemodel_config.robot,
                    controller=noisemodel_config.CONTROLLER,
                    noise_multiplier=noisemodel_config.NOISE_MULTIPLIER,
                ),
            ),
        }

