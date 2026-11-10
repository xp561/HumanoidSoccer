import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from soccer.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def load(self, path: str, load_optimizer: bool = True):
        infos = super().load(path, load_optimizer)
        env = self.env.unwrapped
        motion_cfg = getattr(getattr(env.cfg, "commands", None), "motion", None)
        if (
            getattr(motion_cfg, "train_init_position_curriculum", False)
            and isinstance(infos, dict)
            and "initial_position_control_steps" in infos
        ):
            motion = env.command_manager.get_term("motion")
            motion.initial_position_step_offset = int(infos["initial_position_control_steps"])
            self.env.reset()
        return infos

    def save(self, path: str, infos=None):
        """Save the model and the position curriculum's elapsed control steps."""
        env = self.env.unwrapped
        motion_cfg = getattr(getattr(env.cfg, "commands", None), "motion", None)
        if getattr(motion_cfg, "train_init_position_curriculum", False):
            motion = env.command_manager.get_term("motion")
            infos = dict(infos or {})
            infos["initial_position_control_steps"] = env.common_step_counter + getattr(
                motion, "initial_position_step_offset", 0
            )
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
