from __future__ import annotations

import os
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np
import torch


@dataclass
class _CameraStream:
    label: str
    camera_path: str
    render_product_path: str
    annotator: object


class InProcessCurriculumVideoRecorder:
    """Record per-curriculum videos from the active training simulation."""

    def __init__(
        self,
        env,
        log_dir: str,
        *,
        interval_steps: int,
        video_length: int,
        resolution: tuple[int, int] = (640, 360),
        camera_target_height: float = 0.35,
        camera_distance_scale: float = 0.60,
        camera_height_scale: float = 0.42,
        force_terrain_level: int | None = None,
        logger=None,
    ) -> None:
        if interval_steps <= 0:
            raise ValueError(f"interval_steps must be positive, got {interval_steps}.")
        if video_length <= 0:
            raise ValueError(f"video_length must be positive, got {video_length}.")

        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.log_dir = log_dir
        self.interval_steps = interval_steps
        self.video_length = video_length
        self.resolution = resolution
        self.camera_target_height = camera_target_height
        self.camera_distance_scale = camera_distance_scale
        self.camera_height_scale = camera_height_scale
        self.tile_size = self._terrain_tile_size()
        self.force_terrain_level = force_terrain_level
        self.logger = logger

        self.global_step = -1
        self.capture_step = 0
        self.capture_iteration = 0
        self.disabled = False
        self.active_capture = False
        self.streams: dict[str, _CameraStream] = {}
        self.active_writers: dict[str, object] = {}
        self.active_env_ids: dict[str, int] = {}
        self.active_paths: dict[str, str] = {}

    def on_step(self, learning_iteration: int) -> None:
        if self.disabled:
            return
        try:
            self._on_step_impl(learning_iteration)
        except Exception as exc:
            self.disabled = True
            self._close_writers()
            print(f"[WARN]: Disabled in-process curriculum video recorder after error: {exc}")

    def _on_step_impl(self, learning_iteration: int) -> None:
        self.global_step += 1

        if not self.active_capture and self.global_step % self.interval_steps == 0:
            self._start_capture(learning_iteration)

        if not self.active_capture:
            return

        self._capture_frame()
        self.capture_step += 1
        if self.capture_step >= self.video_length:
            self._finish_capture()

    def _start_capture(self, learning_iteration: int) -> None:
        selected_envs = self._select_envs()
        if not selected_envs:
            return

        self.capture_iteration = learning_iteration
        self.capture_step = 0
        self.active_capture = True
        self.active_env_ids = selected_envs
        self.active_paths.clear()
        self.active_writers.clear()

        capture_dir = os.path.join(self.log_dir, "videos", "curriculum", f"step_{self.global_step:08d}")
        os.makedirs(capture_dir, exist_ok=True)
        for label in selected_envs:
            stream = self._get_or_create_stream(label)
            video_path = os.path.join(capture_dir, f"{label}.mp4")
            self.active_paths[label] = video_path
            self.active_writers[label] = imageio.get_writer(video_path, fps=self._fps(), macro_block_size=1)
            self._update_camera(stream, selected_envs[label])

        print(f"[INFO]: Recording curriculum videos at step {self.global_step}: {selected_envs}")

    def _finish_capture(self) -> None:
        self._close_writers()
        self._log_wandb_videos()
        print(f"[INFO]: Finished curriculum videos for step {self.global_step - self.capture_step + 1}.")
        self.active_capture = False
        self.active_env_ids.clear()
        self.active_paths.clear()
        self.capture_step = 0

    def _close_writers(self) -> None:
        for writer in self.active_writers.values():
            writer.close()
        self.active_writers.clear()

    def _capture_frame(self) -> None:
        for label, env_id in self.active_env_ids.items():
            self._update_camera(self.streams[label], env_id)

        self.base_env.sim.render()

        for label, writer in self.active_writers.items():
            frame = np.asarray(self.streams[label].annotator.get_data())
            if frame.size == 0:
                continue
            if frame.shape[-1] > 3:
                frame = frame[:, :, :3]
            writer.append_data(frame)

    def _select_envs(self) -> dict[str, int]:
        terrain = getattr(getattr(self.base_env, "scene", None), "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is None:
            return {"default": 0}

        if self.force_terrain_level is not None:
            env_ids = (terrain_levels == self.force_terrain_level).nonzero(as_tuple=False).flatten()
            if env_ids.numel() == 0:
                return {}
            return {f"level_{self.force_terrain_level}": int(env_ids[0].item())}

        selected: dict[str, int] = {}
        for level in torch.unique(terrain_levels.detach()).sort().values:
            level_int = int(level.item())
            env_ids = (terrain_levels == level_int).nonzero(as_tuple=False).flatten()
            if env_ids.numel() > 0:
                selected[f"level_{level_int}"] = int(env_ids[0].item())
        return selected

    def _get_or_create_stream(self, label: str) -> _CameraStream:
        if label in self.streams:
            return self.streams[label]

        import isaaclab.sim as sim_utils
        import omni.replicator.core as rep
        from pxr import UsdGeom

        camera_path = f"/World/CurriculumVideo/{label}_camera"
        cam_prim = sim_utils.create_prim(camera_path, prim_type="Camera")
        camera = UsdGeom.Camera(cam_prim)
        camera.CreateFocalLengthAttr().Set(22.0)
        camera.CreateClippingRangeAttr().Set((0.1, 1000.0))
        render_product_path = rep.create.render_product(camera_path, resolution=self.resolution)
        if not isinstance(render_product_path, str):
            render_product_path = render_product_path.path
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach(render_product_path)

        stream = _CameraStream(
            label=label,
            camera_path=camera_path,
            render_product_path=render_product_path,
            annotator=annotator,
        )
        self.streams[label] = stream
        return stream

    def _update_camera(self, stream: _CameraStream, env_id: int) -> None:
        target = self._tile_center(env_id)
        span = max(self.tile_size)
        eye = target + torch.tensor(
            (
                -self.camera_distance_scale * span,
                -self.camera_distance_scale * span,
                self.camera_height_scale * span,
            ),
            dtype=torch.float32,
            device=self.base_env.device,
        )
        self.base_env.sim.set_camera_view(
            eye=tuple(float(x) for x in eye.detach().cpu().tolist()),
            target=tuple(float(x) for x in target.detach().cpu().tolist()),
            camera_prim_path=stream.camera_path,
        )

    def _terrain_tile_size(self) -> tuple[float, float]:
        terrain = getattr(getattr(self.base_env, "scene", None), "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        size = getattr(generator, "size", None)
        if size is None:
            return (8.0, 8.0)
        return (float(size[0]), float(size[1]))

    def _tile_center(self, env_id: int) -> torch.Tensor:
        target = self.base_env.scene.env_origins[env_id].clone()
        target[2] += self.camera_target_height
        return target

    def _fps(self) -> int:
        step_dt = float(getattr(self.base_env, "step_dt", 1.0 / 50.0))
        return max(1, int(round(1.0 / step_dt)))

    def _log_wandb_videos(self) -> None:
        if self.logger is None or getattr(self.logger, "disable_logs", False):
            return
        if getattr(self.logger, "logger_type", "").lower() != "wandb":
            return
        try:
            import wandb

            for label, path in self.active_paths.items():
                wandb.log(
                    {f"Video/curriculum/{label}": wandb.Video(path, fps=self._fps(), format="mp4")},
                    step=self.capture_iteration,
                )
        except Exception as exc:
            print(f"[WARN]: Could not log curriculum videos to wandb: {exc}")
