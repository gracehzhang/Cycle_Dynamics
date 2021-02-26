from pybullet_envs.minitaur.envs.minitaur_trotting_env import MinitaurTrottingEnv
import numpy as np

class AccurateMinitaurTrottingEnv(MinitaurTrottingEnv):
    def render(self, mode="rgb_array", close=False, height=360, width=480, **kwargs):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.minitaur.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                    aspect=float(width) /
                                                                    height,
                                                                    nearVal=0.1,
                                                                    farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=width,
            height=height,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    