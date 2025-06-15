
import gym
import numpy as np
import pybullet as p
import pybullet_data
import time

class ScaraPickPlaceEnv(gym.Env):
    def __init__(self, render=False):
        super(ScaraPickPlaceEnv, self).__init__()
        self.render_mode = render
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.time_step = 1. / 240.
        self.max_steps = 200
        self.step_counter = 0

        self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.robot = None
        self.cube = None
        self.joints = []
        self.ee_idx = -1
        self.attached = False
        self.cube_constraint = None
        self.goal_pos = np.array([0.5, 0.5, 0.1])  # fixed goal for now

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF("scara_orange.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        self.joints = [i for i in range(p.getNumJoints(self.robot))
                       if p.getJointInfo(self.robot, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]
        self.ee_idx = p.getNumJoints(self.robot) - 1

        self.cube_pos = np.array([
            np.random.uniform(0.2, 0.3),
            np.random.uniform(0.2, 0.3),
            0.1
        ])
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05]*3, rgbaColor=[1, 0, 0, 1])
        self.cube = p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=cube_visual,
                                      basePosition=self.cube_pos.tolist())

        self.attached = False
        self.cube_constraint = None
        self.step_counter = 0
        return self._get_obs()

    def _get_obs(self):
        ee_state = p.getLinkState(self.robot, self.ee_idx)[0]
        cube_state = p.getBasePositionAndOrientation(self.cube)[0]
        return np.array(list(ee_state[:3]) + list(cube_state[:3]), dtype=np.float32)

    def step(self, action):
        ee_pos = list(p.getLinkState(self.robot, self.ee_idx)[0])
        target_pos = np.array(ee_pos) + np.clip(action, -0.05, 0.05)

        ik_joints = p.calculateInverseKinematics(self.robot, self.ee_idx, target_pos.tolist())
        for i, j in enumerate(self.joints):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=ik_joints[i], force=10)

        for _ in range(10):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(self.time_step)

        self._try_grasp()

        obs = self._get_obs()
        reward = -np.linalg.norm(obs[:2] - obs[3:5])
        done = self.step_counter >= self.max_steps

        self.step_counter += 1
        return obs, reward, done, {}

    def _try_grasp(self):
        if self.attached:
            return
        ee_pos = p.getLinkState(self.robot, self.ee_idx)[0]
        cube_pos = p.getBasePositionAndOrientation(self.cube)[0]
        dist = np.linalg.norm(np.array(ee_pos[:2]) - np.array(cube_pos[:2]))
        if dist < 0.05 and abs(ee_pos[2] - cube_pos[2]) < 0.05:
            self.cube_constraint = p.createConstraint(
                parentBodyUniqueId=self.robot,
                parentLinkIndex=self.ee_idx,
                childBodyUniqueId=self.cube,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            self.attached = True

    def check_success(self):
        cube_pos = p.getBasePositionAndOrientation(self.cube)[0]
        return np.linalg.norm(np.array(cube_pos[:2]) - self.goal_pos[:2]) < 0.05

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()
