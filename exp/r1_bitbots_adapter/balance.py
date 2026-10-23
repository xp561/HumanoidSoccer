"""Simulation-only feedback planner; produces ordinary fixed-gain PD targets."""

import mujoco
import numpy as np


class BalancePlanner:
    def __init__(self, plant):
        self.plant = plant
        self.robot = r = plant.robot
        self.feet = sorted(r.foot_bodies)
        self.ids = np.asarray(plant.meta["action_joint_ids"])
        self.av = r.dof_ids[self.ids]
        self.aq = r.qpos_ids[self.ids]
        self.kp = np.asarray(plant.meta["stiffness"])[self.ids]
        self.kd = np.asarray(plant.meta["damping"])[self.ids]
        self.effort = np.asarray(plant.meta["effort_limits"])[self.ids]
        self.bounds = []
        for foot in self.feet:
            vertices = []
            for gid in np.flatnonzero(r.model.geom_bodyid == foot):
                mid = r.model.geom_dataid[gid]
                if r.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or r.model.geom_contype[gid] == 0:
                    continue
                start, count = r.model.mesh_vertadr[mid], r.model.mesh_vertnum[mid]
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, r.model.geom_quat[gid])
                vertices.append(
                    r.model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T + r.model.geom_pos[gid]
                )
            points = np.concatenate(vertices)
            self.bounds.append(np.stack([points.min(0), points.max(0)]))
        self.capture()

    @staticmethod
    def upright(quat):
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    def capture(self):
        r = self.robot
        self.positions = r.data.xpos[self.feet].copy()
        self.rotations = [r.data.xquat[b].copy() for b in self.feet]
        bottoms = []
        for i, bid in enumerate(self.feet):
            bounds = self.bounds[i]
            corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
            world = corners @ r.data.xmat[bid].reshape(3, 3).T + r.data.xpos[bid]
            bottoms.append(world[:, 2].min())
            self.positions[i, 2] -= bottoms[-1]
        self.height = float(r.data.qpos[r.root_q + 2] - np.mean(bottoms))
        self.orientation = r.data.qpos[r.root_q + 3 : r.root_q + 7].copy()

    def target(self, posture):
        r = self.robot
        m, d = r.model, r.data
        posture = np.asarray(posture, float)
        if posture.shape != (len(self.av),) or not np.isfinite(posture).all():
            raise ValueError("Invalid balance posture")
        matrices, goals = [], []
        base = np.zeros((6, m.nv))
        base[:, r.root_v : r.root_v + 6] = np.eye(6)
        rotation = np.empty(3)
        mujoco.mju_subQuat(rotation, self.orientation, d.qpos[r.root_q + 3 : r.root_q + 7])
        support_centers = [
            self.positions[i] + d.xmat[b].reshape(3, 3) @ np.mean(self.bounds[i], axis=0)
            for i, b in enumerate(self.feet)
        ]
        center = np.mean(support_centers, axis=0)
        error_xy = center[:2] - d.subtree_com[r.root_id, :2]
        desired = np.r_[
            40 * error_xy - 12 * d.qvel[r.root_v : r.root_v + 2],
            100 * (self.height - d.qpos[r.root_q + 2]) - 12 * d.qvel[r.root_v + 2],
            40 * rotation - 12 * d.qvel[r.root_v + 3 : r.root_v + 6],
        ]
        matrices.append(10 * base)
        goals.append(10 * desired)
        support_jacobians = []
        for i, bid in enumerate(self.feet):
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, bid)
            jac = np.concatenate([jp, jr])
            error = np.empty(3)
            mujoco.mju_subQuat(error, self.rotations[i], d.xquat[bid])
            acceleration = np.r_[80 * (self.positions[i] - d.xpos[bid]), 80 * error] - 18 * (jac @ d.qvel)
            matrices.append(20 * jac)
            goals.append(20 * acceleration)
            bounds = self.bounds[i]
            middle, extent = np.mean(bounds, axis=0), 0.35 * (bounds[1] - bounds[0])
            for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                local = np.array([middle[0] + sx * extent[0], middle[1] + sy * extent[1], bounds[0, 2]])
                point = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local
                point_jac = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, point_jac, None, point, bid)
                support_jacobians.append(point_jac)
        task = np.zeros((len(self.av), m.nv))
        task[np.arange(len(self.av)), self.av] = 1
        matrices.append(0.5 * task)
        goals.append(0.5 * (40 * (posture - d.qpos[self.aq]) - 10 * d.qvel[self.av]))
        acceleration = np.linalg.lstsq(np.concatenate(matrices), np.concatenate(goals), rcond=1e-5)[0]
        mass = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, mass)
        required = mass @ acceleration + d.qfrc_bias
        active = list(range(len(support_jacobians)))
        for _ in range(len(active)):
            jac = np.concatenate([support_jacobians[i] for i in active])
            force = np.linalg.lstsq(jac[:, r.root_v : r.root_v + 6].T, required[r.root_v : r.root_v + 6], rcond=1e-6)[0]
            normal = force.reshape(-1, 3)[:, 2]
            if normal.min() >= -1e-7 or len(active) <= 1:
                break
            del active[int(np.argmin(normal))]
        torque = np.clip((required - jac.T @ force)[self.av], -self.effort, self.effort)
        target = d.qpos[self.aq] + (torque + self.kd * d.qvel[self.av]) / self.kp
        limits = np.asarray(self.plant.protocol["soft_joint_pos_limits"])[self.ids]
        return np.clip(target, limits[:, 0], limits[:, 1])
