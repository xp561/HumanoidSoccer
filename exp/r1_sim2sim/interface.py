"""Strict policy interface using named joints from the Isaac runtime."""
import numpy as np


def inverse_rotate(quaternion, vector):
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("Expected a wxyz quaternion and a spatial vector")
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite rotation input")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Zero quaternion")
    q = q / norm
    cross = np.cross(q[1:], v)
    return v - 2*q[0]*cross + 2*np.cross(q[1:], cross)


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite shape {(size,)}, got {result.shape}")
    return result


def observation(metadata, *, ref_joint_pos, ref_joint_vel, ref_ang_vel,
                root_pos, root_quat, root_ang_vel, joint_pos, joint_vel,
                last_action, ball_pos, goal_pos):
    expected = ["command", "projected_gravity", "motion_ref_ang_vel", "base_ang_vel",
                "joint_pos", "joint_vel", "actions", "target_point_pos", "target_destination_pos_local"]
    if metadata["observation_terms"] != expected:
        raise ValueError("Unsupported observation terms or ordering")
    n = len(metadata["joint_names"])
    a = len(metadata["action_joint_names"])
    root = vector(root_pos, 3, "root position")
    q = vector(root_quat, 4, "root orientation")
    parts = [vector(ref_joint_pos,n,"reference joints"), vector(ref_joint_vel,n,"reference velocities"),
             inverse_rotate(q, [0,0,-1]), vector(ref_ang_vel,3,"reference angular velocity"),
             inverse_rotate(q, vector(root_ang_vel,3,"world angular velocity")),
             vector(joint_pos,n,"joint positions") - vector(metadata["default_joint_pos"],n,"default joints"),
             vector(joint_vel,n,"joint velocities"), vector(last_action,a,"previous actions"),
             inverse_rotate(q,vector(ball_pos,3,"ball position")-root),
             inverse_rotate(q,vector(goal_pos,3,"goal position")-root)]
    result = np.concatenate(parts).astype(np.float32)
    if result.shape != (4*n+a+15,):
        raise ValueError("Observation layout mismatch")
    return result


class RecurrentPolicy:
    def __init__(self, path, metadata):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs = {x.name:x for x in self.session.get_inputs()}
        if set(inputs) != {"obs","h_in","c_in"}:
            raise ValueError("Expected a recurrent actor export with explicit hidden states")
        self.obs_dim = inputs["obs"].shape[-1]
        self.action_dim = len(metadata["action_joint_names"])
        if self.obs_dim != 4*len(metadata["joint_names"])+self.action_dim+15:
            raise ValueError("Policy and metadata dimensions disagree")
        self.state_shape = tuple(inputs["h_in"].shape)
        self.reset()

    def reset(self):
        self.h = np.zeros(self.state_shape,dtype=np.float32)
        self.c = np.zeros(self.state_shape,dtype=np.float32)

    def step(self, obs):
        x = vector(obs,self.obs_dim,"policy observation").astype(np.float32)[None]
        outputs = self.session.run(["actions","h_out","c_out"],{"obs":x,"h_in":self.h,"c_in":self.c})
        action,self.h,self.c = outputs
        return vector(action[0],self.action_dim,"policy action").astype(np.float32)
