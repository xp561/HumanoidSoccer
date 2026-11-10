import math
from dynamic_stack_decider.abstract_decision_element import AbstractDecisionElement
from bitbots_blackboard.capsules.pathfinding_capsule import BallGoalType


class R1AtApproach(AbstractDecisionElement):
    def __init__(self, blackboard, dsd, parameters):
        super().__init__(blackboard, dsd, parameters)
        self.since = None

    def perform(self, reevaluate=False):
        cfg = self.blackboard.config["r1"]
        goal = self.blackboard.pathfinding.get_ball_goal(
            BallGoalType.MAP, cfg["approach_distance"]
        )
        x, y, yaw = self.blackboard.world_model.get_current_position()
        q = goal.pose.orientation
        desired = math.atan2(
            2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
        )
        error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
        distance = math.hypot(goal.pose.position.x - x, goal.pose.position.y - y)
        now = self.blackboard.node.get_clock().now().nanoseconds / 1e9
        near = (
            distance < cfg["approach_position_tolerance"]
            and abs(error) < cfg["approach_yaw_tolerance"]
        )
        if not near:
            self.since = None
        elif self.since is None:
            self.since = now
        self.blackboard.r1.approach = dict(distance=distance, yaw_error=error)
        return (
            "YES"
            if self.since is not None and now - self.since >= cfg["approach_dwell"]
            else "NO"
        )

    def get_reevaluate(self):
        return True
