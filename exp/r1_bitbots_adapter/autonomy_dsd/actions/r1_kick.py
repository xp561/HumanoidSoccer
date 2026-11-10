import math
from action_msgs.msg import GoalStatus
from bitbots_msgs.action import Kick
from bitbots_body_behavior.behavior_dsd.actions.kick_ball import RLKick


class R1Kick(RLKick):
    def perform(self, reevaluate=False):
        self.do_not_reevaluate()
        if self._start_time is None:
            self.blackboard.pathfinding.stop_walk()
            _, _, heading = self.blackboard.world_model.get_current_position()
            self._direction_deg_map = math.degrees(heading)
        super().perform(reevaluate)
        if (
            self._start_time is not None
            and not self.blackboard.kick.is_currently_kicking
        ):
            kick = self.blackboard.kick
            if (
                kick.last_rl_kick_status == GoalStatus.STATUS_SUCCEEDED
                and kick.last_rl_kick_result == Kick.Result.SUCCESS
            ):
                self.blackboard.r1.completed += 1
            else:
                self.blackboard.r1.fault = "supervised_kick_failed"
