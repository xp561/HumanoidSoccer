from dynamic_stack_decider.abstract_action_element import AbstractActionElement
from geometry_msgs.msg import Twist


class R1Search(AbstractActionElement):
    def perform(self, reevaluate=False):
        self.blackboard.pathfinding.cancel_goal()
        msg = Twist()
        msg.angular.z = self.blackboard.config["r1"]["search_yaw_speed"]
        self.blackboard.pathfinding.direct_cmd_vel_pub.publish(msg)
        self.blackboard.r1.search_ticks += 1
