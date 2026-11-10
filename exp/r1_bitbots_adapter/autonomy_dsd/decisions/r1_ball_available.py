from dynamic_stack_decider.abstract_decision_element import AbstractDecisionElement


class R1BallAvailable(AbstractDecisionElement):
    def perform(self, reevaluate=False):
        ctx = self.blackboard.r1
        now = self.blackboard.node.get_clock().now().nanoseconds / 1e9
        valid = (
            0 <= now - ctx.ball_time <= self.blackboard.config["r1"]["ball_max_age"]
            and ctx.ball_valid
        )
        return (
            "YES"
            if valid and self.blackboard.world_model.ball_has_been_seen()
            else "NO"
        )

    def get_reevaluate(self):
        return True
