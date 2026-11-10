from dynamic_stack_decider.abstract_decision_element import AbstractDecisionElement


class R1Operational(AbstractDecisionElement):
    def perform(self, reevaluate=False):
        ctx = self.blackboard.r1
        now = self.blackboard.node.get_clock().now().nanoseconds / 1e9
        if ctx.fault or ctx.status.get("phase") == "fault":
            return "FAULT"
        if (
            not ctx.enabled
            or not ctx.localization_valid
            or not 0 <= now - ctx.status_time <= 0.3
            or now < self.blackboard.config["r1"]["warmup_seconds"]
        ):
            return "HOLD"
        return "RUN"

    def get_reevaluate(self):
        return True
