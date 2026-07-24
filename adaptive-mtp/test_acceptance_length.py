import unittest

from overlay.vllm.v1.spec_decode.dynamic.acceptance_length import (
    AcceptanceLengthController,
)


def observe(controller, *, drafted, accepted, steps):
    update = None
    for _ in range(steps):
        eligible = [0] * controller.max_num_spec_tokens
        position_accepted = [0] * controller.max_num_spec_tokens
        for position in range(1, controller.max_num_spec_tokens + 1):
            if drafted >= position:
                if position == 1 or accepted >= position - 1:
                    eligible[position - 1] = 1
                if accepted >= position:
                    position_accepted[position - 1] = 1
        update = controller.observe_batch(
            num_drafts=1,
            num_draft_tokens=drafted,
            num_accepted_tokens=accepted,
            position_eligible=eligible,
            position_accepted=position_accepted,
        )
    return update


class ProductionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.controller = AcceptanceLengthController(
            max_num_spec_tokens=5,
            observation_window=4,
            depth_ladder=(2, 4, 5),
        )

    def test_code_climbs_and_holds_k5(self):
        update = observe(self.controller, drafted=2, accepted=2, steps=4)
        self.assertEqual((update.num_spec_tokens, update.decision_reason), (4, "probe_k4"))

        update = observe(self.controller, drafted=4, accepted=4, steps=2)
        self.assertEqual((update.num_spec_tokens, update.decision_reason), (5, "probe_k5"))

        update = observe(self.controller, drafted=5, accepted=5, steps=2)
        self.assertEqual((update.num_spec_tokens, update.decision_reason), (5, "k5_hold"))

    def test_prose_stays_at_baseline(self):
        update = observe(self.controller, drafted=2, accepted=1, steps=4)
        self.assertEqual((update.num_spec_tokens, update.decision_reason), (2, "k2_baseline"))

    def test_bad_tail_falls_directly_to_k2(self):
        observe(self.controller, drafted=2, accepted=2, steps=4)
        observe(self.controller, drafted=4, accepted=4, steps=2)
        update = observe(self.controller, drafted=5, accepted=2, steps=2)
        self.assertEqual((update.num_spec_tokens, update.decision_reason), (2, "k5_tail_reject"))

    def test_k5_without_p4_falls_to_k4(self):
        observe(self.controller, drafted=2, accepted=2, steps=4)
        observe(self.controller, drafted=4, accepted=4, steps=2)
        update = observe(self.controller, drafted=5, accepted=4, steps=2)
        self.assertEqual((update.num_spec_tokens, update.decision_reason), (4, "k5_p4_reject"))


if __name__ == "__main__":
    unittest.main()
