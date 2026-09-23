"""Think-exit logit bonus. Imports the sampler step the server actually runs."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_openai import SS_ThinkExit


def step():
    return SS_ThinkExit(close_id = 99, prompt_len = 5, budget = 4, ramp = 4, max_bias = 16)


def test_no_bonus_when_close_tag_is_already_in_the_prompt():
    past = torch.tensor([[1, 99, 2, 3, 4, 5, 6, 7, 8, 9]])
    assert step()._bias(past) == 0


def test_no_bonus_when_close_tag_is_in_the_generated_tokens():
    past = torch.tensor([[1, 2, 3, 4, 5, 7, 99, 8, 9, 10, 11]])
    assert step()._bias(past) == 0


def test_no_bonus_while_generated_length_is_within_the_budget():
    # 5 prompt tokens, then exactly `budget` generated tokens, think still open.
    past = torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9, 10]])
    assert step()._bias(past) == 0


def test_bonus_rises_after_the_budget_and_caps():
    s = step()
    # 2 tokens over a ramp of 4 -> half of max_bias.
    past = torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])
    assert abs(s._bias(past) - 8) < 1e-6
    # one more generated token raises the bonus further.
    longer = torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]])
    assert s._bias(longer) > s._bias(past)
    # past the ramp the bonus stays at max_bias.
    way_over = torch.tensor([[1, 2, 3, 4, 5] + [7] * 20])
    assert abs(s._bias(way_over) - 16) < 1e-6


class _State:
    def __init__(self, past, logits):
        self.past_ids = past
        self.logits = logits
        self.dim = logits.shape[-1]


def test_think_freq_lowers_repeated_tokens_only_while_think_is_open():
    s = SS_ThinkExit(5, prompt_len = 2, budget = 100, ramp = 4, max_bias = 16, think_freq = 1.0, think_freq_window = 32)
    open_state = _State(torch.tensor([[1, 2] + [7] * 16 + [4]]), torch.zeros(1, 12))
    s._penalize_open_think(open_state)
    assert open_state.logits[0, 7].item() == -16
    assert open_state.logits[0, 5].item() == 0
    closed = _State(torch.tensor([[1, 5, 7, 7, 7, 7]]), torch.zeros(1, 12))
    s._penalize_open_think(closed)
    assert torch.count_nonzero(closed.logits).item() == 0


if __name__ == "__main__":
    test_no_bonus_when_close_tag_is_already_in_the_prompt()
    test_no_bonus_when_close_tag_is_in_the_generated_tokens()
    test_no_bonus_while_generated_length_is_within_the_budget()
    test_bonus_rises_after_the_budget_and_caps()
    test_think_freq_lowers_repeated_tokens_only_while_think_is_open()
    print("think-exit tests passed")
