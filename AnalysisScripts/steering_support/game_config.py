#!/usr/bin/env python3
"""Model default, environment list, and the argument namespace the miner reads.

`deception_miner`'s helpers take a single namespace of game settings. The
steering runner builds the same one, so walking an environment here produces the
states the localization pipeline would have produced.
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

GAME_ENVS = ("bs", "gridworld", "interview", "car_sales", "advisor_audit")

# Set DECEPTION_MODEL to a local snapshot to avoid re-downloading.
DEFAULT_MODEL = os.environ.get("DECEPTION_MODEL",
                               "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


def build_game_args(a: argparse.Namespace) -> SimpleNamespace:
    """The namespace `deception_miner`'s helpers read."""
    return SimpleNamespace(
        game=a.env, model_name=a.model_name, is_reasoning_model=True,
        temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
        repetition_penalty=a.repetition_penalty, max_retries=3,
        samples_per_state=a.samples_per_state, seed=a.seed,
        num_players=4, cards_per_player=5,
        grid_width=9, grid_height=9, wall_prob=0.18, max_tries=200, max_steps=60,
        view_radius=2, history_window=15, auto_move_explorer=True,
        interview_conversations_path=a.interview_conversations_path,
        interview_private_profile_name=None,
        car_sales_scenario_name=None, car_sales_max_rounds=4,
    )
