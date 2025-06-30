from __future__ import annotations

import os
import string
import torch
import logging

try:
    # When running *within* the rl package (e.g. `python -m rl.play_hangman_ai`)
    from .train_hangman_ppo import GPT2Policy, obs_to_text  # type: ignore
    from .hangman_gym_env import HangmanGymEnv  # type: ignore
except (ImportError, SystemError):
    # Fallback for running as a plain script: `python rl/play_hangman_ai.py`
    from train_hangman_ppo import GPT2Policy, obs_to_text  # type: ignore
    from hangman_gym_env import HangmanGymEnv  # type: ignore

# Basic logging configuration (ignored if already configured by the trainer)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# Module-level logger
logger = logging.getLogger(__name__)

def load_policy(checkpoint_dir: str = "checkpoints") -> GPT2Policy:
    """Load the latest saved GPT-2 policy from disk."""
    policy = GPT2Policy()
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if not os.path.exists(latest_path):
        raise FileNotFoundError(
            f"No checkpoint found at {latest_path}. Have you trained the model yet?"
        )

    checkpoint = torch.load(latest_path, map_location=policy.device, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()  # inference mode
    return policy


def play_hangman(word: str):
    """Run an interactive AI Hangman session for the supplied *word*."""
    env = HangmanGymEnv(word)
    policy = load_policy()
    letters = string.ascii_lowercase

    # Gymnasium reset returns (obs, info)
    obs, _ = env.reset()
    logger.info("\n=== New Game ===")
    env.render()

    step = 0
    done = False
    total_reward = 0.0

    while not done:
        step += 1
        text_obs = obs_to_text(obs)
        with torch.no_grad():
            action_idx, _, _ = policy.act([text_obs])

        action_int: int = action_idx.item()
        action_letter: str = letters[action_int]

        logger.info("\nStep %d: AI guesses '%s'", step, action_letter)

        next_obs, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated
        obs = next_obs
        total_reward += reward
        env.render()

    # --------------------------------------------------
    # Game over summary
    # --------------------------------------------------
    outcome = "WON! 🎉" if 26 not in obs["mask"] else "lost. 💀"
    logger.info("\nAI %s\nSecret word was: '%s'\nTotal reward: %.2f", outcome, word.lower(), total_reward)


if __name__ == "__main__":
    user_word = input("Enter the secret word for the AI to guess: ").strip().lower()
    play_hangman(user_word) 