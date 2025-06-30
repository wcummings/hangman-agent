import string
from typing import Optional, Tuple, Dict, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging


# Module-level logger (inherits root configuration from trainer)
logger = logging.getLogger(__name__)

class HangmanGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    _UNDERSCORE_TOKEN: int = 26  # special token used for unrevealed letters

    def __init__(self, word_sampler, max_lives: int = 6):
        super().__init__()
        # print(f"Initializing HangmanGymEnv with word: {word} and max_lives: {max_lives}")
        self.word = next(word_sampler)
        if not self.word or not self.word.isalpha():
            raise ValueError("'word' must be a non-empty alphabetic string.")
        if max_lives <= 0:
            raise ValueError("'max_lives' must be positive.")

        self.word: str = self.word.lower()
        self.max_lives: int = max_lives

        # Gymnasium spaces --------------------------------------------------
        self.action_space: spaces.Discrete = spaces.Discrete(26)  # a-z
        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "mask": spaces.Sequence(spaces.Discrete(27)),
                "remaining_lives": spaces.Discrete(max_lives + 1),
                "guessed": spaces.MultiBinary(26),
            }
        )

        # Current episode state ---------------------------------------------
        self._mask: List[str] = []
        self._guessed: set[str] = set()
        self._remaining_lives: int = max_lives
        self._terminated: bool = False

        self.word_sampler = word_sampler

        # Initialise episode -------------------------------------------------
        self.reset(seed=None)

    # ------------------------------------------------------------------
    # Gymnasium required API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Sample a new target word first so that dependent state (mask etc.)
        # is derived from the correct word and remains in sync.
        # NB: ensure the word is lowercase for consistent token mapping.
        self.word = next(self.word_sampler).lower()

        # Reset episode state based on the new word
        self._mask = ["_" for _ in self.word]
        self._guessed = set()
        self._remaining_lives = self.max_lives
        self._terminated = False
        observation = self._get_observation()
        info: Dict = {}
        return observation, info

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("Cannot call step() on a finished episode. Call reset() to start a new game.")

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is invalid for the action space {self.action_space}.")

        letter: str = string.ascii_lowercase[action]
        already_guessed: bool = letter in self._guessed
        self._guessed.add(letter)

        correct: bool = False
        letters_revealed: int = 0

        if not already_guessed:
            if letter in self.word:
                correct = True
                # Reveal all occurrences in the mask and count them
                for idx, ch in enumerate(self.word):
                    if ch == letter:
                        self._mask[idx] = letter
                        letters_revealed += 1
            else:
                self._remaining_lives -= 1
        else:
            self._remaining_lives -= 1 # penalty for guessing a letter that has already been guessed

        # --------------------------------------------------------------
        # Reward shaping (same heuristic as original HangmanEnv)
        # --------------------------------------------------------------
        reward: float = 0.0

        if already_guessed:
            reward -= 1
        else:
            reward += 1  # format reward for providing a new letter

        if correct and not already_guessed:
            # reward scaled by number of unique letters in the word
            reward += (letters_revealed / len(set(self.word))) * 2

        # Termination conditions ----------------------------------------
        if "_" not in self._mask:
            self._terminated = True
            reward += 10  # win bonus
            # logger.info("🎉 Agent won, the word was %s", self.word)
        elif self._remaining_lives <= 0:
            self._terminated = True
            reward -= 1  # loss penalty

        terminated = self._terminated
        truncated = False  # No time-limit based truncation

        info: Dict = {
            "correct": correct,
            "already_guessed": already_guessed,
        }

        # Expose the target word when the episode ends so callers can log it
        if terminated:
            info["word"] = self.word

        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render(self):
        mask_str = " ".join(self._mask)
        guessed_sorted = ", ".join(sorted(self._guessed)) if self._guessed else "<none>"
        print(
            f"Word : {mask_str}\nGuessed : {guessed_sorted}\nLives : {self._remaining_lives}/{self.max_lives}"
        )

    def close(self):
        # No external resources to clean up, but provided for completeness
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_observation(self):
        # Encode current mask
        mask_tokens = [
            (ord(ch) - 97) if ch != "_" else self._UNDERSCORE_TOKEN for ch in self._mask
        ]

        # Encode guessed letters into binary vector
        guessed_bin = np.zeros(26, dtype=np.int8)
        if self._guessed:
            idxs = [ord(ch) - 97 for ch in self._guessed]
            guessed_bin[idxs] = 1

        return {
            "mask": mask_tokens,
            "remaining_lives": self._remaining_lives,
            "guessed": guessed_bin,
        } 