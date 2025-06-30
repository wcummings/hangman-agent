import os
import string
import random
import json
import time
import numpy as np  # Needed for observation processing
from collections import deque, defaultdict
from typing import List, Dict, Any
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter  # TensorBoard integration
from transformers import AutoModel, AutoTokenizer

from torch.amp import autocast, GradScaler
from gymnasium.vector import SyncVectorEnv
import logging


# Configure a module-level logger for consistent, timestamped output
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Increase float32 matmul precision (helps on Apple Silicon; safe to ignore if unsupported)
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    # use logging instead of print for warnings
    logging.warning("set_float32_matmul_precision is not supported on this platform")
    pass

# Do FP optimizaiton

from hangman_gym_env import HangmanGymEnv

def obs_to_text(obs: dict) -> str:
    """Convert environment observation dict (integer tokens) to a textual prompt."""
    # Decode mask tokens (0-25 -> letters, 26 -> '_')
    mask_tokens = obs["mask"]
    mask_chars = ["_" if t == 26 else chr(t + 97) for t in mask_tokens]
    mask_str = " ".join(mask_chars)

    # Decode guessed binary vector into letters
    guessed_vec = obs["guessed"]
    # Support both numpy arrays and Python lists
    if isinstance(guessed_vec, np.ndarray):
        guessed_indices = list(np.where(guessed_vec == 1)[0])
    else:
        guessed_indices = [i for i, g in enumerate(guessed_vec) if g]
    guessed_letters = ", ".join([chr(i + 97) for i in guessed_indices]) if len(guessed_indices) > 0 else "<none>"

    text = f"mask: {mask_str}; guessed: {guessed_letters}; lives: {obs['remaining_lives']}"
    return text


class GPT2Policy(nn.Module):
    """Policy network: DistilGPT-2 backbone + linear head over 26 letters."""

    def __init__(self, lr: float = 3e-5):
        super().__init__()
        # Prefer CUDA (GPU) if available, otherwise fall back to Apple Metal (MPS) and finally CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        # LLMs can process input in batches. To ensure all batches are the same length, padding
        # tokens are added to the end of the sequences. The GPT2 family lacks a pad token by default
        # so we set one here:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.transformer = AutoModel.from_pretrained("distilgpt2").to(self.device)
        
        if self.transformer.config.pad_token_id is None:
            self.transformer.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        hidden_size = self.transformer.config.hidden_size

        # Action head learns to predict the next action, i.e. the next letter to guess
        self.action_head = nn.Linear(hidden_size, 26).to(self.device)
        # Value head learns to guess the expected return (score) of the current state
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, texts: List[str]):
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.transformer(**enc)
        hidden = out.last_hidden_state  # [B, T, H]
        pooled = hidden[:, -1, :]       # [B, H] — use last token as state summary

        logits = self.action_head(pooled)             # [B, 26]
        value = self.value_head(pooled).squeeze(-1)   # [B]

        return logits, value

    def act(self, texts: List[str]):
        """Sample actions and return them with log probabilities."""
        logits, _ = self(texts)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.cpu(), log_probs.cpu(), dist


class TrainingStats:
    """Track and log training statistics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size)
        self.win_history = deque(maxlen=self.window_size)
        self.total_episodes = 0
        self.total_wins = 0
        self.total_losses = 0
        self.start_time = time.time()
        
    def add_episode(self, reward: float, length: int, won: bool):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.win_history.append(won)
        self.total_episodes += 1
        if won:
            self.total_wins += 1
        else:
            self.total_losses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': self.total_episodes,
            'avg_reward': sum(self.episode_rewards) / len(self.episode_rewards),
            'avg_length': sum(self.episode_lengths) / len(self.episode_lengths),
            'win_rate': sum(self.win_history) / len(self.win_history) if self.win_history else 0,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'elapsed_time': time.time() - self.start_time,
        }
    
    def log_stats(self, episode: int):
        stats = self.get_stats()
        if not stats:
            return
            
        logger.info("\n=== Episode %s Stats ===", episode)
        logger.info("Average Reward (last %d): %.2f", len(self.episode_rewards), stats['avg_reward'])
        logger.info("Average Episode Length: %.1f", stats['avg_length'])
        logger.info("Win Rate: %.2f%%", stats['win_rate'] * 100)
        logger.info("Total Wins/Losses: %d/%d", stats['total_wins'], stats['total_losses'])
        logger.info("Elapsed Time: %.1fs", stats['elapsed_time'])
        logger.info("%s", "=" * 30)

class PPOAgent:
    def __init__(
        self,
        env_words: List[str],
        episodes: int = 1000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        update_epochs: int = 4,
        batch_size: int = 32,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 50,
        log_interval: int = 10,
        update_interval: int = 20,
        gradient_accumulation_steps: int = 2,
        expert_episodes: int = 0,
        num_envs: int = 8,
        curriculum_stages: List[List[str]] | None = None,
    ):
        self.env_words = env_words
        # NOTE: this value will be overwritten by the curriculum setup below
        self.episodes = episodes
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        # Ensure checkpoint directory is absolute to avoid accidental writes in unexpected CWDs
        self.checkpoint_dir = (
            checkpoint_dir
            if os.path.isabs(checkpoint_dir)
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoint_dir)
        )
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.update_interval = update_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.expert_episodes = expert_episodes
        self.num_envs = num_envs
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.policy = GPT2Policy()
        self.letters = string.ascii_lowercase
        self.stats = TrainingStats()
        
        # Training state
        self.start_episode = 0
        self.memory = []  # stores transitions for each update cycle
        
        # TensorBoard summary writer
        tb_log_dir = os.path.join(self.checkpoint_dir, "tensorboard")
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        
        # AMP gradient scaler for mixed-precision; enabled only when CUDA is available
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        
        # Track last checkpointed episode to avoid duplicate saves within same episode count
        self._last_checkpoint_episode: int = -1

        self.win_per_word = defaultdict(lambda: [0, 0])  # {word: [wins, total_games]}

        # --------------------------------------------------------------
        # Curriculum setup                                             
        # --------------------------------------------------------------
        if curriculum_stages is None:
            curriculum_stages = [env_words]
        self.curriculum_stages = curriculum_stages
        self.curriculum_stage_idx: int = 0  # starts with the first stage
        # "env_words" always points at the *current* stage's list so that
        # the word‐sampler generator sees updates automatically.
        self.env_words = self.curriculum_stages[self.curriculum_stage_idx]

        # Load checkpoint if exists (updates start_episode etc.)
        self._load_checkpoint()

        # Ensure last_checkpoint is consistent if we resumed
        self._last_checkpoint_episode = max(self._last_checkpoint_episode, self.start_episode - 1)

    def _word_sampler(self):
        while True:
            words = self.env_words[:]
            random.shuffle(words)
            for word in words:
                yield word

    def _save_checkpoint(self, episode: int):
        """Save training checkpoint to disk."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.policy.optimizer.state_dict(),
            'stats': {
                'episode_rewards': list(self.stats.episode_rewards),
                'episode_lengths': list(self.stats.episode_lengths),
                'win_history': list(self.stats.win_history),
                'total_episodes': self.stats.total_episodes,
                'total_wins': self.stats.total_wins,
                'total_losses': self.stats.total_losses,
                'start_time': self.stats.start_time,
            },
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_eps': self.clip_eps,
                'update_epochs': self.update_epochs,
                'batch_size': self.batch_size,
            },
            'win_per_word': dict(self.win_per_word),
            'curriculum_stage_idx': self.curriculum_stage_idx,
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_ep_{episode}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        
        logger.info("Checkpoint saved: %s", checkpoint_path)

    def _load_checkpoint(self):
        """Load training checkpoint from disk if it exists."""
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        
        if os.path.exists(latest_path):
            logger.info("Loading checkpoint from %s", latest_path)
            checkpoint = torch.load(latest_path, map_location=self.policy.device, weights_only=False)
            
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_episode = checkpoint['episode'] + 1
            
            # Restore stats
            stats_data = checkpoint['stats']
            self.stats.episode_rewards = deque(stats_data['episode_rewards'], maxlen=self.stats.window_size)
            self.stats.episode_lengths = deque(stats_data['episode_lengths'], maxlen=self.stats.window_size)
            self.stats.win_history = deque(stats_data['win_history'], maxlen=self.stats.window_size)
            self.stats.total_episodes = stats_data['total_episodes']
            self.stats.total_wins = stats_data['total_wins']
            self.stats.total_losses = stats_data['total_losses']
            self.stats.start_time = stats_data['start_time']
            
            # Restore word-level win stats if available
            wpw = checkpoint.get('win_per_word')
            if wpw is not None:
                # Use defaultdict to maintain default structure
                self.win_per_word = defaultdict(lambda: [0, 0], wpw)
            
            # Restore curriculum stage if present
            self.curriculum_stage_idx = checkpoint.get('curriculum_stage_idx', 0)
            self.curriculum_stage_idx = min(self.curriculum_stage_idx, len(self.curriculum_stages) - 1)
            self.env_words = self.curriculum_stages[self.curriculum_stage_idx]
            
            logger.info("Resumed from episode %d", self.start_episode)
            self.stats.log_stats(self.start_episode - 1)
        else:
            logger.info("No checkpoint found, cold starting with %d expert episodes", self.expert_episodes)
            self._inject_expert_episodes(self.expert_episodes)
            self._update_policy()
            self.memory.clear()

        # Ensure last_checkpoint is consistent if we resumed
        self._last_checkpoint_episode = max(self._last_checkpoint_episode, self.start_episode - 1)

    def train(self):
        """Vectorised PPO training loop using *SyncVectorEnv* for batched data-collection."""

        # --------------------------------------------------------------
        # Create a vectorised environment made up of *self.num_envs*     
        # independent Hangman games.                                     
        # --------------------------------------------------------------
        def _make_env():
            # Each environment starts with a random word; the RandomWord reset logic isn't needed
            # because Gymnasium vector env automatically resets an episode once it terminates.
            return HangmanGymEnv(self._word_sampler())

        envs = SyncVectorEnv([_make_env for _ in range(self.num_envs)])

        obs, _ = envs.reset()

        ep_rewards = [0.0] * self.num_envs
        ep_lengths = [0] * self.num_envs

        episodes_since_update = 0

        # Continue until the desired number of *completed* episodes has been reached
        while self.stats.total_episodes < self.episodes:
            # ----------------------------------------------------------
            # Convert vector obs → list[str] that the language-model can understand
            # ----------------------------------------------------------
            batch_text_obs = [
                obs_to_text({k: obs[k][i] for k in obs}) for i in range(self.num_envs)
            ]

            with torch.no_grad():
                actions_tensor, logp_tensor, _ = self.policy.act(batch_text_obs)

            actions = actions_tensor.numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)

            # ----------------------------------------------------------
            # Store transitions & manage episode ends                    
            # ----------------------------------------------------------
            for i in range(self.num_envs):
                self.memory.append(
                    (
                        batch_text_obs[i],
                        int(actions[i]),
                        float(logp_tensor[i]),
                        float(rewards[i]),
                        bool(dones[i]),
                    )
                )

                ep_rewards[i] += rewards[i]
                ep_lengths[i] += 1

                if dones[i]:
                    won = 26 not in next_obs["mask"][i]
                    self.stats.add_episode(ep_rewards[i], ep_lengths[i], won)

                    # TensorBoard per-episode metrics
                    self.writer.add_scalar("Episode/Reward", ep_rewards[i], self.stats.total_episodes)
                    self.writer.add_scalar("Episode/Length", ep_lengths[i], self.stats.total_episodes)
                    self.writer.add_scalar("Episode/Win", int(won), self.stats.total_episodes)

                    # Console logging
                    if self.stats.total_episodes % self.log_interval == 0:
                        logger.info(
                            "Episode %d: reward=%.2f, length=%d, won=%s",
                            self.stats.total_episodes,
                            ep_rewards[i],
                            ep_lengths[i],
                            won,
                        )

                        if won:
                            # Try to obtain the word from the env instance first; fall back to info dict
                            winning_word = getattr(envs.envs[i], "word", None)
                            if winning_word is None:
                                if isinstance(infos, (list, tuple)) and i < len(infos) and isinstance(infos[i], dict):
                                    winning_word = infos[i].get("word")
                                elif isinstance(infos, dict):
                                    # VectorEnv may return dict-of-lists shape
                                    word_list = infos.get("word")
                                    if word_list is not None and i < len(word_list):
                                        winning_word = word_list[i]

                            if winning_word is not None:
                                logger.info("✅ Winning word: %s", winning_word)

                    if self.stats.total_episodes % (self.log_interval * 5) == 0:
                        self.stats.log_stats(self.stats.total_episodes)

                    # Reset counters for that env (vector env already auto-reset)
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

                    episodes_since_update += 1

                    # -------------------------------
                    # Word-level win tracking
                    # -------------------------------
                    current_word = getattr(envs.envs[i], "word", None)
                    if current_word is None:
                        # Fallback to infos dict structure(s)
                        if isinstance(infos, (list, tuple)) and i < len(infos) and isinstance(infos[i], dict):
                            current_word = infos[i].get("word")
                        elif isinstance(infos, dict):
                            word_list = infos.get("word")
                            if word_list is not None and i < len(word_list):
                                current_word = word_list[i]

                    if current_word is not None:
                        self.win_per_word[current_word][1] += 1  # total games
                        if won:
                            self.win_per_word[current_word][0] += 1  # wins

                        # Occasionally log breakdown by word
                        if self.stats.total_episodes % (self.log_interval * 10) == 0:
                            logger.info("\n--- Win rates by word (wins/total | win%%) ---")
                            for w, (win_cnt, total_cnt) in sorted(self.win_per_word.items()):
                                logger.info("%s: %d/%d = %.2f%%", w, win_cnt, total_cnt, (win_cnt / total_cnt) * 100 if total_cnt else 0.0)

                    # --------------------------------------------------
                    # Curriculum progression
                    # --------------------------------------------------
                    self._maybe_advance_curriculum()

            obs = next_obs

            # ----------------------------------------------------------
            # Optimisation step & checkpointing                          
            # ----------------------------------------------------------
            if episodes_since_update >= self.update_interval:
                self._update_policy()
                self.memory.clear()
                episodes_since_update = 0

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # --------------------------------------------------------------
            # Checkpointing
            # --------------------------------------------------------------
            # In a vector-env we can finish several episodes in a single
            # environment step (``num_envs`` could be > 1).  That means the
            # episode counter may jump *past* an exact multiple of
            # ``checkpoint_interval`` and we would miss the equality check.
            #
            # Instead, we save whenever we have crossed the next checkpoint
            # boundary relative to ``_last_checkpoint_episode``.
            next_checkpoint = self._last_checkpoint_episode + self.checkpoint_interval
            if self.stats.total_episodes >= next_checkpoint > self._last_checkpoint_episode:
                self._save_checkpoint(self.stats.total_episodes)
                self._last_checkpoint_episode = self.stats.total_episodes

        # -----------------------------
        # Training completed           
        # -----------------------------
        if self.stats.total_episodes != self._last_checkpoint_episode:
            self._save_checkpoint(self.stats.total_episodes)
        torch.save(self.policy, os.path.join(self.checkpoint_dir, "final_model.pth"))
        logger.info("Training completed!")

        envs.close()
        self.writer.close()

    # --------------------------------------------------------------
    # PPO update logic (optimized for memory)
    # --------------------------------------------------------------
    def _compute_returns_advantages(self, rewards, dones, values):
        returns = []
        advantages = []
        gae = 0
        future_return = 0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - dones[i]
            future_return = rewards[i] + self.gamma * future_return * mask
            returns.insert(0, future_return)

            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
        return torch.tensor(returns), torch.tensor(advantages)

    def _update_policy(self):
        if not self.memory:
            return
        
        # Enable anomaly detection for debugging NaN's
        torch.autograd.set_detect_anomaly(True)
            
        logger.info("Updating policy with %d transitions...", len(self.memory))

        # Unpack memory
        texts, actions, old_logp, rewards, dones = zip(*self.memory)
        device = self.policy.device
        actions = torch.tensor(actions, device=device)
        old_logp = torch.tensor(old_logp, device=device)
        rewards = list(rewards)
        dones = list(map(float, dones))

        logger.info("[Returns] Min: %.2f, Max: %.2f, Mean: %.2f", min(rewards), max(rewards), sum(rewards)/len(rewards))

        with torch.no_grad():
            logits, values = self.policy(texts)
            values = torch.cat([values, torch.tensor([0.0], device=device)])  # append V(s_T)=0

        returns, advantages = self._compute_returns_advantages(rewards, dones, values)
        returns = returns.to(device)
        # Normalize returns like advantages, but guard against very high std dev
        std = returns.std()
        if std > 1e-5:
            returns = (returns - returns.mean()) / (std + 1e-8)
        else:
            logger.warning("Skipping return normalization due to low std: %s", std.item())

        advantages = advantages.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logger.debug("First 10 returns: %s", returns[:10].tolist())
        logger.debug("Min/Max/Mean return: %.2f / %.2f / %.2f", returns.min().item(), returns.max().item(), returns.mean().item())

        dataset = list(zip(texts, actions, old_logp, returns, advantages))
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        update_count = 0
        
        for epoch in range(self.update_epochs):
            random.shuffle(dataset)
            
            # Use gradient accumulation for memory efficiency
            self.policy.optimizer.zero_grad()
            
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i : i + self.batch_size]
                b_texts, b_actions, b_old_logp, b_returns, b_adv = zip(*batch)

                with autocast(
                    enabled=self.scaler.is_enabled(),
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    device_type="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
                ):
                    logits, values = self.policy(list(b_texts))
                    # Clip values to prevent explosion (cast to fp32 first for numerical stability)
                    values = torch.clamp(values.float(), -100.0, 100.0)

                dist = Categorical(logits=logits.float())
                new_logp = dist.log_prob(torch.stack(b_actions))
                ratio = (new_logp - torch.stack(b_old_logp)).exp()

                # Policy loss
                unclipped = ratio * torch.stack(b_adv)
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * torch.stack(b_adv)
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = (values - torch.stack(b_returns)).pow(2).mean()
                # Total loss
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                self.writer.add_scalar('Policy/Entropy', entropy.item(), self.stats.total_episodes)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate loss statistics
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                update_count += 1

                # Update weights every gradient_accumulation_steps
                if (i // self.batch_size + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler.is_enabled():
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.policy.optimizer)
                        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                        self.scaler.step(self.policy.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                        self.policy.optimizer.step()
                    self.policy.optimizer.zero_grad()

        # Final optimizer step **only if** there are still gradients waiting to be applied. This
        # avoids calling ``GradScaler.step`` without a prior ``scale``/``backward`` pass, which
        # triggers the "No inf checks were recorded" assertion observed on some hardware.

        gradients_present = any(
            p.grad is not None and p.grad.data is not None for p in self.policy.parameters()
        )

        if gradients_present:
            if self.scaler.is_enabled():
                # Safely apply the remaining accumulated gradients
                self.scaler.unscale_(self.policy.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.scaler.step(self.policy.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy.optimizer.step()
        
        # Log training metrics
        if update_count > 0:
            avg_loss = total_loss / update_count
            avg_policy_loss = total_policy_loss / update_count
            avg_value_loss = total_value_loss / update_count
            avg_value_pred = values.mean().item()
            # v_pred = values[:-1]
            # if v_pred.numel() == 0:
            #     avg_value_pred = 0.0
            # else:
            #     avg_value_pred = v_pred.mean().item()
            logger.info(
                "Training Loss: %.4f (Policy: %.4f, Value: %.4f)",
                avg_loss,
                avg_policy_loss,
                avg_value_loss,
            )
            logger.info(
                "Avg value pred: %.2f, Avg return: %.2f",
                avg_value_pred,
                returns.mean().item(),
            )
            logger.debug("Entropy: %.2f", dist.entropy().mean().item())

            # --- TensorBoard logging for losses ---
            self.writer.add_scalar('Loss/Total', avg_loss, self.stats.total_episodes)
            self.writer.add_scalar('Loss/Policy', avg_policy_loss, self.stats.total_episodes)
            self.writer.add_scalar('Loss/Value', avg_value_loss, self.stats.total_episodes)
            self.writer.add_scalar('ValueHead/Mean_Predicted', values[:-1].mean().item(), self.stats.total_episodes)
            self.writer.add_scalar('ValueHead/Mean_Return', returns.mean().item(), self.stats.total_episodes)
            for name, param in self.policy.named_parameters():
                if "action_head" in name:
                    self.writer.add_histogram(f"Weights/ActionHead/{name}", param, self.stats.total_episodes)
                elif "value_head" in name:
                    self.writer.add_histogram(f"Weights/ValueHead/{name}", param, self.stats.total_episodes)

    def _inject_expert_episodes(self, num_episodes: int = 5):
        logger.info("🔁 Injecting %d expert episodes into memory...", num_episodes)
        word_iter = self._word_sampler()
        for _ in range(num_episodes):
            env = HangmanGymEnv(word_iter)
            obs, _ = env.reset()
            ep_done = False
            word = env.word
            guessed = set()

            for letter in sorted(set(word)):  # guess unique letters only
                text_obs = obs_to_text(obs)
                action_idx = self.letters.index(letter)
                logp = 0.0  # not used, just a dummy value

                obs, reward, terminated, truncated, info = env.step(action_idx)
                ep_done = terminated or truncated

                # Always record the transition — even if it's terminal
                self.memory.append((text_obs, action_idx, logp, reward, ep_done))
                guessed.add(letter)

                if ep_done:
                    break  # ✅ only break after the winning step is included


            logger.debug("Injected: '%s' with guesses: %s", word, guessed)

    # ------------------------------------------------------------------
    # Curriculum helpers
    # ------------------------------------------------------------------
    def _maybe_advance_curriculum(self):
        """Advance to the next curriculum stage once the agent achieves
        a 100% win-rate over the sliding window defined in TrainingStats.
        """
        # Nothing to do if we are already at the final stage
        if self.curriculum_stage_idx >= len(self.curriculum_stages) - 1:
            return

        # Require a *full* window of victories to move on
        if len(self.stats.win_history) == self.stats.window_size and all(self.stats.win_history):
            self.curriculum_stage_idx += 1
            self.env_words = self.curriculum_stages[self.curriculum_stage_idx]
            # Reset word-level tracking so that logging only reflects the
            # current stage.
            self.win_per_word.clear()

            logger.info(
                "🎓 Curriculum advanced to stage %d/%d – now training on %d words",
                self.curriculum_stage_idx + 1,
                len(self.curriculum_stages),
                len(self.env_words),
            )


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Curriculum definition                                         
    # --------------------------------------------------------------
    SMALL_WORDS = ["car", "bus", "dog", "cat"]
    FULL_WORDS = [
        "car", "bus", "dog", "cat", "judo", "aloe", "soda",
        "apple", "banana", "cherry", "date", "fig", "grape",
        "planet", "orange", "rabbit",
    ]

    CURRICULUM = [SMALL_WORDS, FULL_WORDS]

    agent = PPOAgent(
        env_words=FULL_WORDS,  # used as fallback; curriculum will start with SMALL_WORDS
        curriculum_stages=CURRICULUM,
        episodes=7_000,
        checkpoint_interval=500,
        log_interval=10,
        update_interval=20,
        gradient_accumulation_steps=2,
        expert_episodes=15,
        num_envs=8
    )
    agent.train() 