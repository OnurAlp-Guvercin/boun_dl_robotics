import argparse
import json
import multiprocessing as mp
import os
import queue
import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from homework2 import Hw2Env

from tqdm import tqdm
import matplotlib.pyplot as plt

STATE_HIGH_LEVEL = "high_level"
STATE_PIXELS = "pixels"
STATE_CHOICES = [STATE_HIGH_LEVEL, STATE_PIXELS]

CMD_TRAIN = "train"
CMD_TEST = "test"
DEFAULT_COMMAND = CMD_TEST

DEFAULT_RUN_DIR = "runs_new_2/hw2/dqn"
DEFAULT_DEVICE = "auto"
DEFAULT_RENDER_MODE = "offscreen" # offscreen or gui
DEFAULT_STATE_MODE = STATE_HIGH_LEVEL

# State-based default hyperparameters (course update).
DEFAULT_N_ACTIONS = 8
DEFAULT_MAX_TIMESTEPS = 50
DEFAULT_N_EPISODES = 2_500
DEFAULT_BATCH_SIZE = 128
DEFAULT_GAMMA = 0.995
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_MIN = 0.10
DEFAULT_EPSILON_DECAY = 20_000
DEFAULT_TAU = 0.01
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_REPLAY_BUFFER_SIZE = 20_000
DEFAULT_WARMUP_EPISODES = 100
DEFAULT_LEARN_UPDATES = 1

DEFAULT_SEED = 42
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_LOG_EVERY = 25
DEFAULT_EVAL_EPISODES = 100
DEFAULT_EVAL_EPSILON = 0.0
DEFAULT_NUM_COLLECTORS = 16
DEFAULT_COLLECTOR_QUEUE_SIZE = 40_000
DEFAULT_COLLECTOR_SYNC_UPDATES = 25

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EpisodeMetric(TypedDict):
    episode: int
    total_reward: float
    reward_per_step: float
    steps: float
    epsilon: float
    mean_loss: Optional[float]
    replay_size: float


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != DEFAULT_DEVICE:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_existing_path(path_str: str) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Path does not exist: {raw}")

    candidates: List[Path] = []
    for candidate in ((Path.cwd() / raw).resolve(), (PROJECT_ROOT / raw).resolve()):
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Checkpoint file not found: '{path_str}'. Searched:\n{searched}\n"
        "Run training first or pass --checkpoint-path with the correct file."
    )


def moving_average(values: List[float], window: int) -> List[float]:
    """Draw a moving average curve with the specified window size. The result is the same length as input."""
    if not values:
        return []
    window = max(1, min(window, len(values)))
    cumsum = np.cumsum(np.array(values, dtype=np.float64))
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        result.append(float(total / (i - start + 1)))
    return result


def save_training_plots(history: List[EpisodeMetric], out_dir: Path) -> None:
    """Saves reward and reward-per-step plots from training history."""
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = [int(h["episode"]) for h in history]
    rewards = [float(h["total_reward"]) for h in history]
    rps_vals = [float(h["reward_per_step"]) for h in history]
    smooth_window = min(100, len(history))
    rewards_ma = moving_average(rewards, smooth_window)
    rps_ma = moving_average(rps_vals, smooth_window)

    plt.figure(figsize=(9, 4))
    plt.plot(episodes, rewards, alpha=0.35, linewidth=1.0, label="reward")
    plt.plot(episodes, rewards_ma, linewidth=2.0, label=f"reward (MA{smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(episodes, rps_vals, alpha=0.35, linewidth=1.0, label="reward_per_step")
    plt.plot(episodes, rps_ma, linewidth=2.0, label=f"reward_per_step (MA{smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward per Step")
    plt.title("DQN Training Reward per Step")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rps_plot.png", dpi=150)
    plt.close()


def _obs_to_float_numpy(obs) -> np.ndarray:
    if isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy()
    else:
        arr = np.asarray(obs)
    return arr.astype(np.float32, copy=False)


def get_observation(env: Hw2Env, state_mode: str, step_state=None) -> np.ndarray:
    if state_mode == STATE_HIGH_LEVEL:
        return np.asarray(env.high_level_state(), dtype=np.float32)
    if step_state is None:
        step_state = env.state()
    arr = _obs_to_float_numpy(step_state)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def to_storage_state(obs: np.ndarray, state_mode: str) -> np.ndarray:
    if state_mode == STATE_HIGH_LEVEL:
        return np.asarray(obs, dtype=np.float32).copy()
    arr = np.asarray(obs, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def batch_states_to_tensor(states: Tuple[np.ndarray, ...], state_mode: str, device: torch.device) -> torch.Tensor:
    stacked = np.stack(states, axis=0)
    if state_mode == STATE_HIGH_LEVEL:
        return torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
    return torch.from_numpy(stacked).to(device=device, dtype=torch.float32).div_(255.0)


class ReplayBuffer:
    def __init__(self, capacity: int, state_mode: str) -> None:
        self.capacity = int(capacity)
        self.state_mode = state_mode
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append(
            (
                to_storage_state(state, self.state_mode),
                int(action),
                float(reward),
                to_storage_state(next_state, self.state_mode),
                float(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[int(i)] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = batch_states_to_tensor(states, self.state_mode, device)
        next_states_t = batch_states_to_tensor(next_states, self.state_mode, device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        return states_t, actions_t, rewards_t, next_states_t, dones_t


class MLPQNetwork(nn.Module):
    def __init__(self, in_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNQNetwork(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = feat.mean(dim=(2, 3))
        return self.head(feat)


def build_q_network(state_mode: str, n_actions: int) -> nn.Module:
    if state_mode == STATE_HIGH_LEVEL:
        return MLPQNetwork(in_dim=6, n_actions=n_actions)
    return CNNQNetwork(n_actions=n_actions)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.mul_(1.0 - tau).add_(src_param, alpha=tau)


def epsilon_by_step(step: int, eps_start: float, eps_end: float, eps_decay: float) -> float:
    """Calculate epsilon value for epsilon-greedy action selection based on the current step."""
    if eps_decay <= 0:
        return float(eps_end)
    value = eps_end + (eps_start - eps_end) * np.exp(-float(step) / float(eps_decay))
    return float(max(eps_end, value))


def epsilon_greedy_action(
    model: nn.Module,
    obs: np.ndarray,
    epsilon: float,
    n_actions: int,
    state_mode: str,
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        return int(np.random.randint(n_actions))
    model.eval()
    with torch.no_grad():
        obs_t = batch_states_to_tensor((obs,), state_mode=state_mode, device=device)
        q_values = model(obs_t)
    return int(torch.argmax(q_values, dim=1).item())


def optimize_dqn(
    online_model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    grad_clip: float,
    device: torch.device,
) -> Optional[float]:
    if len(replay_buffer) < batch_size:
        return None

    online_model.train()
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size=batch_size, device=device)

    q_values = online_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_next_q = target_model(next_states).max(dim=1).values
        target_q = rewards + gamma * (1.0 - dones) * max_next_q

    loss = F.smooth_l1_loss(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=grad_clip)
    optimizer.step()
    soft_update(target_model, online_model, tau)
    return float(loss.item())


def _model_state_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _queue_put_with_stop(q, item, stop_event, timeout_s: float = 0.1) -> bool:
    while not stop_event.is_set():
        try:
            q.put(item, timeout=timeout_s)
            return True
        except queue.Full:
            continue
    return False


def _queue_replace_latest(cmd_queue, item) -> None:
    while True:
        try:
            cmd_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                cmd_queue.get_nowait()
            except queue.Empty:
                return


def _collector_worker(
    worker_id: int,
    state_mode: str,
    n_actions: int,
    max_timesteps: int,
    seed: int,
    render_mode: str,
    data_queue,
    cmd_queue,
    stop_event,
) -> None:
    set_seeds(seed + 1000 + worker_id)
    env = Hw2Env(n_actions=n_actions, render_mode=render_mode)
    env._max_timesteps = int(max_timesteps)

    model = build_q_network(state_mode=state_mode, n_actions=n_actions).cpu()
    model.eval()
    epsilon = 1.0

    def _drain_commands() -> None:
        nonlocal epsilon
        while True:
            try:
                cmd, payload = cmd_queue.get_nowait()
            except queue.Empty:
                break
            if cmd == "set_epsilon":
                epsilon = float(payload)
            elif cmd == "update_model":
                model.load_state_dict(payload)
                model.eval()

    while not stop_event.is_set():
        env.reset()
        obs = get_observation(env, state_mode=state_mode) # [ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]
        done = False
        total_reward = 0.0
        steps = 0

        while not done and not stop_event.is_set():
            _drain_commands()
            if random.random() < epsilon:
                action = int(np.random.randint(n_actions))
            else:
                with torch.no_grad():
                    obs_t = batch_states_to_tensor((obs,), state_mode=state_mode, device=torch.device("cpu"))
                    q_values = model(obs_t)
                    action = int(torch.argmax(q_values, dim=1).item())

            step_state, reward, is_terminal, is_truncated = env.step(action)
            next_obs = get_observation(env, state_mode=state_mode, step_state=step_state)
            done = bool(is_terminal or is_truncated)

            pushed = _queue_put_with_stop(
                data_queue,
                ("transition", obs, action, float(reward), next_obs, float(done)),
                stop_event=stop_event,
            )
            if not pushed:
                break

            obs = next_obs
            total_reward += float(reward)
            steps += 1

        if stop_event.is_set():
            break
        _queue_put_with_stop(data_queue, ("episode_end", float(total_reward), int(steps)), stop_event=stop_event)


def train(
    run_dir: str = DEFAULT_RUN_DIR,
    state_mode: str = DEFAULT_STATE_MODE,
    n_actions: int = DEFAULT_N_ACTIONS,
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    n_episodes: int = DEFAULT_N_EPISODES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    epsilon_min: float = DEFAULT_EPSILON_MIN,
    epsilon_decay: float = DEFAULT_EPSILON_DECAY,
    tau: float = DEFAULT_TAU,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    n_replay_buffer: int = DEFAULT_REPLAY_BUFFER_SIZE,
    n_warmup_episodes: int = DEFAULT_WARMUP_EPISODES,
    n_learn_updates: int = DEFAULT_LEARN_UPDATES,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    render_mode: str = DEFAULT_RENDER_MODE,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    log_every: int = DEFAULT_LOG_EVERY,
    num_collectors: int = DEFAULT_NUM_COLLECTORS,
    collector_queue_size: int = DEFAULT_COLLECTOR_QUEUE_SIZE,
    collector_sync_updates: int = DEFAULT_COLLECTOR_SYNC_UPDATES,
) -> Dict[str, object]:
    
    if state_mode not in STATE_CHOICES:
        raise ValueError(f"state_mode must be one of {STATE_CHOICES}")
    if n_actions <= 1:
        raise ValueError("n_actions must be > 1")

    set_seeds(seed)
    dev = resolve_device(device)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    online_model = build_q_network(state_mode=state_mode, n_actions=n_actions).to(dev)
    target_model = build_q_network(state_mode=state_mode, n_actions=n_actions).to(dev)
    target_model.load_state_dict(online_model.state_dict())
    optimizer = torch.optim.AdamW(online_model.parameters(), lr=lr, weight_decay=weight_decay)
    replay_buffer = ReplayBuffer(capacity=n_replay_buffer, state_mode=state_mode)

    episode_history: List[EpisodeMetric] = []
    best_reward = float("-inf")
    epsilon_value = float(epsilon)
    global_steps = 0
    total_updates = 0
    if num_collectors <= 1:
        env = Hw2Env(n_actions=n_actions, render_mode=render_mode)
        env._max_timesteps = int(max_timesteps)

        for episode in tqdm(range(1, n_episodes + 1), desc="episodes", leave=True):
            env.reset()
            obs = get_observation(env, state_mode=state_mode) #[ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]
            done = False
            total_reward = 0.0
            episode_steps = 0
            losses: List[float] = []

            while not done:
                if episode <= n_warmup_episodes:
                    action = int(np.random.randint(n_actions))
                else:
                    epsilon_value = epsilon_by_step(
                        step=global_steps,
                        eps_start=epsilon,
                        eps_end=epsilon_min,
                        eps_decay=epsilon_decay,
                    )
                    action = epsilon_greedy_action(
                        model=online_model,
                        obs=obs,
                        epsilon=epsilon_value,
                        n_actions=n_actions,
                        state_mode=state_mode,
                        device=dev,
                    )

                step_state, reward, is_terminal, is_truncated = env.step(action)
                next_obs = get_observation(env, state_mode=state_mode, step_state=step_state)
                done = bool(is_terminal or is_truncated)

                replay_buffer.append(obs, int(action), float(reward), next_obs, bool(done))
                obs = next_obs

                total_reward += float(reward)
                episode_steps += 1
                global_steps += 1

                if episode > n_warmup_episodes and len(replay_buffer) >= batch_size:
                    for _ in range(max(1, n_learn_updates)):
                        loss = optimize_dqn(
                            online_model=online_model,
                            target_model=target_model,
                            optimizer=optimizer,
                            replay_buffer=replay_buffer,
                            batch_size=batch_size,
                            gamma=gamma,
                            tau=tau,
                            grad_clip=grad_clip,
                            device=dev,
                        )
                        if loss is not None:
                            losses.append(loss)
                            total_updates += 1

            reward_per_step = total_reward / max(1, episode_steps)
            mean_loss: Optional[float] = float(np.mean(losses)) if len(losses) > 0 else None
            episode_history.append(
                {
                    "episode": episode,
                    "total_reward": float(total_reward),
                    "reward_per_step": float(reward_per_step),
                    "steps": float(episode_steps),
                    "epsilon": float(epsilon_value),
                    "mean_loss": float(mean_loss) if mean_loss is not None else None,
                    "replay_size": float(len(replay_buffer)),
                }
            )

            if total_reward > best_reward:
                best_reward = float(total_reward)
                torch.save(online_model.state_dict(), out_dir / "best.pt")

            if episode == 1 or episode % max(1, log_every) == 0 or episode == n_episodes:
                recent = episode_history[-min(50, len(episode_history)):]
                recent_reward = float(np.mean([x["total_reward"] for x in recent]))
                recent_rps = float(np.mean([x["reward_per_step"] for x in recent]))
                print(
                    f"[DQN] episode={episode}/{n_episodes} reward={total_reward:.4f} rps={reward_per_step:.4f} "
                    f"eps={epsilon_value:.4f} replay={len(replay_buffer)} recent50_reward={recent_reward:.4f} "
                    f"recent50_rps={recent_rps:.4f}"
                )
    else:
        if collector_queue_size < 100:
            raise ValueError("collector_queue_size must be >= 100 for parallel collection.")
        if collector_sync_updates < 1:
            raise ValueError("collector_sync_updates must be >= 1")

        print(f"[DQN] parallel collection enabled with num_collectors={num_collectors}")
        ctx = mp.get_context("spawn")
        data_queue = ctx.Queue(maxsize=int(collector_queue_size))
        stop_event = ctx.Event()
        cmd_queues = []
        procs = []
        recent_losses: Deque[float] = deque(maxlen=512)
        episodes_done = 0
        transitions_seen = 0

        try:
            for worker_id in range(int(num_collectors)):
                cmd_q = ctx.Queue(maxsize=2)
                p = ctx.Process(
                    target=_collector_worker,
                    args=(
                        worker_id,
                        state_mode,
                        n_actions,
                        max_timesteps,
                        seed,
                        render_mode,
                        data_queue,
                        cmd_q,
                        stop_event,
                    ),
                )
                p.start()
                cmd_queues.append(cmd_q)
                procs.append(p)

            init_weights = _model_state_to_cpu(online_model)
            for cmd_q in cmd_queues:
                _queue_replace_latest(cmd_q, ("update_model", init_weights))
                _queue_replace_latest(cmd_q, ("set_epsilon", epsilon_value))

            pbar = tqdm(total=n_episodes, desc="episodes", leave=True)
            while episodes_done < n_episodes:
                try:
                    item = data_queue.get(timeout=0.02)
                except queue.Empty:
                    item = None

                if item is not None:
                    if item[0] == "transition":
                        _, obs, action, reward, next_obs, done = item
                        replay_buffer.append(obs, int(action), float(reward), next_obs, bool(done))
                        transitions_seen += 1
                    elif item[0] == "episode_end":
                        _, total_reward, episode_steps = item
                        episodes_done += 1
                        pbar.update(1)
                        if episodes_done > n_warmup_episodes:
                            epsilon_value = epsilon_by_step(
                                step=transitions_seen,
                                eps_start=epsilon,
                                eps_end=epsilon_min,
                                eps_decay=epsilon_decay,
                            )
                            for cmd_q in cmd_queues:
                                _queue_replace_latest(cmd_q, ("set_epsilon", epsilon_value))

                        reward_per_step = float(total_reward) / max(1, int(episode_steps))
                        mean_loss: Optional[float] = (
                            float(np.mean(recent_losses)) if len(recent_losses) > 0 else None
                        )
                        episode_history.append(
                            {
                                "episode": episodes_done,
                                "total_reward": float(total_reward),
                                "reward_per_step": float(reward_per_step),
                                "steps": float(episode_steps),
                                "epsilon": float(epsilon_value),
                                "mean_loss": float(mean_loss) if mean_loss is not None else None,
                                "replay_size": float(len(replay_buffer)),
                            }
                        )

                        if float(total_reward) > best_reward:
                            best_reward = float(total_reward)
                            torch.save(online_model.state_dict(), out_dir / "best.pt")

                        if (
                            episodes_done == 1
                            or episodes_done % max(1, log_every) == 0
                            or episodes_done == n_episodes
                        ):
                            recent = episode_history[-min(50, len(episode_history)):]
                            recent_reward = float(np.mean([x["total_reward"] for x in recent]))
                            recent_rps = float(np.mean([x["reward_per_step"] for x in recent]))
                            print(
                                f"[DQN] episode={episodes_done}/{n_episodes} reward={float(total_reward):.4f} "
                                f"rps={reward_per_step:.4f} eps={epsilon_value:.4f} replay={len(replay_buffer)} "
                                f"recent50_reward={recent_reward:.4f} recent50_rps={recent_rps:.4f} "
                                f"transitions={transitions_seen}"
                            )

                if episodes_done > n_warmup_episodes and len(replay_buffer) >= batch_size:
                    for _ in range(max(1, n_learn_updates)):
                        loss = optimize_dqn(
                            online_model=online_model,
                            target_model=target_model,
                            optimizer=optimizer,
                            replay_buffer=replay_buffer,
                            batch_size=batch_size,
                            gamma=gamma,
                            tau=tau,
                            grad_clip=grad_clip,
                            device=dev,
                        )
                        if loss is not None:
                            recent_losses.append(loss)
                            total_updates += 1
                            if total_updates % collector_sync_updates == 0:
                                weights = _model_state_to_cpu(online_model)
                                for cmd_q in cmd_queues:
                                    _queue_replace_latest(cmd_q, ("update_model", weights))
            pbar.close()
        finally:
            stop_event.set()
            for p in procs:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2.0)

    torch.save(online_model.state_dict(), out_dir / "last.pt")
    save_training_plots(history=episode_history, out_dir=out_dir)

    final_summary = {
        "best_reward": float(best_reward),
        "final_epsilon": float(epsilon_value),
        "episodes": int(n_episodes),
        "total_updates": int(total_updates),
        "state_mode": state_mode,
        "n_actions": int(n_actions),
    }
    metrics = {
        "config": {
            "state_mode": state_mode,
            "n_actions": n_actions,
            "max_timesteps": max_timesteps,
            "n_episodes": n_episodes,
            "batch_size": batch_size,
            "gamma": gamma,
            "epsilon": epsilon,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "tau": tau,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_replay_buffer": n_replay_buffer,
            "n_warmup_episodes": n_warmup_episodes,
            "n_learn_updates": n_learn_updates,
            "seed": seed,
            "device": str(dev),
            "render_mode": render_mode,
            "grad_clip": grad_clip,
            "num_collectors": num_collectors,
            "collector_queue_size": collector_queue_size,
            "collector_sync_updates": collector_sync_updates,
        },
        "summary": final_summary,
        "history": episode_history,
    }
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[DQN] training completed. Best checkpoint: best.pt")
    return final_summary


def evaluate_policy(
    model: nn.Module,
    env: Hw2Env,
    state_mode: str,
    n_actions: int,
    n_eval_episodes: int,
    epsilon: float,
    device: torch.device,
) -> List[Dict[str, float]]:
    model.eval()
    history: List[Dict[str, float]] = []

    for episode in tqdm(range(1, n_eval_episodes + 1), desc="test", leave=False):
        env.reset()
        obs = get_observation(env, state_mode=state_mode)
        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done:
            action = epsilon_greedy_action(
                model=model,
                obs=obs,
                epsilon=epsilon,
                n_actions=n_actions,
                state_mode=state_mode,
                device=device,
            )
            step_state, reward, is_terminal, is_truncated = env.step(action)
            obs = get_observation(env, state_mode=state_mode, step_state=step_state)
            done = bool(is_terminal or is_truncated)
            success = success or bool(is_terminal)
            total_reward += float(reward)
            steps += 1

        history.append(
            {
                "episode": float(episode),
                "total_reward": float(total_reward),
                "reward_per_step": float(total_reward / max(1, steps)),
                "steps": float(steps),
                "success": float(1.0 if success else 0.0),
            }
        )

    return history


def test(
    checkpoint_path: str = f"{DEFAULT_RUN_DIR}/best.pt",
    run_dir: str = DEFAULT_RUN_DIR,
    state_mode: str = DEFAULT_STATE_MODE,
    n_actions: int = DEFAULT_N_ACTIONS,
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    n_eval_episodes: int = DEFAULT_EVAL_EPISODES,
    epsilon: float = DEFAULT_EVAL_EPSILON,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    render_mode: str = DEFAULT_RENDER_MODE,
) -> Dict[str, float]:
    if state_mode not in STATE_CHOICES:
        raise ValueError(f"state_mode must be one of {STATE_CHOICES}")

    set_seeds(seed)
    dev = resolve_device(device)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_existing_path(checkpoint_path)

    model = build_q_network(state_mode=state_mode, n_actions=n_actions).to(dev)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))

    env = Hw2Env(n_actions=n_actions, render_mode=render_mode)
    env._max_timesteps = int(max_timesteps)

    history = evaluate_policy(
        model=model,
        env=env,
        state_mode=state_mode,
        n_actions=n_actions,
        n_eval_episodes=n_eval_episodes,
        epsilon=epsilon,
        device=dev,
    )

    rewards = [x["total_reward"] for x in history]
    rps_vals = [x["reward_per_step"] for x in history]
    steps_vals = [x["steps"] for x in history]
    success_vals = [x["success"] for x in history]

    summary = {
        "mean_total_reward": float(np.mean(rewards)),
        "std_total_reward": float(np.std(rewards)),
        "mean_reward_per_step": float(np.mean(rps_vals)),
        "std_reward_per_step": float(np.std(rps_vals)),
        "mean_steps": float(np.mean(steps_vals)),
        "success_rate": float(np.mean(success_vals)),
        "episodes": int(n_eval_episodes),
        "epsilon": float(epsilon),
    }
    print(f"[DQN] test summary: {summary}")

    with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "history": history,
                "config": {
                    "checkpoint_path": str(ckpt_path),
                    "state_mode": state_mode,
                    "n_actions": n_actions,
                    "max_timesteps": max_timesteps,
                    "n_eval_episodes": n_eval_episodes,
                    "seed": seed,
                    "device": str(dev),
                    "render_mode": render_mode,
                },
            },
            f,
            indent=2,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework 2 - DQN training and evaluation.")
    sub = parser.add_subparsers(dest="command", required=False)

    p_train = sub.add_parser(CMD_TRAIN)
    p_train.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    p_train.add_argument("--state-mode", type=str, default=DEFAULT_STATE_MODE, choices=STATE_CHOICES)
    p_train.add_argument("--n-actions", type=int, default=DEFAULT_N_ACTIONS)
    p_train.add_argument("--max-timesteps", type=int, default=DEFAULT_MAX_TIMESTEPS)
    p_train.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_train.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p_train.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    p_train.add_argument("--epsilon-min", type=float, default=DEFAULT_EPSILON_MIN)
    p_train.add_argument("--epsilon-decay", type=float, default=DEFAULT_EPSILON_DECAY)
    p_train.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_train.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p_train.add_argument("--n-replay-buffer", type=int, default=DEFAULT_REPLAY_BUFFER_SIZE)
    p_train.add_argument("--n-warmup-episodes", type=int, default=DEFAULT_WARMUP_EPISODES)
    p_train.add_argument("--n-learn-updates", type=int, default=DEFAULT_LEARN_UPDATES)
    p_train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_train.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p_train.add_argument("--render-mode", type=str, default=DEFAULT_RENDER_MODE, choices=["offscreen", "gui"])
    p_train.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    p_train.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    p_train.add_argument("--num-collectors", type=int, default=DEFAULT_NUM_COLLECTORS)
    p_train.add_argument("--collector-queue-size", type=int, default=DEFAULT_COLLECTOR_QUEUE_SIZE)
    p_train.add_argument("--collector-sync-updates", type=int, default=DEFAULT_COLLECTOR_SYNC_UPDATES)

    p_test = sub.add_parser(CMD_TEST)
    p_test.add_argument("--checkpoint-path", type=str, default=f"{DEFAULT_RUN_DIR}/best.pt")
    p_test.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    p_test.add_argument("--state-mode", type=str, default=DEFAULT_STATE_MODE, choices=STATE_CHOICES)
    p_test.add_argument("--n-actions", type=int, default=DEFAULT_N_ACTIONS)
    p_test.add_argument("--max-timesteps", type=int, default=DEFAULT_MAX_TIMESTEPS)
    p_test.add_argument("--n-eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    p_test.add_argument("--epsilon", type=float, default=DEFAULT_EVAL_EPSILON)
    p_test.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_test.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p_test.add_argument("--render-mode", type=str, default=DEFAULT_RENDER_MODE, choices=["offscreen", "gui"])

    args = parser.parse_args()
    if args.command is None:
        args.command = DEFAULT_COMMAND

    if args.command == CMD_TRAIN:
        train(
            run_dir=getattr(args, "run_dir", DEFAULT_RUN_DIR),
            state_mode=getattr(args, "state_mode", DEFAULT_STATE_MODE),
            n_actions=getattr(args, "n_actions", DEFAULT_N_ACTIONS),
            max_timesteps=getattr(args, "max_timesteps", DEFAULT_MAX_TIMESTEPS),
            n_episodes=getattr(args, "n_episodes", DEFAULT_N_EPISODES),
            batch_size=getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
            gamma=getattr(args, "gamma", DEFAULT_GAMMA),
            epsilon=getattr(args, "epsilon", DEFAULT_EPSILON),
            epsilon_min=getattr(args, "epsilon_min", DEFAULT_EPSILON_MIN),
            epsilon_decay=getattr(args, "epsilon_decay", DEFAULT_EPSILON_DECAY),
            tau=getattr(args, "tau", DEFAULT_TAU),
            lr=getattr(args, "lr", DEFAULT_LR),
            weight_decay=getattr(args, "weight_decay", DEFAULT_WEIGHT_DECAY),
            n_replay_buffer=getattr(args, "n_replay_buffer", DEFAULT_REPLAY_BUFFER_SIZE),
            n_warmup_episodes=getattr(args, "n_warmup_episodes", DEFAULT_WARMUP_EPISODES),
            n_learn_updates=getattr(args, "n_learn_updates", DEFAULT_LEARN_UPDATES),
            seed=getattr(args, "seed", DEFAULT_SEED),
            device=getattr(args, "device", DEFAULT_DEVICE),
            render_mode=getattr(args, "render_mode", DEFAULT_RENDER_MODE),
            grad_clip=getattr(args, "grad_clip", DEFAULT_GRAD_CLIP),
            log_every=getattr(args, "log_every", DEFAULT_LOG_EVERY),
            num_collectors=getattr(args, "num_collectors", DEFAULT_NUM_COLLECTORS),
            collector_queue_size=getattr(args, "collector_queue_size", DEFAULT_COLLECTOR_QUEUE_SIZE),
            collector_sync_updates=getattr(args, "collector_sync_updates", DEFAULT_COLLECTOR_SYNC_UPDATES),
        )
    else:
        test(
            checkpoint_path=getattr(args, "checkpoint_path", f"{DEFAULT_RUN_DIR}/best.pt"),
            run_dir=getattr(args, "run_dir", DEFAULT_RUN_DIR),
            state_mode=getattr(args, "state_mode", DEFAULT_STATE_MODE),
            n_actions=getattr(args, "n_actions", DEFAULT_N_ACTIONS),
            max_timesteps=getattr(args, "max_timesteps", DEFAULT_MAX_TIMESTEPS),
            n_eval_episodes=getattr(args, "n_eval_episodes", DEFAULT_EVAL_EPISODES),
            epsilon=getattr(args, "epsilon", DEFAULT_EVAL_EPSILON),
            seed=getattr(args, "seed", DEFAULT_SEED),
            device=getattr(args, "device", DEFAULT_DEVICE),
            render_mode=getattr(args, "render_mode", DEFAULT_RENDER_MODE),
        )


if __name__ == "__main__":
    main()
