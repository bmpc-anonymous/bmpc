from time import time
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer

class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action, _, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None, expert_value=None, expert_action_dist=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        if expert_value is None:
            expert_value = torch.tensor(float('nan'))
        if expert_action_dist is None:
            expert_action_dist = torch.full(size=(2*self.cfg.action_dim,), fill_value=float('nan'))
        expert_action = torch.full_like(self.env.rand_act(), float('nan'))
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            expert_action=expert_action.unsqueeze(0),
            expert_value=expert_value.unsqueeze(0),
            expert_action_dist=expert_action_dist.unsqueeze(0),
        ), batch_size=(1,))
        return td

    def train(self):
        """Train agent."""
        train_metrics, done, eval_next = {}, True, True
        value_stats_list = [[], []] # [values_pi, values_mpc]
        self.reanalyze_count = 0
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")                    
                    eval_next = False
                if self._step > 0:
                    tds = torch.cat(self._tds)
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"],
                    )
                    train_metrics.update(self.common_metrics())
                    # log value difference histograms and action std
                    if len(value_stats_list[0]) > 0:
                        value_mpc_pi_diffs = np.array(value_stats_list[1]) - np.array(value_stats_list[0])
                        value_mpc_pi_diff = value_mpc_pi_diffs.mean()
                        train_metrics.update(
                            value_mpc_pi_diff=value_mpc_pi_diff,
                            value_mpc_pi_diffs=value_mpc_pi_diffs,
                        )
                        # log action std (ignore first action which is nan)
                        _,std = tds["expert_action_dist"][1:].chunk(2,dim=-1)
                        train_metrics.update({"act_std":std.mean().item()})
                        value_stats_list = [[], []]
                    self.logger.log(train_metrics, "train")
                    self._ep_idx = self.buffer.add(tds)
                obs = self.env.reset()
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step >= self.cfg.seed_steps:
                action, expert_value, expert_action_dist, value_stats = self.agent.act(
                    obs, t0=len(self._tds) == 1, return_stats=True
                )
                value_stats_list[0].append(value_stats[0])
                value_stats_list[1].append(value_stats[1])
            else:
                action = self.env.rand_act()
                expert_value, expert_action_dist = None, None
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward, expert_value, expert_action_dist))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...", flush=True)
                else:
                    num_updates = 1
                if num_updates == 1: # non-pretrain update
                    reanalyze = (self.cfg.reanalyze_interval > 0) and (self.reanalyze_count % self.cfg.reanalyze_interval == 0)
                    _train_metrics = self.agent.update(self.buffer, reanalyze=reanalyze, pretrain=False)
                    self.reanalyze_count += 1
                else: # pretrain update
                    for _ in range(num_updates):
                        _train_metrics = self.agent.update(self.buffer, reanalyze=False, pretrain=True)
                train_metrics.update(_train_metrics)

            self._step += 1

        self.logger.finish(self.agent)
