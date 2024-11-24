import torch
import torch.nn.functional as F
from common import math
from common.scale import RunningScale
from common.world_model import WorldModel

class BMPC:
    """
    BMPC agent. Implements training + inference.
    """

    def __init__(self, cfg):
        self.device = torch.device(cfg.device)
        
        # Heuristic for large action spaces
        cfg.iterations += 2 * int(
            cfg.action_dim >= 20
        )          
        self.cfg = cfg
        self.model = WorldModel(cfg).to(self.device)
                    
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Vs.parameters() if \
                    self.cfg.use_v_instead_q else self.model._Qs.parameters()},
                {
                    "params": (
                        self.model._task_emb.parameters() if self.cfg.multitask else []
                    )
                },
            ],
            lr=self.cfg.lr,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
        )
        self.model.eval()
        self.scale = RunningScale(cfg, min_scale=self.cfg.pi_loss_min_scale)
        self.discount = (
            torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
                device=self.device,
            )
            if self.cfg.multitask
            else self._get_discount(cfg.episode_length)
        )
        self._prev_reanalyze_std = -1 # for log


    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
        )

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=torch.device(self.cfg.device))
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None, return_stats=False):
        """
        Select an action by planning in the latent space of the world model.

        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).
            return_stats (bool): Whether to return the action value stats during planning
        Returns:
            torch.Tensor: Action to take in the environment.
            List: action value stats during planning.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        
        if self.cfg.mpc:
            a, value, action_dist, value_stats = self.plan(
                z,
                t0=t0,
                eval_mode=eval_mode,
                task=task,
                return_stats=return_stats
            )
            return a.squeeze(0).cpu(), value.squeeze().cpu(), \
                action_dist.squeeze(0).cpu(), value_stats
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
            return a.cpu(), None, None, [0,0]

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, horizon=None):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        if horizon is None:
            horizon = self.cfg.horizon
        G, discount = 0, 1
        for t in range(horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G += discount * reward
            discount *= (
                self.discount[torch.tensor(task)]
                if self.cfg.multitask
                else self.discount
            )
        if self.cfg.use_v_instead_q:
            return G + discount * self.model.Q(z, None, task, return_type="avg")
        else:
            a = self.model.pi(z, task)[1]
            return G + discount * self.model.Q(z, a, task, return_type="avg")

    # Batched MPPI
    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None, return_stats=False, \
        update_prev_mean=True, horizon=None, reanalyze=False):
        """
        Plan a batched sequence of actions using the learned world model.

        Args:
            z (torch.Tensor): Latent state from which to plan, the shape should be [batchsize, latent_dim].
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).
            return_stats (bool): Whether to return the action value stats during planning,
                return zero list if return_stats=False.
            update_prev_mean (bool): Whether to update self._prev_mean using mean,
                when batched plan in reanalyze, update_prev_mean should be False.
            horizon (int): Planning horizon, if not specify, use self.cfg.horizon.
            reanalyze (bool): Reanalyzing or not.
        Returns:
            torch.Tensor: Action to take in the environment.
            List: action value stats during planning.
        """
        
        horizon = self.cfg.horizon if horizon is None else horizon
        batchsize = z.shape[0]
        z = z.unsqueeze(1) # z.shape:(batchsize,1,latent_dim)
        
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                horizon,
                batchsize,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            _z = z.repeat(1, self.cfg.num_pi_trajs, 1)
            for t in range(horizon - 1):
                pi_actions[t] = self.model.pi(_z, task, expl=reanalyze)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task, expl=reanalyze)[1]
                        
        # Initialize state and parameters
        _z = z.repeat(1, self.cfg.num_samples, 1)
        mean = torch.zeros(horizon, batchsize, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std * torch.ones(
            horizon, batchsize, self.cfg.action_dim, device=self.device
        )
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            horizon,
            batchsize,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            actions[:, :, :self.cfg.num_pi_trajs] = pi_actions

        estimate_pi_values = return_stats
        # Iterate MPPI
        for i in range(self.cfg.iterations):
            # Sample actions from MPPI action distribution
            actions[:, :, self.cfg.num_pi_trajs:] = (
                mean.unsqueeze(2)
                + std.unsqueeze(2)
                * torch.randn(
                    horizon,
                    batchsize,
                    self.cfg.num_samples - self.cfg.num_pi_trajs,
                    self.cfg.action_dim,
                    device=std.device,
                )
            ).clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(_z, actions, task, horizon=horizon).nan_to_num_(0)
            if estimate_pi_values:
                value_pi = value[:,:self.cfg.num_pi_trajs,:].mean()
                estimate_pi_values = False
            elite_idxs = torch.topk(
                value, self.cfg.num_elites, dim=1
            ).indices
            elite_value = value.gather(1, elite_idxs)
            elite_idxs = elite_idxs.unsqueeze(0).expand(actions.shape[0],-1,-1,actions.shape[-1])
            elite_actions = actions.gather(2, elite_idxs)
            
            # Update parameters
            max_value = elite_value.max(1,keepdim=True)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(1,keepdim=True)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=2) / (
                score.sum(1).unsqueeze(0) + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - mean.unsqueeze(2)) ** 2, dim=2
                )
                / (score.sum(1).unsqueeze(0) + 1e-9)
            ).clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]
            
        # Select action
        # use torch.multinomial instead of np.random.choice
        expand_shape = list(elite_actions.shape)
        expand_shape[1] = -1
        expand_shape[2] = -1
        idx = score.squeeze(-1).multinomial(1)
        values = elite_value.gather(1, idx.view(batchsize,1,1)).squeeze(1)
        actions = elite_actions.gather(2, idx.view(1,batchsize,1,1).expand(*expand_shape)).squeeze(2)
        
        if update_prev_mean:
            self._prev_mean = mean
            
        action_dist = torch.cat([actions[0],std[0]], dim=-1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(batchsize, self.cfg.action_dim, device=std.device)
        value_stats = [value_pi.item(), max_value.item()] if return_stats else [0,0]
        return a.clamp_(-1, 1), values, action_dist, value_stats


    def compute_pi_loss(self, zs, expert_actions_dist, task):
        zs = zs[:-1]
        mus, pis, log_pis, log_stds = self.model.pi(zs, task)
        rho = torch.pow(
            self.cfg.rho, torch.arange(self.cfg.horizon, device=self.device)
        )
        actions_dist = torch.cat([mus, log_stds.exp()], dim=-1)
        pi_loss = math.kl_div(actions_dist, expert_actions_dist).mean(-1, keepdim=True)
        if self.cfg.pi_loss_norm:
            self.scale.update(pi_loss[0])
            pi_loss = self.scale(pi_loss)
            
        ent_loss = (log_pis.mean(dim=(1, 2)) * rho).mean()
        pi_loss = pi_loss.mean(dim=(1, 2)) * rho
        pi_loss = pi_loss.mean()
        pi_loss = pi_loss + self.cfg.entropy_coef * ent_loss
        
        pi_metrics = {
            "pi_loss": pi_loss.item(),
            "ent_loss": ent_loss.item(),
            "pi_log_std": log_stds.mean().item(),
            "pi_scale": float(self.scale.value),
        }
        return pi_loss, pi_metrics


    @torch.no_grad()
    def _td_target_Q(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        discount = (
            self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        )
        target_type = "min" if self.cfg.min_td_target else "avg"
        pi = self.model.pi(next_z, task)[1]
        return reward + discount * self.model.Q(
            next_z, pi, task, return_type=target_type, target=True
        )
            
    @torch.no_grad()
    def _td_target_V(self, zs, task):
        Gs, discount = 0, 1
        zs_ = zs.clone()
        for _ in range(self.cfg.nstep_td_horizon):
            actions = self.model.pi(zs_, task)[1]
            rewards = math.two_hot_inv(self.model.reward(zs_, actions, task), self.cfg)
            zs_ = self.model.next(zs_, actions, task)
            Gs += discount * rewards
            discount *= (
                self.discount[torch.tensor(task)]
                if self.cfg.multitask
                else self.discount
            )
        td_target = Gs + discount * self.model.Q(zs_, None, task, return_type="avg", target=True)
        return td_target

    def update(self, buffer, reanalyze=False, pretrain=False):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer.

        Returns:
            dict: Dictionary of training statistics.
        """
        metrics = { "pi_loss": 0,
                    "ent_loss": 0,
                    "pi_log_std": 0,
                    "pi_scale":0,
                    "reanalyze_std":-1}        
        (obs, action, reward, task, _, _, expert_action_dist, _), info = buffer.sample()

        # preprocess expert actions
        expert_action_dist_ = expert_action_dist.clone()
        expert_action_mu, expert_action_std = expert_action_dist_.chunk(2, dim=-1)
        nan_idx = torch.isnan(expert_action_mu) # samples which are not generated by mpc policy
        expert_action_mu[nan_idx] = action[nan_idx]
        expert_action_std[nan_idx] = torch.ones_like(action[nan_idx])
        expert_action_dist_ = torch.cat([expert_action_mu,expert_action_std],dim=-1)
                
        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
        true_zs = torch.cat([z.unsqueeze(0), next_z[:-1]],dim=0).clone().detach() # latent from real obs
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t + 1] = z

        # Predictions
        _zs = zs[:-1]
        if self.cfg.use_v_instead_q:
            td_targets = self._td_target_V(true_zs, task)
            qs = self.model.Q(_zs, None, task, return_type="all")
        else:
            td_targets = self._td_target_Q(next_z, reward, task)
            qs = self.model.Q(_zs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        consistency_loss *= 1 / self.cfg.horizon
        reward_loss *= 1 / self.cfg.horizon
        value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)
                    
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        model_metrics = {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
        }
        metrics.update(model_metrics)

        # Update pi
        if not pretrain:
            pi_loss, pi_metrics = self.compute_pi_loss(zs.detach(), expert_action_dist_, task)
            self.pi_optim.zero_grad(set_to_none=True)
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model._pi.parameters(), self.cfg.grad_clip_norm
            )
            self.pi_optim.step()
            metrics.update(pi_metrics)
            
        # Update target Q-functions
        if self.cfg.use_v_instead_q:
            self.model.soft_update_target_V()
        else:
            self.model.soft_update_target_Q() 
        self.model.eval()

        # Lazy reanalyze
        if reanalyze:
            _, reanalyzed_actions_dist = self._reanalyze(true_zs, task, info, buffer)
            _,std = reanalyzed_actions_dist.chunk(2,dim=-1)
            self._prev_reanalyze_std = std.mean().item()
        metrics.update({"reanalyze_std":self._prev_reanalyze_std})
        return metrics
    
    @torch.no_grad()
    def _reanalyze(self, zs, task, info, buffer):
        '''
        do re-planning
        '''
        self.model.eval()
            
        # reanalyze
        z_ = zs[:,:self.cfg.reanalyze_batch_size,:].reshape(-1, self.cfg.latent_dim)
        reanalyzed_actions, reanalyzed_values, reanalyzed_actions_dist, _ = self.plan(\
                z_, t0=True, task=task, update_prev_mean=False, horizon=self.cfg.reanalyze_horizon, reanalyze=True)
        
        # update reanalyzed data to buffer
        index_list = info['index'][1:, :self.cfg.reanalyze_batch_size].flatten().tolist()
        with buffer._buffer._replay_lock: # Add lock to prevent any unexpected data change due to thread risk
            buffer._buffer._storage._storage['expert_action'][index_list] = \
                reanalyzed_actions.to(buffer._buffer._storage.device)
            buffer._buffer._storage._storage['expert_action_dist'][index_list] = \
                reanalyzed_actions_dist.to(buffer._buffer._storage.device)
            buffer._buffer._storage._storage['expert_value'][index_list] = \
                reanalyzed_values.flatten().to(buffer._buffer._storage.device)

        self.model.train()
        return reanalyzed_actions.view(self.cfg.horizon, self.cfg.reanalyze_batch_size, -1), \
            reanalyzed_actions_dist.view(self.cfg.horizon, self.cfg.reanalyze_batch_size, -1)
