import os
os.environ['MUJOCO_GL'] = 'osmesa' # 'osmesa'/'egl'/'off'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg, save_cfg
from common.seed import set_seed

# If you use torch>=2.4.0dev, you must do "import triton" before import make_env,
# or the code will crash sliently in condaenv/lib/python3.9/site-packages/triton/backends/nvidia/compiler.py line 2:
# "from triton._C.libtriton import ir, passes, llvm, nvidia"
import triton
# More precisely, if you import mujoco renderer through
# condaenv/lib/python3.9/site-packages/dm_control/_render/__init__.py line 87,
# and then do "import triton", the program will crash, if you do "import triton" first, it will be all good.
# Also, if you specify os.environ['MUJOCO_GL'] = 'off', which means do not import renderer, it will be all good.

from envs import make_env
from bmpc import BMPC
from common.logger import make_dir

torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a BMPC checkpoint.

	Most relevant args:
		`task`: task name
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
 
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print('pid:', os.getpid())
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']), flush=True)
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = BMPC(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	make_dir(cfg.work_dir)
	save_cfg(cfg, cfg.work_dir)
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes):
			print("Episode {} start.".format(i+1))
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:
				action, _, _, _ = agent.act(obs, t0=t==0, task=task_idx, eval_mode=True)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render())
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
			print("Episode {} end.".format(i+1))
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'), flush=True)
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
