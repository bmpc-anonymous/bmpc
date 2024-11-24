import os
import warnings
os.environ['MUJOCO_GL'] = 'off' # 'osmesa'/'egl'/'off'
os.environ['LAZY_LEGACY_OP'] = '0'

warnings.filterwarnings('ignore')
import torch
import hydra
from termcolor import colored
from common.parser import parse_cfg, save_cfg
from common.seed import set_seed

from envs import make_env

from common.buffer import Buffer
from bmpc import BMPC
from trainer.online_trainer import OnlineTrainer
from common.logger import TBLogger

torch.backends.cudnn.benchmark = True

# config_name specify the config file in {config_path}/{config_name}.yaml
@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
    """
    Script for training BMPC agents.

    Most relevant args:
        `task`: task name
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `steps`: number of training/environment steps (default: 10M)
        `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
        $ python train.py task=dog-run steps=500000
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.steps > 0, 'Must train for at least 1 step.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print('pid:', os.getpid())
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir, flush=True)

    trainer_cls = OnlineTrainer
    logger_cls = TBLogger
    trainer = trainer_cls(
        cfg=cfg,
        env=make_env(cfg),
        agent=BMPC(cfg),
        buffer=Buffer(cfg),
        logger=logger_cls(cfg),
    )
    save_cfg(cfg, cfg.work_dir) # save parsed config, must after the logger's init
    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
    train()
