from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, linewidth=120)

from src import cli
from src.checkpoint import CheckpointableData, Checkpointer
from src.config import BaseConfig, Require
from src.defaults import ROOT_DIR
from src.log import default_log as log
from src.rpt import RPT
from src.shared import get_env
from src.torch_util import device


SAVE_PERIOD = 5


class Config(BaseConfig):
    env_name = Require(str)
    seed = 1
    epochs = 1000
    alg_cfg = RPT.Config()


def main(cfg):
    env_factory = lambda: get_env(cfg.env_name)
    data = CheckpointableData()
    algorithm = RPT(cfg.alg_cfg, env_factory, data)
    algorithm.to(device)
    checkpointer = Checkpointer(algorithm, log.dir, 'rpt_ckpt_{}.pt')
    data_checkpointer = Checkpointer(data, log.dir, 'rpt_data.pt')

    if data_checkpointer.try_load():
        loaded_epoch = checkpointer.load_latest(range(0, cfg.epochs, SAVE_PERIOD))
        if isinstance(loaded_epoch, int):
            assert loaded_epoch == algorithm.epochs_completed

    algorithm.setup()
    if algorithm.epochs_completed == 0:
        algorithm.evaluate()
    while algorithm.epochs_completed < cfg.epochs:
        log('Beginning RPT epoch {}'.format(algorithm.epochs_completed.item() + 1))
        algorithm.epoch()
        algorithm.evaluate()
        if algorithm.epochs_completed % SAVE_PERIOD == 0:
            checkpointer.save(algorithm.epochs_completed.item())
            data_checkpointer.save()


if __name__ == '__main__':
    assert Path(ROOT_DIR).is_dir(), ROOT_DIR
    cli.main(Config(), main)
