from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, linewidth=120)

from src import cli
from src.checkpoint import CheckpointableData, Checkpointer
from src.config import BaseConfig, Require
from src.defaults import ROOT_DIR
from src.log import default_log as log
from src.shared import get_env
from src.sskp import N_EVAL_EPISODES, SSKP
from src.torch_util import device


SAVE_PERIOD = 5


class Config(BaseConfig):
    env_name = Require(str)
    seed = 1
    epochs = 1000
    alg_cfg = SSKP.Config()


def main(cfg):
    env_factory = lambda: get_env(cfg.env_name)
    data = CheckpointableData()
    algorithm = SSKP(cfg.alg_cfg, env_factory, data)
    algorithm.to(device)
    checkpointer = Checkpointer(algorithm, log.dir, 'sskp_ckpt_{}.pt')
    best_checkpointer = Checkpointer(algorithm, log.dir, 'sskp_best_ckpt.pt')
    data_checkpointer = Checkpointer(data, log.dir, 'sskp_data.pt')

    def evaluate_and_save_best():
        result = algorithm.evaluate()
        if algorithm.record_evaluation(result):
            best_checkpointer.save()
            log('Saved best SSkP checkpoint @ epoch {}: return {:.2f}, '
                'violations {}/{}'.format(
                    algorithm.best_eval_epoch.item(),
                    algorithm.best_eval_return.item(),
                    algorithm.best_eval_violations.item(),
                    N_EVAL_EPISODES,
                ))
        return result

    if data_checkpointer.try_load():
        loaded_epoch = checkpointer.load_latest(
            range(0, cfg.epochs + 1, SAVE_PERIOD)
        )
        if isinstance(loaded_epoch, int):
            assert loaded_epoch == algorithm.epochs_completed

    algorithm.setup()
    if algorithm.epochs_completed == 0:
        checkpointer.save(0)
        data_checkpointer.save()
        evaluate_and_save_best()
    while algorithm.epochs_completed < cfg.epochs:
        log('Beginning SSkP epoch {}'.format(algorithm.epochs_completed.item() + 1))
        algorithm.epoch()
        evaluate_and_save_best()
        if algorithm.epochs_completed % SAVE_PERIOD == 0:
            checkpointer.save(algorithm.epochs_completed.item())
            data_checkpointer.save()


if __name__ == '__main__':
    assert Path(ROOT_DIR).is_dir(), ROOT_DIR
    cli.main(Config(), main)
