#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 15.08.22
@author: leonhard.hennig@dfki.de
"""
import logging
from typing import Sequence

import hydra
from omegaconf import DictConfig, OmegaConf
from run_binary_relation_clf import main as sherlock_main

logger = logging.getLogger(__name__)


def hydra_to_argparse(cfg: DictConfig) -> Sequence[str]:
    """
    Convert a Hydra config object to a string that can be passed to argparse. Very hacky way to pass args to avoid
    conflating the Sherlock project with Hydra dependencies and rewriting run_binary_relation_clf.py
    """
    arg_str = []
    cfg["seed"] = cfg["seed"] * cfg["run_id"]
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.pop("run_id")
    resolved.pop("scenario_name")

    for (k, v) in resolved.items():
        if isinstance(v, bool):
            if v:
                arg_str.append(f"--{k}")
        else:
            arg_str.extend([f"--{k}", f"{v}"])
    return arg_str


@hydra.main(config_name="config", config_path="../../config", version_base="1.2")
def evaluate(cfg: DictConfig) -> None:
    """
    Conducts evaluation given the configuration.
    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    args = hydra_to_argparse(cfg)
    logger.debug(args)
    sherlock_main(args)


if __name__ == "__main__":
    evaluate()
