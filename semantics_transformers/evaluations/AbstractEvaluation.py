# -*- coding: utf-8 -*-
# @DateTime :2021/3/2
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod


class AbstractEvaluation(ABC):
    @abstractmethod
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        pass
