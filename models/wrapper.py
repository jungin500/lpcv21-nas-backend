import numpy as np
import torch
from abc import *


class Wrapper(metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, frame: np.ndarray) -> torch.Tensor:
        pass
