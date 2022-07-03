from abc import ABC, abstractmethod

from meta_learning_framework.types import Targets


class ErrorMeasurer(ABC):
    @abstractmethod
    def measure(self, base_preds: Targets, y_true: Targets) -> Targets:
        pass