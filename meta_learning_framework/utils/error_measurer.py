from abc import ABC, abstractmethod

from meta_learning_framework.types import Target, Targets


class ErrorMeasurer(ABC):
    @abstractmethod
    def measure(self, base_preds: Targets) -> Target:
        pass