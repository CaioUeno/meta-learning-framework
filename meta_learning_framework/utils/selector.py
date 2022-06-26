from abc import ABC, abstractmethod

from meta_learning_framework.types import Target, Targets


class Selector(ABC):
    @abstractmethod
    def select(self, base_preds: Targets) -> Target:
        pass