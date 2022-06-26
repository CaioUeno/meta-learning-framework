from abc import ABC, abstractmethod

from meta_learning_framework.types import Target, Targets


class Combiner(ABC):
    @abstractmethod
    def combine(self, base_preds: Targets) -> Target:
        pass
