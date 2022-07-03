from abc import ABC, abstractmethod

from meta_learning_framework.types import Targets


class Selector(ABC):
    @abstractmethod
    def select(self, base_errors: Targets) -> Targets:
        pass