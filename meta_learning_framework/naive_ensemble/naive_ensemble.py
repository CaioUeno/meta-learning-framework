from typing import List

import numpy as np
from meta_learning_framework.base_model.base_model import BaseModel
from meta_learning_framework.types import Instances, Targets
from meta_learning_framework.utils.combiner import Combiner


class NaiveEnsemble:
    def __init__(self, base_models: List[BaseModel], combiner: Combiner) -> None:

        self.base_models = base_models
        self.combiner = combiner

    def fit(self, X: Instances, y: Targets):

        for bm in self.base_models:
            bm.fit(X, y)

    def predict(self, X: Instances):

        predictions = np.empty(len(X), len(self.base_models))
        for idx, bm in enumerate(self.base_models):
            predictions[:, idx] = bm.predict(X)

        
