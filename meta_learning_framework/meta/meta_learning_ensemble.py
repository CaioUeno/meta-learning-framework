from typing import Any, List, Union

import numpy as np
from meta_learning_framework.base_model import BaseModel
from meta_learning_framework.meta import MetaClassifier
from meta_learning_framework.types import Instances, Targets
from meta_learning_framework.utils import Combiner, ErrorMeasurer, Selector
from sklearn.model_selection import BaseCrossValidator, cross_val_predict


class MetaLearningEnsemble:

    """
    Meta learning ensembler. It organizes base models and the meta classifier objects.

    Parameters
    ----------
        meta_classifier : MetaClassifier
            Model to learn the best subset of models to use on the ensemble (instance level);
        base_models : List[BaseModel]
            List of available models;
        combiner : Combiner
            Object that implements a combine function (how to combine selected models' outputs);
        error_measurer : ErrorMeasurer
            Object that calculates the error between
        selector : Selector
            Ob
    """

    def __init__(
        self,
        meta_classifier: MetaClassifier,
        base_models: List[BaseModel],
        combiner: Combiner,
        error_measurer: ErrorMeasurer,
        selector: Selector,
    ) -> None:

        self.meta_classifier = meta_classifier
        self.base_models = base_models
        self.combiner = combiner
        self.error_measurer = error_measurer
        self.selector = selector

    def fit_base_models(self, X: Instances, y: Targets) -> None:

        for bm in self.base_models:
            bm.fit(X, y)

    def fit_meta_classifier(self, X: Instances, y: Targets) -> None:

        self.meta_classifier.fit(X, y)

    def fit(
        self,
        X: Instances,
        y: Targets,
        cv: Union[int, BaseCrossValidator] = 10,
        verbose: bool = False,
    ) -> None:

        cross_val_y = np.empty(shape=(len(X), len(self.base_models)))
        for idx, bm in enumerate(self.base_models):
            cross_val_y[:, idx] = cross_val_predict(estimator=bm, X=X, y=y, cv=cv)

        self.fit_base_models(X, y)

        meta_y = np.empty(shape=(len(X), len(self.base_models)))
        for idx, (y_true, y_pred) in enumerate(zip(y, cross_val_y)):

            errors = self.error_measurer.measure(y_true, y_pred)
            print(errors)
            print(self.selector.select(errors))
            meta_y[idx, :] = self.selector.select(errors)

        self.fit_meta_classifier(X, meta_y)

    def predict(self, X: Instances, verbose: bool = False) -> np.ndarray:

        # make it faster
        predictions = np.empty(len(X))
        for idx, x in enumerate(X):

            base_preds = np.asarray([bm.predict([x])[0] for bm in self.base_models])
            meta_pred = self.meta_classifier.predict([x])[0]

            if meta_pred.sum() > 0:
                predictions[idx] = self.combiner.combine(
                    base_preds[meta_pred.astype(bool)]
                )
            else:
                predictions[idx] = self.combiner.combine(base_preds)

        return predictions
