from typing import Any, List, Union

import numpy as np
from meta_learning_framework.base_model import BaseModel
from meta_learning_framework.exceptions import InheritanceError
from meta_learning_framework.meta_classifier import MetaClassifier
from meta_learning_framework.types import Instances, Targets
from meta_learning_framework.utils import Combiner, ErrorMeasurer, Selector
from sklearn.model_selection import BaseCrossValidator, cross_val_predict


class MetaLearningEnsemble:
    def __init__(
        self,
        meta_classifier: MetaClassifier,
        base_models: List[BaseModel],
        combiner: Combiner,
        error_measurer: ErrorMeasurer,
        selector: Selector,
    ) -> None:

        MetaLearningEnsemble.__validate_params__(
            meta_classifier, base_models, combiner, error_measurer, selector
        )

        self.meta_classifier = meta_classifier
        self.base_models = base_models
        self.combiner = combiner
        self.error_measurer = error_measurer
        self.selector = selector

    @staticmethod
    def __validate_params__(
        meta_classifier: Any,
        base_models: Any,
        combiner: Any,
        error_measurer: Any,
        selector: Any,
    ) -> None:

        MetaLearningEnsemble.__validate_meta_classifier__(meta_classifier)
        MetaLearningEnsemble.__validate_base_models__(base_models)

    def __validate_meta_classifier__(meta_classifier: Any) -> bool:

        parent_classes = [c.__name__ for c in meta_classifier.__class__.__bases__]

        if "MetaClassifier" not in parent_classes:
            raise InheritanceError(
                f"Meta classifier variable must inherit from MetaClassifier class."
            )

    def __validate_base_models__(base_models: Any):

        for bm in base_models:

            parent_classes = [c.__name__ for c in bm.__class__.__bases__]

            if "BaseModel" not in parent_classes:
                raise InheritanceError(
                    f"Base model variable must inherit from BaseModel class."
                )

    def __validate_combiner__(combiner: Any):

        parent_classes = [c.__name__ for c in combiner.__class__.__bases__]

        if "Combiner" not in parent_classes:
            raise InheritanceError(
                f"Combiner variable must inherit from Combiner class."
            )

    def __validate_c__(combiner: Any):

        parent_classes = [c.__name__ for c in combiner.__class__.__bases__]

        if "Combiner" not in parent_classes:
            raise InheritanceError(
                f"Combiner variable must inherit from Combiner class."
            )

    def __fit_base_models__(self, X: Instances, y: Targets) -> None:

        for bm in self.base_models:
            bm.fit(X, y)

    def __fit_meta_classifier__(self, X: Instances, y: Targets) -> None:

        self.meta_classifier.fit(X, y)

    def fit(
        self,
        X: Instances,
        y: Targets,
        cv: Union[int, BaseCrossValidator] = 10,
        verbose: bool = False,
    ) -> None:

        meta_y = np.empty(shape=(len(X), len(self.base_models)))
        for idx, bm in enumerate(self.base_models):
            meta_y[:, idx] = cross_val_predict(estimator=bm, X=X, y=y, cv=cv)

        self.__fit_base_models__(X, y)
        self.__fit_meta_classifier__(X, meta_y)

    def predict(self, X: Instances, verbose: bool = False) -> np.ndarray:

        # make it faster
        predictions = np.empty(len(X))
        for idx, x in enumerate(X):

            base_preds = np.asarray([bm.predict([x])[0] for bm in self.base_models])
            meta_pred = self.meta_classifier.predict([x])[0]

            if meta_pred.sum() > 0:
                predictions[idx] = self.combiner.combine(base_preds[meta_pred])
            else:
                predictions[idx] = self.combiner.combine(base_preds)

        return predictions
