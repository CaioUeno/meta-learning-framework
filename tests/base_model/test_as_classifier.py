import numpy as np
import pytest
from sklearn.dummy import DummyClassifier


class TestBaseModelAsClassifier:

    """
    Class to aggregate unit tests for the BaseModel class when implemented as a classifier.
    """

    def test_name_validation(self, concrete_base_model_class):

        objs = [-21, 0.7, {"name": "dummy"}, ["dummy"], {"dummy"}]

        for obj in objs:
            with pytest.raises(
                TypeError, match=r"Parameter name must be of type string*"
            ):
                concrete_base_model_class(
                    name=obj, model=DummyClassifier(strategy="most_frequent")
                )

    def test_expected_name_as_str(self, concrete_base_model_class):

        clf = concrete_base_model_class(
            name="base_classifier", model=DummyClassifier(strategy="most_frequent")
        )

        assert clf.name == "base_classifier"

    # this test *is not* implemented because depends on the user's implementation
    # def test_fit_method(self, ...)

    def test_predict_method(self, concrete_base_model_class, classification_sample):

        X, y = classification_sample
        clf = concrete_base_model_class(
            name="clf", model=DummyClassifier(strategy="most_frequent")
        )
        clf.fit(X, y)
        y_pred = clf.predict(X)

        assert len(y_pred) == len(X)
        assert isinstance(y_pred, np.ndarray)
