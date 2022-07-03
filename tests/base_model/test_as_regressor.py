import numpy as np
import pytest
from sklearn.dummy import DummyRegressor


class TestBaseModelAsRegressor:

    """
    Class to aggregate unit tests for the BaseModel class when implemented as a regressor.
    """

    def test_name_validation(self, concrete_base_model_class):

        objs = [-21, 0.7, {"name": "dummy"}, ["dummy"], {"dummy"}]

        for obj in objs:
            with pytest.raises(
                TypeError, match=r"Parameter name must be of type string*"
            ):
                concrete_base_model_class(
                    name=obj, model=DummyRegressor(strategy="mean")
                )

    def test_expected_name_as_str(self, concrete_base_model_class):

        clf = concrete_base_model_class(
            name="base_regressor", model=DummyRegressor(strategy="mean")
        )

        assert clf.name == "base_regressor"

    # this test *is not* implemented because depends on the user's implementation
    # def test_fit_method(self, ...)

    def test_predict_method(self, concrete_base_model_class, regression_sample):

        X, y = regression_sample
        rgr = concrete_base_model_class(
            name="rgr", model=DummyRegressor(strategy="mean")
        )
        rgr.fit(X, y)
        y_pred = rgr.predict(X)

        assert len(y_pred) == len(X)
        assert isinstance(y_pred, np.ndarray)
