import numpy as np
import pytest


class TestBaseModel:
    def test_name_validation(
        self, concrete_base_model_class, not_str_objects, diff_tasks_models
    ):

        # iterate over classifier and regressor
        for dtm in diff_tasks_models:

            # iterate over different types
            for wrong_type_name in not_str_objects:
                with pytest.raises(TypeError):
                    concrete_base_model_class(name=wrong_type_name, model=dtm)

    def test_expected_name_as_str(self, concrete_base_model_class, diff_tasks_models):

        # iterate over classifier and regressor
        for dtm in diff_tasks_models:

            clf = concrete_base_model_class(name="string_name", model=dtm)

            assert clf.name == "string_name"

    # this test *is not* implemented because depends on the user's implementation
    # def test_fit_method(self, ...)

    def test_predict_method_output(
        self, concrete_base_model_class, data_sample, diff_tasks_models
    ):

        X, y = data_sample

        # iterate over classifier and regressor
        for dtm in diff_tasks_models:

            clf = concrete_base_model_class(name="clf", model=dtm)

            clf.fit(X, y)
            y_pred = clf.predict(X)

            assert len(y_pred) == len(X)
            assert isinstance(y_pred, np.ndarray)
