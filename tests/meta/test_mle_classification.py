import pytest
from meta_learning_framework.meta import MetaLearningEnsemble


class TestMetaLearningEnsembleClassification:
    def test_fit_base_models(
        self, mle_classification_init_params, classification_data_sample
    ):

        X, y = classification_data_sample

        mle = MetaLearningEnsemble(**mle_classification_init_params)
        mle.fit_base_models(X, y)

        # use predict method to test whether base models are actually fitted
        for bm in mle.base_models:
            bm.predict(X)

        assert True

    def test_fit_method(
        self, mle_classification_init_params, classification_data_sample
    ):

        X, y = classification_data_sample

        mle = MetaLearningEnsemble(**mle_classification_init_params)
        mle.fit(X, y)

        assert True

    def test_predict_method(
        self, mle_classification_init_params, classification_data_sample
    ):

        X, y = classification_data_sample

        mle = MetaLearningEnsemble(**mle_classification_init_params)
        mle.fit(X, y)
        print(mle.predict(X))

        assert True
