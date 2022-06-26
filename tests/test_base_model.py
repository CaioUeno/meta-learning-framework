class TestBaseModel:
    def test_init(self, base_classifier, sample):

        X, y = sample
        base_classifier.fit(X, y)

        assert True
