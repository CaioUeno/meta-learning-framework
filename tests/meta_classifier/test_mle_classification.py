import pytest
from meta_learning_framework.exceptions import InheritanceError
from meta_learning_framework.meta_classifier import MetaLearningEnsemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


class TestMetaLearningEnsembleClassification:
    def test_meta_classifier_validation(
        self,
        useless_class,
        concrete_base_model_class,
        concrete_combiner,
        concrete_error_measurer,
        concrete_selector,
    ):

        with pytest.raises(InheritanceError):

            MetaLearningEnsemble(
                meta_classifier=useless_class(),
                base_models=[
                    concrete_base_model_class(
                        name="Decision Tree", model=DecisionTreeClassifier()
                    ),
                    concrete_base_model_class(
                        name="Nearest Neighbors", model=KNeighborsClassifier()
                    ),
                    concrete_base_model_class(name="Naive Bayes", model=GaussianNB()),
                ],
                combiner=concrete_combiner(),
                error_measurer=concrete_error_measurer(),
                selector=concrete_selector(),
            )

    # def test_meta_classifier_init_expected(
    #     self,
    #     concrete_meta_clasifier_class,
    #     concrete_base_model_class,
    #     concrete_combiner,
    #     concrete_error_measurer,
    #     concrete_selector,
    # ):
    #     MetaLearningEnsemble(
    #         meta_classifier=concrete_meta_clasifier_class(
    #             model=KNeighborsClassifier(n_neighbors=3)
    #         ),
    #         base_models=[
    #             concrete_base_model_class(
    #                 name="Decision Tree", model=DecisionTreeClassifier()
    #             ),
    #             concrete_base_model_class(
    #                 name="Nearest Neighbors", model=KNeighborsClassifier()
    #             ),
    #             concrete_base_model_class(name="Naive Bayes", model=GaussianNB()),
    #         ],
    #         combiner=concrete_combiner(),
    #         error_measurer=concrete_error_measurer(),
    #         selector=concrete_selector(),
    #     )
    #     assert True
