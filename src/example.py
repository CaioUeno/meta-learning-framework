from metamodel import MetaLearningModel
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

X = load_iris().data
y = load_iris().target

models = [KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=7), KNeighborsClassifier(n_neighbors=11)]

model = MetaLearningModel(DecisionTreeClassifier(), models,
                          'classification', 'score')

model.fit(X, y, 10)
print(model.predict(X))
print(model.n_meta_models)
