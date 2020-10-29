from metamodel import MetaLearningModel
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

X = load_iris().data
y = load_iris().target

model = MetaLearningModel(DecisionTreeClassifier(), [KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=7), KNeighborsClassifier(n_neighbors=11)],
                          'classification', 'score')

model.fit(X, y, 10)
print(model.predict(X))
