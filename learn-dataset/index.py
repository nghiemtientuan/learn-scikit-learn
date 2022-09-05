from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)
model = DecisionTreeClassifier()
my_model = model.fit(x_train, y_train)
x_new = np.array([[5.9, 4.0, 3.2, 1.0]])

print('score', my_model.score(x_test, y_test))
print('x_new', my_model.predict(x_new))
