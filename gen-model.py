from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
import pickle

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

# Split dataset into train and test
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3,
                     random_state=2018)


knn = KNN(n_neighbors=3)

# train model
knn.fit(X_train, y_train)

filename = '/home/sumit123/finalized_model.pkl'
pickle.dump(knn, open(filename, 'wb'))
