from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import pandas as pd

# Grab the dataset from scikit-learn
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print(X, y)

# Build and train the model
model = SVC(random_state=101)
model.fit(X_train, y_train)
print("Score on the training set is: {:2}".format(model.score(X_train, y_train)))
print("Score on the test set is: {:.2}".format(model.score(X_test, y_test)))

# Save the model
model_filename = 'iris.pkl'
print(f"Saving model to {model_filename}...")
joblib.dump(model, model_filename)