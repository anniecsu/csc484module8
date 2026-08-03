import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load data
diabetes = pd.read_csv("diabetes.csv")

# define features (x) and target (y)
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

# call train_test_split function to split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

# define and compile model
model = KNeighborsClassifier(n_neighbors=5)

# fit the model
model.fit(X_train, y_train)

# evaluate training accuracy
train_accuracy = model.score(X_train, y_train)
print(f"\nTraining Accuracy: {train_accuracy*100:.2f}%")

# evaluate testing accuracy
test_accuracy = model.score(X_test, y_test)
print(f"Testing Accuracy: {test_accuracy*100:.2f}%")

# make predictions
predictions = model.predict(X_test)

# print chart comparing predictions with actual results
print(f"\nSample predictions:")
print(f"{'Index':<8}{'Actual':<10}{'Predicted':<12}{'Result'}")
for i in range(15):
    actual = int(y_test.iloc[i])
    predicted = int(predictions[i])
    if actual == predicted:
        result = "Correct"
    else:
        result = "Incorrect"
    print(f"{i:<8}{actual:<10}{predicted:<12}{result}")

