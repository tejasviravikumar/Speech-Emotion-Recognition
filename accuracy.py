from preprocess import X_train,X_test, y_test , y_train , le
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(random_state=0,max_iter=2000).fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual labels:", le.inverse_transform(y_test))
print("Predicted labels:",le.inverse_transform(y_pred))

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)