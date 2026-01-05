from preprocess import X_train, X_test, y_train, y_test, le
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm_clf = SVC(kernel='rbf', class_weight='balanced', C=10, gamma='scale')

svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

print("Actual labels:", le.inverse_transform(y_test))
print("Predicted labels:", le.inverse_transform(y_pred))

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
