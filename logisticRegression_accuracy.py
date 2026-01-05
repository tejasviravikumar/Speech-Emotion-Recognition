from preprocess import X_train, X_test, y_train, y_test, le , X , y
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression(max_iter=1000,random_state=0).fit(X,y)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print("Actual labels:", le.inverse_transform(y_test))
print("Predicted labels:", le.inverse_transform(y_pred))

acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

