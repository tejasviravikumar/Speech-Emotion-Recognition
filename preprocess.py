import pandas as pd
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('ravdess_metadata.csv')
feature_cols = [f"mfcc_{i+1}" for i in range(30)] + \
               [f"sc_{i+1}" for i in range(7)] + \
               [f"chroma_{i+1}" for i in range(12)]

X = df[feature_cols].values

le = LabelEncoder()
y = le.fit_transform(df["Emotion"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

