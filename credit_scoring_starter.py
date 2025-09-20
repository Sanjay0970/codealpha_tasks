# Credit Scoring Model — Starter Script
# Run: python credit_scoring_starter.py
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import joblib

DATA_PATH = 'credit_synthetic.csv'  # change to your CSV path if needed

df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)
print('Head:'); print(df.head())

target = 'default'
X = df.drop(columns=[target])
y = df[target]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                     ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocess = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols),
                                             ('cat', categorical_transformer, categorical_cols)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=400, class_weight='balanced', random_state=42),
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=300, random_state=42)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, clf in models.items():
    pipe = Pipeline(steps=[('preprocess', preprocess), ('clf', clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f'{name}: ROC-AUC CV = {scores.mean():.3f} ± {scores.std():.3f}')

# Fit final model
pipe = Pipeline(steps=[('preprocess', preprocess),
                      ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=400, random_state=42))])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print('Test ROC-AUC:', roc_auc_score(y_test, y_proba))

# Save model
Path('models').mkdir(exist_ok=True)
joblib.dump(pipe, 'models/credit_scoring_model.joblib')
print('Saved model to models/credit_scoring_model.joblib')
