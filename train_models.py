import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib
import os

data = pd.read_csv("diabetes.csv")

features_to_fix = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[features_to_fix] = data[features_to_fix].replace(0, np.nan)
data.fillna(data.median(numeric_only=True), inplace=True)

features = [
    "Pregnancies",
    "Glucose",
    "BMI",
    "Age"
]

X = data[features]
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

best_model = None
best_name = ""
best_accuracy = 0
best_preds = None

print("Training and evaluating models...")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_model = model
        best_name = name
        best_accuracy = acc
        best_preds = preds

os.makedirs("models", exist_ok=True)

joblib.dump(models["Logistic Regression"], "models/logistic.pkl")
joblib.dump(models["Decision Tree"], "models/decision_tree.pkl")
joblib.dump(models["Random Forest"], "models/random_forest.pkl")
joblib.dump(models["KNN"], "models/knn.pkl")
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nBest Model:", best_name)
print(f"Accuracy: {best_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, best_preds))

cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1]
)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nAll trained models saved successfully!")
print("Models saved inside: models/")
