import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

# ------------------------ Load and Prepare Data ------------------------
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------ Train Models ------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)

# ------------------------ Predict and Combine ------------------------
rf_probs = rf.predict_proba(X_test)[:, 1]
svm_probs = svm.predict_proba(X_test)[:, 1]

w_rf, w_svm = 0.6, 0.4
hybrid_probs = (w_rf * rf_probs) + (w_svm * svm_probs)
hybrid_pred = (hybrid_probs >= 0.5).astype(int)

# ------------------------ Evaluate ------------------------
accuracy = accuracy_score(y_test, hybrid_pred)
f1 = f1_score(y_test, hybrid_pred)
print(f"Hybrid Accuracy: {accuracy:.2f}")
print(f"Hybrid F1-Score: {f1:.2f}")

# ------------------------ Prediction Function ------------------------
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    new_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree_function, age]])
    rf_prob = rf.predict_proba(new_data)[0][1]
    svm_prob = svm.predict_proba(new_data)[0][1]
    hybrid_score = (w_rf * rf_prob) + (w_svm * svm_prob)
    return 1 if hybrid_score >= 0.5 else 0

# Example usage
new_data_point = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
prediction = predict_diabetes(*new_data_point)
print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")

# ------------------------ Save Models ------------------------
with open('random_forest_model.pickle', 'wb') as f:
    pickle.dump(rf, f)

with open('svm_model.pickle', 'wb') as f:
    pickle.dump(svm, f)

with open('ensemble_weights.pickle', 'wb') as f:
    pickle.dump({'rf_weight': w_rf, 'svm_weight': w_svm}, f)

# ------------------------ Visualizations ------------------------

# 1. Confusion Matrix and ROC Curve
cm = confusion_matrix(y_test, hybrid_pred)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, hybrid_probs)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'],
            ax=axs[0])
axs[0].set_title("Confusion Matrix - Hybrid Model")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("Actual")

# ROC Curve
axs[1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc(fpr_rf, tpr_rf):.2f})')
axs[1].plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc(fpr_svm, tpr_svm):.2f})')
axs[1].plot(fpr_hybrid, tpr_hybrid, label=f'Hybrid (AUC = {auc(fpr_hybrid, tpr_hybrid):.2f})', linewidth=2.5)
axs[1].plot([0, 1], [0, 1], 'k--')
axs[1].set_title("ROC Curve")
axs[1].set_xlabel("False Positive Rate")
axs[1].set_ylabel("True Positive Rate")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()

# 2. Feature Importances
plt.figure(figsize=(8, 4))
sns.barplot(x=rf.feature_importances_, y=X.columns)
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()

# 3. Heatmap (Correlation)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()

# 4. User Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=data, palette='Set2')
plt.title("User Distribution: Diabetic vs Non-Diabetic")
plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
plt.ylabel("Count")
plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'])
plt.tight_layout()

# Show all figures at once
plt.show()
