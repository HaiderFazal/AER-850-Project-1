import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import joblib

#Step 1: Data Processing

df = pd.read_csv("C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project_1_Data.csv")
print(df.info())

#Step 2: Data Visualization

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plotting X, Y, Z against Step
scatter_plot=ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='magma')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Scatter Plot of X,Y,Z against Step')
cbar = plt.colorbar(scatter_plot)
cbar.set_label('Step')
plt.show()

#Step 3: Correlation Analysis

corr_matrix = df.corr()
sns.heatmap(corr_matrix,cmap='Reds',annot=True)
plt.show()

#Step 4: Classification Model Development/ Engineering

# Data splitting
my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 777)
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)

X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]

# Model 1 - Support Vector Machine (SVM)
svc = SVC()
param_grid_svc = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_svc.fit(X_train, y_train)
best_model_svc = grid_search_svc.best_estimator_
#print("Best SVM Model:", best_model_svc)
y_train_pred_svc = best_model_svc.predict(X_train)
y_test_pred_svc = best_model_svc.predict(X_test)

# Model 2 - Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1,2,4]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
#print(grid_search_dt.best_params_)
best_model_dt = grid_search_dt.best_estimator_
#print("Best Decision Tree Model:", best_model_dt)
y_train_pred_dt = best_model_dt.predict(X_train)
y_test_pred_dt = best_model_dt.predict(X_test)

# Model 3 - Random Forest
random_forest = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [10, 30, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_
#print("Best Random Forest Model:", best_model_rf)
y_train_pred_rf = best_model_rf.predict(X_train)
y_test_pred_rf = best_model_rf.predict(X_test)

# Model 4 -  SVM RandomizedSearchCV 
svc_random = SVC()
param_grid_svc_random = {
    'kernel': ['linear', 'rbf'],
    'C': uniform(0.1,100),
    'gamma': ['scale', 'auto']
}
random_search_svc = RandomizedSearchCV(svc, param_grid_svc_random, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1,random_state=42)
random_search_svc.fit(X_train, y_train)
best_model_svc_random = random_search_svc.best_estimator_
y_train_pred_svc_random = best_model_svc.predict(X_train)
y_test_pred_svc_random = best_model_svc.predict(X_test)
#print("Best SVM Model using RandomizedSearchCV:", best_model_svc_random)

#Step 5: Model Performance Analysis
  
#SVC Model Performance
for i in range(5):
   print("(SVC) Step Predictions:", y_train_pred_svc[i], "(SVC) Step Actual values:", y_train[i])

f1_train_SVC = f1_score(y_train, y_train_pred_svc, average='weighted')
precision_train_SVC = precision_score(y_train, y_train_pred_svc, average='weighted')
accuracy_train_SVC = accuracy_score(y_train, y_train_pred_svc)

print("\nSVC Training Set Evaluation:")
print("F1 Score: ", f1_train_SVC)
print("Precision: ", precision_train_SVC)
print("Accuracy: ", accuracy_train_SVC)
print()

# Decision Tree Model Performance 
for i in range(5):
   print("(DT) Step Predictions:", y_train_pred_dt[i], "(DT) Step Actual values:", y_train[i])

f1_train_DT = f1_score(y_train, y_train_pred_dt, average='weighted')
precision_train_DT = precision_score(y_train, y_train_pred_dt, average='weighted')
accuracy_train_DT = accuracy_score(y_train, y_train_pred_dt)

print("\nDecision Tree Set Evaluation:")
print("F1 Score: ", f1_train_DT)
print("Precision: ", precision_train_DT)
print("Accuracy: ", accuracy_train_DT)
print()

# Random Forest Model Performance 
for i in range(5):
   print("(RF) Step Predictions:", y_train_pred_rf[i], "(RF) Step Actual values:", y_train[i])

f1_train_RF = f1_score(y_train, y_train_pred_rf, average='weighted')
precision_train_RF = precision_score(y_train, y_train_pred_rf, average='weighted')
accuracy_train_RF = accuracy_score(y_train, y_train_pred_rf)

print("\nRandom Forest Set Evaluation:")
print("F1 Score: ", f1_train_RF)
print("Precision: ",precision_train_RF)
print("Accuracy: ", accuracy_train_RF)
print()

# SVC RandomizedSearchCV Model Performance 
for i in range(5):
   print("(Randomized SVC ) Step Predictions:", y_train_pred_svc_random[i], "(Randomized SVC) Step Actual values:", y_train[i])

f1_train_random = f1_score(y_train, y_train_pred_svc_random, average='weighted')
precision_train_random = precision_score(y_train, y_train_pred_svc_random, average='weighted')
accuracy_train_random = accuracy_score(y_train, y_train_pred_svc_random)

print("\nRandomized SVC Set Evaluation:")
print("F1 Score: ", f1_train_random)
print("Precision: ",precision_train_random)
print("Accuracy: ", accuracy_train_random)
print()

# Plot Confusion Matrix for Selected Model = (Random Forest)
conf_matrix_train_RF = confusion_matrix(y_train, y_train_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train_RF, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Step")
plt.ylabel("True Step")
plt.show()

#Step 6: Stacked Model Performance Analysis
stacking_model = StackingClassifier(
    estimators=[('svc', best_model_svc), ('rf', best_model_rf)],
    final_estimator=LogisticRegression()
)

stacking_model.fit(X_train, y_train)

y_train_pred_stacked = stacking_model.predict(X_train)
y_test_pred_stacked = stacking_model.predict(X_test)

#for i in range(5):
   #print("(Stacked Model) Step Predictions:", y_train_pred_stacked[i], "(Stacked Model) Step Actual values:", y_train[i])

f1_train_stacked = f1_score(y_train, y_train_pred_stacked, average='weighted')
precision_train_stacked = precision_score(y_train, y_train_pred_stacked, average='weighted')
accuracy_train_stacked = accuracy_score(y_train, y_train_pred_stacked)

print("Stacking Classifier Training Set Evaluation:")
print("F1 Score: ", f1_train_stacked)
print("Precision: ", precision_train_stacked)
print("Accuracy: ", accuracy_train_stacked)

#stacked confusion matrix 
conf_matrix_train_stacked = confusion_matrix(y_train, y_train_pred_stacked)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train_stacked, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Stacked Model Confusion Matrix ")
plt.xlabel("Predicted Step")
plt.ylabel("True Step")
plt.show()

#Step 7: Model Evaluation

svc_model_file = "best_svc_model.pkl"
joblib.dump(best_model_svc, svc_model_file)
print(f"SVC model saved as {svc_model_file}")

loaded_svc_model = joblib.load(svc_model_file)

coordinates = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

predictions_svc = loaded_svc_model.predict(coordinates)

print("Predicted Maintenance Steps (SVC Model):")
print(predictions_svc)

# TA-DAAA THE END !!!!!