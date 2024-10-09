import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#Step 1 - Data Processing/Data Shuffling and Splitting

df = pd.read_csv("C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850\Project_1_Data.csv")
print(df.info())

my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 333)
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)

X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]

#Step 2 - Data Visualization

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plotting X, Y, Z against Step
scatter_plot=ax.scatter(X_train['X'], X_train['Y'], X_train['Z'], c=y_train, cmap='magma')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Scatter Plot of X,Y,Z against Step')
cbar = plt.colorbar(scatter_plot)
cbar.set_label('Step')
plt.show()

#Step 3 - Correlation Analysis

corr=X_train.corr()
sns.heatmap(corr,cmap='rocket',annot=True)
plt.show()

#Step 4 - Classification Model Development/ Engineering

my_model1=LinearRegression()
my_model1.fit(X_train,y_train)
y_pred_train1 = my_model1.predict(X_train)

for i in range(5):
   print("Predictions:", y_pred_train1[i], "Actual values:", y_train[i])

mae_train1 = mean_absolute_error(y_pred_train1, y_train)
print("Model 1 training MAE is: ", round(mae_train1,2))   

# """Cross Validation Model 1"""
cv_scores_model1 = cross_val_score(my_model1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae1 = -cv_scores_model1.mean()
print("Model 1 Mean Absolute Error (CV):", round(cv_mae1, 2))

my_model2=LogisticRegression()
my_model2.fit(X_train,y_train)


#Extra notes
#x_col=df[['X','Y','Z']]
#y_col=df['Step']

#corr=corr.drop(['Step'],axis='columns')
#corr=corr.drop(['Step'],axis='rows')

#X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size=0.2, stratify=y_col, random_state=42)