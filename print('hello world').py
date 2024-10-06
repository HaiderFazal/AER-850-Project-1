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
#from mpl_toolkits.mplot3d import Axes3D

#Step 1 - Data Processing

df = pd.read_csv("C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850\Project_1_Data.csv")
print(df.info())

#Step 2 - Data Visualization

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
cbar = plt.colorbar(scatter_plot,ax=ax)
cbar.set_label('Step')
plt.show()

#Step 3 - Correlation Analysis
corr=df.corr()
corr=corr.drop(['Step'],axis='columns')
corr=corr.drop(['Step'],axis='rows')
sns.heatmap(corr,cmap='rocket',annot=True)
plt.show()

#Step 4 - Classification Model Development/ Engineering

