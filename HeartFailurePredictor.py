import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Feature Importance
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
model = ExtraTreesClassifier()
model.fit(x,y)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
print(feat_importances.nlargest(5))
'''
time                 0.295991
ejection_fraction    0.131900
serum_creatinine     0.119739
age                  0.091897
serum_sodium         0.084024
'''
# We find that time is the most important, followed by ejection_fraction etc.
# We then tested our models below with time alone vs adding one parameter at a time and found that
# the combination of the 2 most important features leads to the highest accuracy
# Here below is the code using these 2 features

