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

# Feature Importance with a Forest of Trees
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

x = df.loc[:,['time','ejection_fraction']].values
y = df.loc[:,"DEATH_EVENT"].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=2)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
print("The accuracy of Logistic Regression is: " + str(100* log_reg_acc) +"%")

# Support Vector Classification
sv_clf = SVC()
sv_clf.fit(x_train, y_train)
sv_clf_pred = sv_clf.predict(x_test)
sv_clf_acc = accuracy_score(y_test, sv_clf_pred)
print("The accuracy of SVC is: " + str(100* sv_clf_acc) +"%")

# KNClassifier
kn_clf = KNeighborsClassifier(n_neighbors=6)
kn_clf.fit(x_train, y_train)
kn_pred = kn_clf.predict(x_test)
kn_acc = accuracy_score(y_test, kn_pred)
print("The accuracy of KN Classifier is: " + str(100* kn_acc) +"%")

# DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("The accuracy of Decision Tree is: " + str(100* dt_acc) +"%")

# RandomForestClassifier
r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
r_clf.fit(x_train, y_train)
r_pred = r_clf.predict(x_test)
r_acc = accuracy_score(y_test, r_pred)
print("The accuracy of Random Forest is: " + str(100* r_acc) +"%")

'''
The accuracy of Logistic Regression is: 90.0%
The accuracy of SVC is: 90.0%
The accuracy of KN Classifier is: 91.66666666666666%
The accuracy of Decision Tree is: 90.0%
The accuracy of Random Forest is: 88.33333333333333%
'''
#We have therefore built a model using KN Classifier which predicts heart failure with an accuracy of 91.66%