from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

crop = pd.read_csv("Crop_recommendation.csv")

X = crop[['N','P','K','temperature','humidity','ph','rainfall']]
y = crop['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=48)

scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.fit_transform(X_test)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

svm = SVC()
rf = RandomForestClassifier()

svm.fit(X_train_scaled,y_train)
rf.fit(X_train_scaled,y_train)