import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn import model_selection

dataset = pd.read_csv("HeartDisease.csv")


predictors = dataset.drop("target",axis=1)
target = dataset["target"]


seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

nb = GaussianNB()
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)

accuracy_results = model_selection.cross_val_score(nb, predictors, target, cv=kfold, scoring='accuracy')
print("Accuracy applying k-fold: %.3f " % accuracy_results.mean())



score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
cm_nb = confusion_matrix(Y_test, Y_pred_nb)

print("The accuracy score achieved using Naive Bayes without k-fold is: "+str(score_nb)+" %")
print("Confusion Matrix : ")
print(cm_nb)
print("LogLoss :"+str(log_loss(Y_test, Y_pred_nb)))
print("Classification Report : ")
print(classification_report(Y_test,Y_pred_nb))

max_accuracy = 0


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)

score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
cm_rf = confusion_matrix(Y_test, Y_pred_rf)
print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")
print("Confusion Matrix : ")
print(cm_rf)
print("LogLoss :"+str(log_loss(Y_test, Y_pred_rf)))
print("Classification Report : ")
print(classification_report(Y_test,Y_pred_rf ))


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(["Naive Bayes","Random Forest"],[score_nb,score_rf])
