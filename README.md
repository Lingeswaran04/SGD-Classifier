# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Load the Iris dataset using load_iris() from sklearn.datasets.Create a DataFrame from the dataset and add the target column.
3. Separate the features (x) and target labels (y).Split the dataset into training and testing sets using train_test_split() with an 80-20 split.
4. Instantiate an SGD Classifier (SGDClassifier) with a maximum of 1000 iterations and tolerance (tol=1e-3).
5. Train the model using the training data (x_train and y_train) with the fit() method.
6. Predict the target labels for the test set (x_test) using the trained model.
7. Calculate the accuracy of the model using accuracy_score().Generate the confusion matrix using confusion_matrix().
8. Print the accuracy and confusion matrix.

## Program:
### DATA:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: LINGESWARAN K
RegisterNumber: 212222110022
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```
### SGDClassifier:
```
iris = load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
```
```
x=df.drop('target',axis=1)
y=df['target']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(x_train, y_train)
```
### ACCURACY:
```
sgd_clf.fit(x_train, y_train)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
### CONFUSION MARIX:
```
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
## Output:
### DATA:
![image](https://github.com/user-attachments/assets/03192351-7bb9-4aa5-a7cd-5d47414c2238)
### SGDClassifier:
![image](https://github.com/user-attachments/assets/8a285b91-d149-4a8e-8de6-ffce187aef97)
### ACCURACY:
![image](https://github.com/user-attachments/assets/c93be364-5fb2-458a-b88c-1dab94d3b108)
### CONFUSION MARIX:
![image](https://github.com/user-attachments/assets/f14d99a2-441f-4525-ba3d-5138908fda9e)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
