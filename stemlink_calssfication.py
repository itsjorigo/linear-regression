import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# as_frame=True loads the data in a dataframe format, with other metadata besides it
california_housing = fetch_california_housing(as_frame=True)
# Select only the dataframe part and assign it to the df variable
df = california_housing.frame
# print(df)

# # Preprocessing Data for Classification
df["MedHouseValCat"] = pd.qcut(df["MedHouseVal"], 4, retbins=False, labels=[1, 2, 3, 4])


y = df['MedHouseValCat']
X = df.drop(['MedHouseVal', 'MedHouseValCat'], axis=1)

# print(df)


#
# # Splitting Data into Train and Test Sets
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
#
# # Feature Scaling for Classification
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#
# # Training and Predicting for Classification
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#
# Evaluating KNN for Classification
acc = classifier.score(X_test, y_test)
print(acc)  # 0.6191860465116279
#
# classes_names = ['class 1', 'class 2', 'class 3', 'class 4']
# cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=classes_names, index=classes_names)
# sns.heatmap(cm, annot=True, fmt='d')
#
# print(classification_report(y_test, y_pred))
#
# # Finding the Best K for KNN Classification
# f1s = []
# for i in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     f1s.append(f1_score(y_test, pred_i, average='weighted'))
#
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 40), f1s, color='red', linestyle='dashed', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.title('F1 Score K Value')
# plt.xlabel('K Value')
# plt.ylabel('F1 Score')
#
classifier15 = KNeighborsClassifier(n_neighbors=15)
classifier15.fit(X_train, y_train)
y_pred15 = classifier15.predict(X_test)
print(classification_report(y_test, y_pred15))
#
#
# # New Prediction
new_data = pd.DataFrame([[8.3252 ,41.0 ,2 	,4 ,  322.0 	,   2.555556 ,  37.88 ,	-122.23  ]], columns=X.columns)
print(new_data)

# Scaling the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Making the prediction
new_prediction = classifier15.predict(new_data_scaled)
#
print("The predicted class for the new data is:", new_prediction)