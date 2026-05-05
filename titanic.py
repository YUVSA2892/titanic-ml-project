# Step 1: Import libraries
# Step 2: Load dataset
# Step 3: Handle missing values
# Step 4: Data analysis
# Step 5: Feature engineering
# Step 6: Model training
# Step 7: Evaluation

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


import pandas as pd

data = pd.read_csv("titanic.csv")

df = pd.DataFrame(data)

print("First Five Rows: \n",df.head())
print(df.info())
print("Description of Data: \n",df.describe())
print("Empty Values: ")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Has_Cabin"] = df["Cabin"].notnull().astype(int)
df = df.drop(columns=["Cabin"])
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print(df)
print("\nAfter Cleaning:\n")
print(df.isnull().sum())

df.groupby("Sex")["Survived"].mean().plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.show()

df.groupby("Pclass")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Class")
plt.show()

plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.show()


df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

df = df.drop(columns=["Name", "Ticket", "PassengerId"])

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
