# Iris Flower Classification using K-Nearest Neighbors (KNN)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Step 2: Load Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Step 3: Data Exploration
print("First 5 rows of the dataset:")
print(df.head())

# Pairplot for visualization
sns.pairplot(df, hue='species', palette='Set2')
plt.suptitle("Iris Pairplot", y=1.02)
plt.show()

# Step 4: Prepare Data
X = df[iris.feature_names]
y = df['species']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Apply KNN Algorithm
k = 3  # You can tune this value later
knn = KNeighborsClassifier(n_neighbors=k)

# Step 6: Train the Model
knn.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = knn.predict(X_test)

# Step 8: Evaluate the Model
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy with k={k}: {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['setosa', 'versicolor', 'virginica'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['setosa', 'versicolor', 'virginica'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Step 9: Try Different k Values
accuracies = []
k_range = range(1, 11)

for k_val in k_range:
    model = KNeighborsClassifier(n_neighbors=k_val)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc_k = accuracy_score(y_test, pred)
    accuracies.append(acc_k)
    print(f"Accuracy for k={k_val}: {acc_k:.2f}")

# Step 10: Plot Accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_range, accuracies, marker='o', linestyle='-', color='purple')
plt.title("Accuracy vs. k Value")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.xticks(k_range)
plt.grid(True)
plt.show()
