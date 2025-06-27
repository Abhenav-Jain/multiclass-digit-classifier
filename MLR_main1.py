'''
✅ Question 1: Digit Classification (MNIST Subset)
🧠 Concept: Classic multi-class classification — 0 to 9 digits using Logistic Regression

📌 Task:
1. Load the load_digits dataset from sklearn.datasets.
2. Use only first 1000 samples to speed up training.
3. Train a Logistic Regression classifier.
4. Print accuracy and confusion matrix.
5. Visualize predictions for 10 random images.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load and prepare dataset
digits = load_digits()
x, y = digits.data[:1000], digits.target[:1000]

# Visualize one sample
plt.gray()
plt.matshow(digits.images[20])
plt.title(f"Label: {y[20]}")
plt.show()

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Visualize 10 predictions
for i in range(10):
    idx = np.random.randint(0, len(x_test))
    plt.matshow(x_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f"Actual: {y_test[idx]}, Predicted: {y_pred[idx]}")
    plt.show()

# Check the error using confusion matrix and seaborn
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
classes = np.unique(y)  # classes from 0 to 9

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            square=True, cbar=True, linewidths=0.5, linecolor='gray')

plt.title("Confusion Matrix of Digit Classification", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
