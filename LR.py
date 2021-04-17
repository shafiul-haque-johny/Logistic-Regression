# Logistic Regression By Md.Shafiul Haque Johny

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import random

iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

X_list = X.tolist()
Y_list = y.tolist()

Training_x = []
Training_y = []

Validation_x = []
Validation_y = []

Testing_x = []
Testing_y = []

G_Training_X = []
G_Training_Y = []

G_validation_X = []
G_validation_Y = []

G_testing_X = []
G_testing_Y = []

correct = 0
Total_loss = 0

for i in range(len(X_list)):

    X_list[i].insert(0, 1)

for i in range(len(X_list)):

    R = random.uniform(0, 1)

    if R >= 0 and R <= 0.7:

        Training_x.append(X_list[i])
        Training_y.append(Y_list[i])

    elif R > 0.7 and R <= 0.85:

        Validation_x.append(X_list[i])
        Validation_y.append(Y_list[i])

    else:
        Testing_x.append(X_list[i])
        Testing_y.append(Y_list[i])

print(len(Training_x))
print(len(Validation_x))
print(len(Testing_x))

Training_x_array = np.array(Training_x)
Training_y_array = np.array(Training_y)

Validation_x_array = np.array(Validation_x)
Validation_y_array = np.array(Validation_y)

Testing_x_array = np.array(Testing_x)
Testing_y_array = np.array(Testing_y)

Theta = [0.6, 0.7, 0.1]

lr = 0.001  # 0.1 , 0.01, 0.001, 0.0001, 0.00001

for i in range(100):

    for j in range(len(Training_x)):

        z = np.dot(Training_x_array[j], Theta)
        h = 1 / (1 + np.exp(-z))
        loss = (-Training_y_array[j] * np.log(h)) - ((1 - Training_y_array[j]) * np.log(1 - h))
        Total_loss = Total_loss + loss
        dv = Training_x_array[j] * (h - Training_y_array[j])
        Theta = Theta - (dv * lr)

    Total_loss = Total_loss / len(Training_x)

    G_Training_Y.append(Total_loss)
    G_Training_X.append(i)


plt.figure(figsize=(6, 6))
plt.plot(G_Training_X, G_Training_Y)
plt.title("Loss Function")
plt.xlabel("Iteration(Epoch)")
plt.ylabel("Train Loss")

print("Theta", Theta)

for i in range(len(Validation_x)):

    z = np.dot(Validation_x_array[i], Theta)

    h = 1 / (1 + np.exp(-z))

    if h >= 0.5:
        h = 1

    else:
        h = 0

    if h == Validation_y_array[i]:

        correct += 1

        G_validation_X.append(i)
        G_validation_Y.append(correct)

Validation_acc = (correct / len(Validation_x)) * 100
print('Validation Accuracy= ', Validation_acc)

correct = 0

for i in range(len(Testing_x)):

    z = np.dot(Testing_x_array[i], Theta)
    h = 1 / (1 + np.exp(-z))

    if h >= 0.5:
        h = 1

    else:
        h = 0

    if h == Testing_y_array[i]:

        correct += 1

        G_testing_X.append(i)
        G_testing_Y.append(correct)

Testing_acc = (correct / len(Testing_x)) * 100
print('Test Accuracy = ', Testing_acc)

plt.show()
