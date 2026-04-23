import os
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import zipfile
matplotlib.use("TkAgg")


with zipfile.ZipFile("data/clf-data.zip") as zip_ref:
    zip_ref.extractall("data/")


input_dir = "data/clf-data/"

categories = ["empty", "not_empty"]

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)


data = np.asarray(data)
labels = np.asarray(labels)


# train, test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
print(len(x_train), len(x_test))
print(np.bincount(y_train), np.bincount(y_test))
# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grip_search = GridSearchCV(classifier, parameters)
grip_search.fit(x_train, y_train)

# test performance
best_estimator = grip_search.best_estimator_
y_pred = best_estimator.predict(x_test)

score = accuracy_score(y_pred, y_test)

print('{}% of samples were correctly classifiers'.format(str(score * 100)))
score = accuracy_score(y_pred, y_test)
pickle.dump(best_estimator, open("model/model.p", 'wb'))
# visualization
print("Best parameters:", grip_search.best_params_)
print("Best CV score:", grip_search.best_score_)

# 1. Hiển thị accuracy của train và test
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Train vs Test Accuracy
train_acc = best_estimator.score(x_train, y_train)
test_acc = best_estimator.score(x_test, y_test)

axes[0].bar(['Train', 'Test'], [train_acc, test_acc], color=['#2ecc71', '#3498db'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Train vs Test Accuracy')
axes[0].set_ylim([0, 1])
for i, v in enumerate([train_acc, test_acc]):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Plot 2: Kết quả GridSearch - Mean test score cho mỗi tham số
cv_results = grip_search.cv_results_
mean_test_scores = cv_results['mean_test_score']
params = cv_results['params']

# Tạo labels cho các tham số
param_labels = [f"C={p['C']}\nγ={p['gamma']}" for p in params]

axes[1].bar(range(len(mean_test_scores)), mean_test_scores, color='#e74c3c')
axes[1].set_ylabel('Mean CV Score')
axes[1].set_xlabel('Parameter Combination')
axes[1].set_title('GridSearchCV Results')
axes[1].set_xticks(range(len(mean_test_scores)))
axes[1].set_xticklabels(param_labels, fontsize=8)
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.show()

# 3. Hiển thị confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Empty', 'Not Empty'])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.show()