from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load labels from labels.csv
labels_df = pd.read_csv('./dataset/labels.csv')

# Load images and corresponding labels
image_data = []
labels = []

data_dir = 'dataset/myData/'
for class_id, class_name in labels_df.values:
    class_dir = os.path.join(data_dir, str(class_id))
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = Image.open(image_path)
        image_data.append(np.array(image))
        labels.append(class_id)

# Convert lists to NumPy arrays
X = np.array(image_data)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the pixel values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Define the MLP model with hyperparameter tuning
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = mlp.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Create a table to present the comparative analysis
performance_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'MLP Model': [accuracy, precision, recall, f1]
}

performance_df = pd.DataFrame(performance_data)
print(performance_df)
