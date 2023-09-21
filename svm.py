import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

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
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_data.append(image_array)
        labels.append(class_id)

# Convert lists to NumPy arrays
X = np.array(image_data)
y = np.array(labels)

# Flatten the image data
X = X.reshape(X.shape[0], -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create and train an SVM model
svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=False)
svm_model.fit(X_train, y_train_encoded)

# Evaluate the SVM model on the test set
y_svm_pred = svm_model.predict(X_test)

# Calculate performance metrics for the SVM model
precision = precision_score(y_test_encoded, y_svm_pred, average='weighted')
recall = recall_score(y_test_encoded, y_svm_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_svm_pred, average='weighted')

# Create a table to present the comparative analysis
performance_data = {
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'SVM Model': [precision, recall, f1]
}

performance_df = pd.DataFrame(performance_data)
print(performance_df)


# Need to add hyperparameter tuning (GridSearch)