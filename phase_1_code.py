import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the MLP model using tf.keras.Sequential
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),  # Flatten the input
    keras.layers.Dense(100, activation='relu'),    # 1st hidden layer with 100 units and ReLU activation
    keras.layers.Dense(50, activation='relu'),     # 2nd hidden layer with 50 units and ReLU activation
    keras.layers.Dense(len(labels_df), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32)

# Evaluate the model on the test set
_, accuracy = model.evaluate(X_test, y_test)

# Make predictions using the trained model
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate performance metrics
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