import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import keras_tuner as kt  # Import Keras Tuner
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

def build_hypermodel(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))

    # Tune the number of units in the Dense layers
    hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units1, activation='relu'))

    hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units2, activation='relu'))

    model.add(keras.layers.Dense(len(labels_df), activation='softmax'))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

tuner = kt.Hyperband(build_hypermodel,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='my_project')
tuner.search(X_train, y_train, epochs=50, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = build_hypermodel(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)
eval_result = model.evaluate(X_test, y_test)
print("Test loss:", eval_result[0])
print("Test accuracy:", eval_result[1])