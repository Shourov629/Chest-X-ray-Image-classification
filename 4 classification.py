import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 64
num_classes = 4

train_dir = '/kaggle/input/4-class/4 class/train'
val_dir = '/kaggle/input/4-class/4 class/validation'

img_width, img_height = 224, 224

model4_54m_2 = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='sparse')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_width, img_height),
                                                  batch_size=batch_size, class_mode='sparse')

model4_54m_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epoch = 50
history = model4_54m_2.fit(train_generator, epochs=epoch, validation_data=val_generator)


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the test data directory
test_data_dir = '/kaggle/input/4-class/4 class/test'

# Define hyperparameters
batch_size = 64  # Adjust batch size as needed

# Create a data generator for the test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),  # ResNet50 input size
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False  # Important: Do not shuffle the test data
)

test_loss, test_accuracy = model4_54m_2.evaluate(test_generator)

print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_accuracy:.4f}')

# Predict class probabilities for each sample in the test set
predictions = model4_54m_2.predict(test_generator)

# Get the predicted class labels (argmax)
predicted_labels = np.argmax(predictions, axis=1)

# Get the true class labels
true_labels = test_generator.classes

# Calculate confusion matrix, classification report, and other evaluation metrics if needed
from sklearn.metrics import confusion_matrix, classification_report

confusion = confusion_matrix(true_labels, predicted_labels)
classification_report_str = classification_report(true_labels, predicted_labels, digits=4)

# Create a DataFrame from the confusion matrix
class_names = list(test_generator.class_indices.keys())
cm_df = pd.DataFrame(confusion, index=class_names, columns=class_names)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='gray', cbar=False)

plt.xlabel('Predicted Labels', fontsize=18, fontweight='bold')
plt.ylabel('True Labels', fontsize=18, fontweight='bold')
plt.title('Confusion Matrix', fontsize=18, fontweight='bold')

plt.savefig('/kaggle/working/confusion_matrix for model4_54m_2.png')

plt.show()

print("\nClassification Report:")
print(classification_report_str)


acc3 = history.history['accuracy']
val_acc3 = history.history['val_accuracy']

loss3 = history.history['loss']
val_loss3 = history.history['val_loss']

# Create DataFrames with labels
acc_3C = pd.DataFrame({'train accuracy': acc3, 'validation accuracy': val_acc3})
loss_3C = pd.DataFrame({'train loss': loss3, 'validation loss': val_loss3})

acc_3C.to_csv("/kaggle/working/accuracy of train vs validation for model4_54m_2.csv")
loss_3C.to_csv("/kaggle/working/loss of train vs validation for model4_54m_2.csv")

import matplotlib.pyplot as plt
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), acc3, label='Training Accuracy')
plt.plot(range(epoch), val_acc3, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(epoch), loss3, label='Training Loss')
plt.plot(range(epoch), val_loss3, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()