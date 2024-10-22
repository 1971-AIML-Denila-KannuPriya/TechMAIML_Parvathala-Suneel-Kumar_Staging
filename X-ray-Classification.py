import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size for ResNet50 (224x224)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Image augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Rotation augmentation
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,
    zoom_range=0.2,  # Zoom augmentation
    horizontal_flip=True,  # Randomly flip images
    fill_mode='nearest',  # Fill missing pixels after augmentation
    validation_split=0.2  # Split for validation
)
# D:\Datasets\train
train_generator = train_datagen.flow_from_directory(
    r'D:\Datasets\train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    r'D:\Datasets\test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
# Load the pre-trained ResNet50 model without the top layer
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',  # Pre-trained on ImageNet
    include_top=False,  # Exclude fully-connected layer at the top
    input_shape=IMAGE_SIZE + (3,)  # Input size (224, 224, 3)
)

# Freeze the base model layers to prevent training them
base_model.trainable = False

# Create a new model on top
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),  # Reduce the dimensions
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluate on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator)

# Predict on validation set
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices, yticklabels=train_generator.class_indices)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes))

