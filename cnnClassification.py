import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models
from keras_preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import configparser
import zero_one_loss

cfgfile = r'cfg.ini'
config = configparser.ConfigParser()
config.read(cfgfile)

# Database path from configuration file
db_path = r'%s' % config.get('DB','db_path')


# Class names from configuration file
class_names = [x.strip() for x in (config.get('CLASSES', 'class_names')).split(',')]

test_path = db_path + r'\Test'
training_path = db_path + r'\Training'

test_images = []
train_images = []
train_labels = []
test_labels = []

#Train images loading
train_filepaths = glob.glob((training_path+"\**\*.jpg"), recursive=True)
for file in train_filepaths:
    image = load_img(file)
    rgb_image = img_to_array(image)
    train_images.append(rgb_image)
    label_name = os.path.basename(os.path.dirname(os.path.dirname(file))) # Get label name from folder name
    label = class_names.index(label_name) # Numeric label
    train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels, dtype='int64')

#Test images loading
test_filepaths = glob.glob((test_path+"\**\*.jpg"), recursive=True)

for file in test_filepaths:
    image = load_img(file)
    rgb_image = img_to_array(image)
    test_images.append(rgb_image)
    label_name = os.path.basename(os.path.dirname(os.path.dirname(file))) # Get label name from folder name
    label = class_names.index(label_name) # Numeric label
    test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels, dtype='int64')

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

y_pred = model.predict(test_images)
y_true = test_labels

z_o_loss = zero_one_loss.zero_one_loss(y_true, y_pred)
print('\nZero one loss:',z_o_loss)