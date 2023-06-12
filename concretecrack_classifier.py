"""
Assessment 3: Concrete Crack Image Classification
URL: https://data.mendeley.com/datasets/5y9wdsg2zt/2
"""
#%%
#split dataset into train, val and test
#define location of dataset folder
import splitfolders
input_folder = "Dataset_ConcreteCrack"
#use ratio to split dataset and output on another folder
splitfolders.ratio(input_folder, output="DatasetSplit_ConcreteCrack", seed=42, ratio=(0.7, 0.2, 0.1), group_prefix=None)

#%%
#1.Import package
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, datetime
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, applications, callbacks

#%%
#define path
#PATH = os.getcwd()
#train_path = os.path.join(PATH, "DatasetSplit_ConcreteCrack\train")
#val_path = os.path.join(PATH, "DatasetSplit_ConcreteCrack\val")
#test_path = os.path.join(PATH, "DatasetSplit_ConcreteCrack\test")
#%%
#2.Data loading
#define the path to folder
train_path = r"C:\Users\user\Desktop\YP-AI03\DeepLearning\CAPSTONE\Assessment_3\DatasetSplit_ConcreteCrack\train"
val_path = r"C:\Users\user\Desktop\YP-AI03\DeepLearning\CAPSTONE\Assessment_3\DatasetSplit_ConcreteCrack\val"
test_path = r"C:\Users\user\Desktop\YP-AI03\DeepLearning\CAPSTONE\Assessment_3\DatasetSplit_ConcreteCrack\test"

#%%
#define batch size and image size
BATCH_SIZE = 128
IMG_SIZE = (160,160)

#%%
#load all data as tensorflow dataset
train_dataset = keras.utils.image_dataset_from_directory(train_path, batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True)
val_dataset = keras.utils.image_dataset_from_directory(val_path, batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True)
test_dataset = keras.utils.image_dataset_from_directory(test_path, batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True)

#%%
#3.Data inspection
# extract the class names
class_names = train_dataset.class_names
print(class_names)

#%%
#plot to see some example of image
plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()
# %%
#4. Convert the all dataset into prefetchdataset
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = val_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
#5. create a Keras Model for data augmentation
data_aug = keras.Sequential()
data_aug.add(layers.RandomFlip("horizontal_and_vertical"))
data_aug.add(layers.RandomRotation(0.3))

#%%
#6. plot to see the data augmentation model
for images, labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        augmented_image = data_aug(tf.expand_dims(first_image, axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')
plt.show()

# %%
#7. Create a layer to perform the pixel standardization
preprocess_input = applications.mobilenet_v2.preprocess_input

#%%
#8. Apply transfer learning
#get pre-train model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

#%%
#set pre-train feature extractor as non-trainable/freeze
base_model.trainable = False
base_model.summary()

#%%
#build classifier
global_avg = keras.layers.GlobalAveragePooling2D()
#output layer
output_layer = keras.layers.Dense(len(class_names), activation='softmax')

#%%
#create final model
#input layer
inputs = keras.Input(shape=IMG_SHAPE)
#data augmentation model
x = data_aug(inputs)
#pixel standardaization
x = preprocess_input(x)
#feature extraction layer
x = base_model(x, training=False)
#global average pooling layer
x = global_avg(x)
#output layer
outputs = output_layer(x)

#%%
#instantiate the final model
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#%%
#9.Compile the model
optimize = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimize, loss=loss, metrics=['accuracy'])

#%%
#10. evaluate the model before training
loss0, acc0 = model.evaluate(pf_test)
print("-----Evaluation before training------")
print("Loss = ",loss0)
print("Accuracy = ",acc0)

#%%
#11.create tensorboard callback object for plotting
tb = keras.callbacks.TensorBoard(log_dir='logdir\\{}'.format("assessment_3"))

#add early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.0001, mode='max')

#%%
#12.Model training
history = model.fit(pf_train, validation_data=pf_val, epochs=10, callbacks=[tb,early_stopping])

#%%
#13. evaluate finale transfer learning
test_loss, test_accuracy =model.evaluate(pf_test)
print("After Training\nTest accuracy: ",test_accuracy)

#%%
#14.Model deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch), axis=1)
#label and prediction stack in one numpy array for comparison
label_vs_prediction = np.transpose(np.vstack((label_batch, y_pred)))

#%%
#15. save model in .h5
save_path = os.path.join("save_model", "transfer_learning_ConcreteCrack_model.h5")
model.save(save_path)

# %%
