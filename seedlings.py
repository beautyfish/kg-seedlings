import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tqdm import tqdm
from glob import glob
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 32
epochs = 20
target_size = (200, 200)

# Data import
train_datagen = ImageDataGenerator(rotation_range=40,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',  # this is the target directory
    target_size=target_size,  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

validation_generator = val_datagen.flow_from_directory(
    'data/validation',  # this is the target directory
    target_size=target_size,  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# Model definition
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=train_generator.image_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(12, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Training
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5',
                               verbose=1, save_best_only=True)

history = model.fit_generator(train_generator,
        steps_per_epoch=3831 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=919 // batch_size,
        verbose=1,
        callbacks=[checkpointer])

model.load_weights('saved_models/weights.best.hdf5')

seedlings_names = [item[11:] for item in sorted(glob("data/train/*"))]

f = open('results.csv', 'w')
f.write('file,species\n')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=target_size)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

test_files = glob("data/test/*")
test = paths_to_tensor(test_files).astype('float32')/255

output = model.predict(test)
for [o, name] in zip(output, test_files):
    f.write(name[10:] + ',')
    seedling = seedlings_names[np.argmax(o)]
    f.write(seedling)
    f.write('\n')
f.close()
