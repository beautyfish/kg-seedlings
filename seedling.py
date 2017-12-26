import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import PIL

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
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_generator.image_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(Flatten())

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
        verbose=2,
        callbacks=[checkpointer])

'''
f = open('results.csv', 'w')
f.write('file, ')

output = model.predict(test)
filenames = os.listdir('data/test')
for [o, name] in zip(output, filenames):
    f.write(name[:-4] + ',')
    o.tofile(f, sep=',', format='%.17f')
    f.write('\n')
f.close()
'''