from os import listdir
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from cv2 import cv2
import pickle


raw_folder = 'Data/'

def save_data(folder = raw_folder):

    images = []
    labels = []

    for folder in listdir(raw_folder):
        if folder != '.DS_Store':
            print("Folder = ", folder)


            for file in listdir(raw_folder + folder):
                if file != '.DS_Store':
                    print("File = ", file)

                    images.append( cv2.resize(cv2.imread(r"{}{}{}{}".format(raw_folder, folder, "/", file)), dsize=(128, 128)))
                    labels.append( folder)

    x = np.array(images)
    y = np.array(labels)

    encoder = LabelBinarizer()
    y_onehot = encoder.fit_transform(y)

    file = open("train_file.data", "wb")
    pickle.dump((x, y_onehot), file)
    file.close()

def load_data():
    load_file = open("train_file.data", 'rb')
    (images, labels) = pickle.load(file=load_file)
    load_file.close()

    print(images.shape)
    print(labels.shape)

    return images, labels

# save_data(raw_folder)

(x_data, y_data) = load_data()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=100)

def get_model():

    model = VGG16(weights='imagenet', include_top=False)

    for layer in model.layers:
        layer.trainable = False

    input = Input(shape=(128, 128, 3), name='input_image')
    output = model(input)

    lay = Flatten(name = 'flatten')(output)
    lay = Dense(4096, activation='relu', name='fc1')(lay)
    lay = Dropout(0.5)(lay)
    lay = Dense(4096, activation='relu', name='fc2')(lay)
    lay = Dropout(0.5)(lay)
    lay = Dense(4, activation='softmax', name='classifier')(lay)

    final_model = Model(inputs=input, outputs=lay)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return final_model

model = get_model()
filepath = "weight--{epoch:02d}--{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

#lam giau du lieu dau vao

avg = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                         rescale=1./255, width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         brightness_range=[0.2,1.5], fill_mode='nearest')

avg_val = ImageDataGenerator(rescale=1./255)

my_model = model.fit_generator(avg.flow(x_train, y_train, batch_size=64),
                               epochs=50,
                               validation_data=avg.flow(x_test, y_test, batch_size=len(x_test)),
                               callbacks=callback_list)

model.save("VGGModel.h5")

if __name__ == '__main__':
    model = get_model()


