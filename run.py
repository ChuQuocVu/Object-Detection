from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Dropout, Input
from cv2 import cv2
import numpy as np


def get_model():
    model = VGG16(weights='imagenet', include_top=False)

    for layer in model.layers:
        layer.trainable = False

    input = Input(shape=(128, 128, 3), name='input_image')
    output = model(input)

    lay = Flatten(name='flatten')(output)
    lay = Dense(4096, activation='relu', name='fc1')(lay)
    lay = Dropout(0.5)(lay)
    lay = Dense(4096, activation='relu', name='fc2')(lay)
    lay = Dropout(0.5)(lay)
    lay = Dense(5, activation='softmax', name='classifier')(lay)

    final_model = Model(inputs=input, outputs=lay)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return final_model


my_model = get_model()
my_model.load_weights("E:\Pycharm\Object_classifier\model\weights-37-1.00.hdf5")


def run(model, image_file=None):
    class_name = ['00000', '10000VND', '50000VND', 'Notebook', "Pen"]

    if image_file is not None:
        image_capture = cv2.VideoCapture(image_file)
    else:
        image_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = image_capture.read()
        if not ret:
            continue

        frame_test = cv2.resize(frame, dsize=(128, 128))
        frame_test = frame_test.astype('float') * 1. / 255
        frame_test = np.expand_dims(frame_test, axis=0)

        predict = model.predict(frame_test)

        print("Label is: ", class_name[np.argmax(predict[0])], predict[0])
        print(np.max(predict[0], axis=0))

        if (np.max(predict[0]) >= 0.8) and (np.argmax(predict[0]) != 0):
            cv2.putText(frame, class_name[np.argmax(predict[0])], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0),
                        2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    image_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run(my_model, image_file=None)
