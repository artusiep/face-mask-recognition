import math

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras


def index_scale(index, scale):
    result_index = math.floor(index * scale)
    if result_index < 0:
        return 0
    return result_index


def display_image(face_image):
    img = Image.fromarray(face_image, 'RGB')
    img.show()


def image_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_for_classification(image, x, y, h, w, IMAGE_DIMENSIONS):
    face_img = image[y:y + h, x:x + w]
    face_img_rgb = image_to_rgb(face_img)

    resized = cv2.resize(face_img_rgb, IMAGE_DIMENSIONS)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, *IMAGE_DIMENSIONS, 3))

    # display_image(face_img)
    reshaped = np.vstack([reshaped])
    return reshaped


def present_detection_and_classification(image, x, y, w, h):
    label_value = np.argmax(result, axis=1)[0]
    mark_face(image, label_value, x, y, w, h)
    label_name = label_face_mask(image, label_value, x, y)
    return label_value, label_name


def label_face_mask(image, label_value, x, y):
    """
    Add Label with FaceMask status to camera preview
    """
    labels_dict = {0: 'correct', 1: 'incorrect', 2: 'no_mask'}
    label_name = labels_dict[label_value]
    cv2.putText(image, label_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return label_name


def mark_face(image, label_value, x, y, w, h):
    """
    Draw Rectangles around the face
    """
    color_dict = {0: (0, 255, 0), 1: (255, 255, 0), 2: (255, 0,)}
    cv2.rectangle(image, (x, y), (x + w, y + h), color_dict[label_value], 2)
    cv2.rectangle(image, (x, y - 40), (x + w, y), color_dict[label_value], -1)


size = 4

webcam = cv2.VideoCapture(0)  # Use camera 0

# We load the xml file. @Andrzej here are more of them https://github.com/opencv/opencv/tree/master/data/haarcascades
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = keras.models.load_model('saved_models/first-best-model-0.9969')

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 1)  # Flip to act as a mirror
    # display_image(im_rgb)

    SIZE = 256
    IMAGE_SIZE = (SIZE, SIZE)
    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (_x, _y, _w, _h) = [v * size for v in f]  # Scale the shapesize backup
        (_x, _y, _w, _h) = (index_scale(_x, 0.90), index_scale(_y, 0.70), index_scale(_w, 1.2), index_scale(_h, 1.25))

        prepared_image = prepare_for_classification(im, _x, _y, _w, _h, IMAGE_SIZE)

        result = model.predict(prepared_image)

        present_detection_and_classification(im, _x, _y, _w, _h)

    # Show the image
    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()
