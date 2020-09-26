import io
import base64
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np



class syndicai(object):
    
    def __init__(self):
        self._model = None

    def predict(self, X, features_names=None):

        input_image = io.BytesIO(base64.b64decode(X))

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = tensorflow.keras.models.load_model('keras_model.h5')

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(input_image)

        # read labels
        labels = np.loadtxt('labels.txt', dtype=str).tolist()

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        # image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data).tolist()[0]

        output = {}
        for x in range(0, len(prediction)):
            output[labels[x]] = prediction[x]

        print(output)

        return [output]


    def metrics(self):
        return [{"type": "COUNTER", "key": "mycounter", "value": 1}]

