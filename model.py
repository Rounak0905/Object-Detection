from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
from PIL import Image
import pyttsx3

class Model:

    def __init__(self):
        self.model = LinearSVC(max_iter=10000)  # Increased max_iter for convergence

    def train_model(self, counters):
        img_list = []
        class_list = []

        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            img = img.reshape(16950)
            img_list.append(img)
            class_list.append(1)

        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            img = img.reshape(16950)
            img_list.append(img)
            class_list.append(2)

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.model.fit(img_list, class_list)
        engine = pyttsx3.init()
        engine.say("Model successfully trained!")
        engine.runAndWait()
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = Image.open("frame.jpg")
        img.thumbnail((150, 150), Image.Resampling.LANCZOS)
        img.save("frame.jpg")

        img = cv.imread('frame.jpg', cv.IMREAD_GRAYSCALE)
        img = img.reshape(16950)
        prediction = self.model.predict([img])

        return prediction[0]
