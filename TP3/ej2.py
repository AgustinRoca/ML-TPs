
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split

from ConfusionMatrix import ConfusionMatrix

class Ej2():
    def __init__(self, files_path='Data'):
        print("Ej 2")
        sky_train = Image.open(files_path + '/cielo.jpg')
        grass_train = Image.open(files_path + '/pasto.jpg')
        cow_train = Image.open(files_path + '/vaca.jpg')
        self.image_test = Image.open(files_path + '/cow.jpg')
        self.google_image = Image.open(files_path + '/cows_google.jpg')

        sky_pixels = []
        for x in range(sky_train.size[0]):
            for y in range(sky_train.size[1]):
                pixel = sky_train.getpixel((x, y))
                sky_pixels.append(pixel)
        sky_classes = ['Sky'] * len(sky_pixels)

        grass_pixels = []
        for x in range(grass_train.size[0]):
            for y in range(grass_train.size[1]):
                pixel = grass_train.getpixel((x, y))
                grass_pixels.append(pixel)
        grass_pixels = list(set(grass_pixels))
        grass_classes = ['Grass'] * len(grass_pixels)

        cow_pixels = []
        for x in range(cow_train.size[0]):
            for y in range(cow_train.size[1]):
                pixel = cow_train.getpixel((x, y))
                cow_pixels.append(pixel)
        cow_pixels = list(set(cow_pixels))
        cow_classes = ['Cow'] * len(cow_pixels)

        data = {'Pixel': sky_pixels + grass_pixels + cow_pixels, 'Class': sky_classes + grass_classes + cow_classes}

        # Create DataFrame  
        self.df = pd.DataFrame(data)   
        self.clf = None

    def run(self, kernel='linear'):   
        x = list(self.df['Pixel'])
        y = list(self.df['Class'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15) # 80% training and 20% test
        kernels = ['linear', 'poly', 'rbf'] # cant use precomputed because input is not a square matrix, sigmoid very bad and waste a lot of time, not worth
        cs = [0.001, 0.01, 0.1, 1, 10]
        errors = {}
        
        for kernel in kernels: 
            errors[kernel] = {}
            for c in cs:
                print(kernel, c)
                self.train(x_train, y_train, kernel, c)
                print('Training done')
                avg_error = self.test(x_test, y_test)
                errors[kernel][c] = avg_error
            cs_, errors_ = zip(*errors[kernel].items())
            cs_ = list(cs_)
            errors_ = list(errors_)
            plt.plot(cs_, errors_)
            plt.scatter(cs_, errors_, label=kernel)
        plt.title("Errores a distintos Kernels y C")
        plt.xlabel("C")
        plt.ylabel("Error")
        plt.xscale('log')
        plt.legend()
        plt.show()
        
        min_error = None
        best_kernel = None
        best_c = None
        for kernel in kernels:
            for c in cs:
                error = errors[kernel][c]
                if min_error is None or error < min_error:
                    min_error = error
                    best_kernel = kernel
                    best_c = c

        self.train(x_train, y_train, best_kernel, best_c)
        self.classify_image(self.image_test, out_path=f'out_{best_kernel}_{best_c}.png')
        self.classify_image(self.google_image, out_path=f'out_{best_kernel}_{best_c}_google.png')
        

    def train(self, x_train, y_train, kernel='linear', C=1):
        self.clf = svm.SVC(kernel=kernel, C=C)
        self.clf.fit(x_train, y_train)

    def test(self, x_test, y_test):
        ys_predicted = self.clf.predict(x_test)
        cm = ConfusionMatrix(['Cow', 'Grass', 'Sky'])
        for y, y_expected in zip(ys_predicted, y_test):
            cm.add_result(y_expected, y)
        print(cm)
        errors = cm.calculate_errors()
        accuracies = {k: 1-v for k,v in errors.items()}
        print('Accuracy:', accuracies)
        return np.mean(list(errors.values()))

    def classify_image(self, image, out_path='out.png'):
        image_pixels = []
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                image_pixels.append(list(image.getpixel((x, y))))

        #Predict the response for test dataset
        y_pred = np.array(self.clf.predict(image_pixels))
        y_pred = np.reshape(y_pred, image.size)
        output_image_pixels = np.empty((image.size[0],image.size[1],3), dtype=np.uint8)
        for row in range(len(y_pred)):
            for column in range(len(y_pred[row])):
                color = self.class_to_color(y_pred[row][column])
                output_image_pixels[row][column][0] = color[0]
                output_image_pixels[row][column][1] = color[1]
                output_image_pixels[row][column][2] = color[2]

        img = Image.fromarray(output_image_pixels, 'RGB')
        img = img.transpose(Image.TRANSPOSE)
        img.save(out_path)

    def class_to_color(self, clazz):
        if clazz == 'Sky':
            return (0, 0, 255)
        elif clazz == 'Grass':
            return (0, 255, 0)
        elif clazz == 'Cow':
            return (123,63,0)
        else:
            return (0,0,0)