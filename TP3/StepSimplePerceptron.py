import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from matplotlib.lines import Line2D


class StepSimplePerceptron:
    def __init__(self):
        self.weights = None
        self.weights_history = None
        self.epochs = 0
        self.limit = 0

    def activation(self,exitation):
        return 1 if exitation >= 0 else -1

    def fit(self,  entries, expected_outputs, learn_factor,limit,plot = True,gif_title="animation.gif",last_title="perceptron_simple.png"):
        self.limit = limit

        n_samples = entries.shape[0]
        #print("n_samples: ", n_samples)
       
        n_features = entries.shape[1]
        #print("n_features: ", n_features)

        # Concateno un -1 a cada ejemplo de entrenamiento
        x = np.concatenate([entries, -1*np.ones((n_samples, 1))], axis=1)
        #print("entries ", x)
        #print("classes:", expected_outputs)
        # Creo los weights inicializados en 0 y pongo el w0 en 1
        self.weights = np.zeros((n_features+1,))
        # self.weights = np.random.random_sample(n_features + 1) * 2 - 1
        self.weights[-1] = 1
        self.weights_history = [self.weights]
        #print("initial weights: ",self.weights)
    
        error_min = n_samples * 2 
        self.epochs = 0

        while error_min > 0  and self.epochs < limit:

            #print(f"\n################### Epoch: {self.epochs} #####################")
            for j in range(n_samples):
                entry = x[j, :]
                #print(f"{j}) entry[{j}]", entry)
                #print(f"{j}) weights: ",self.weights)
                exitation  = np.dot(entry,self.weights)
                #print(f"{j}) exitation: ",exitation)

                activation = self.activation(exitation)
                #print(f"{j}) activation: ",activation)
                #print(f"{j}) expected output: ",expected_outputs[j])
                # Si expeced output == activation entonces no se aprende nada
                delta_w = learn_factor*(expected_outputs[j] - activation)*entry
                #print(f"{j}) delta w : ",delta_w)
                self.weights += delta_w
                #print(f"{j}) new weights: ",self.weights)

                self.weights_history.append(np.copy(self.weights))
                quadratic_error = self.calculate_error(x, expected_outputs)
                #print(f"{j}) quadratic error: ",quadratic_error)
                if quadratic_error < error_min:
                    #print("new error min: ",quadratic_error)
                    error_min = quadratic_error
                    weights_min = np.copy(self.weights)
                    
            self.epochs += 1
     
        print("Epochs: ", self.epochs)
        print("Min Error: ", error_min)
        print("Min Weights: ",weights_min)
        if plot:
         self.plot(entries, expected_outputs,gif_title=gif_title,last_title=last_title)
        return weights_min,error_min

    def calculate_error(self, entries,expected_outputs):
        error = 0
        # x = (1,2)
     
        for i in range(len(entries)):
          
            activation = self.activation(np.dot(self.weights, entries[i, :]))
            error += (expected_outputs[i] - activation)**2
        return error

    def predict(self, entries):
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return
        n_samples = entries.shape[0]
        # Add column of 1s
        x = np.concatenate([entries, -1*np.ones((n_samples, 1))], axis=1)

        y = np.matmul(x, self.weights)
      
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)

        return y

    def plot(self, entries, expected_outputs,box_limits=[0,5],gif_title="animation.gif",last_title = "perceptron_simple.png"):
      
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return
        n_features = entries.shape[1]
        if n_features != 2:
            print('n_features must be 2')
            return

        fig, axes = plt.subplots()
        plt.grid(True)

        camera = Camera(fig)
  
        entry_len = len(entries)
        handles = [Line2D(range(1), range(1), marker='o', markerfacecolor="red", color='white', label='1'),
                       Line2D([0], [0], marker='o', markerfacecolor="blue", color='white', label='-1')]

        wh_len  =len(self.weights_history)
        for i in range(wh_len):
            weights = self.weights_history[i]
        
            for entry, target in zip(entries, expected_outputs):
                plt.plot(entry[0], entry[1], 'ro' if (target == 1.0) else 'bo', label='%i' % target)

            y = np.array([])
            x = np.array([])

            slope = -weights[0]/weights[1]
            intercept = weights[2] / weights[1]
            k = 0

            delta = np.array([-1, 6])

            for xi in delta:
                y1 = ((slope * xi) + intercept)
                if delta[0] <= y1 <= delta[1]:
                    x = np.append(x, [xi])
                    y = np.append(y, [y1])
                    k += 1
                    if k == 2:
                        break
            if k < 2:
                for yi in delta:
                    x1 = (yi - intercept) / slope
                    if delta[0] <= x1 <= delta[1]:

                        x = np.append(x, [x1])
                        y = np.append(y, [yi])
                        k += 1
                        if k == 2:
                            break
            #print("Xs: ",x,"  Ys: ", y)
            epoch = int(i / entry_len)
            plt.text(0.5, 1.01, f"Epochs: {epoch+1}/{self.limit+1}", horizontalalignment='center',
                     verticalalignment='bottom',
                     transform=axes.transAxes)
            plt.xlim( box_limits)
            plt.ylim( box_limits)
            plt.legend(handles=handles, loc='lower right')
            plt.plot(x, y, 'k')
            # if i == wh_len-1:
            #     plt.savefig(last_title)

            camera.snap()
           
        animation = camera.animate(interval=50, repeat=False)
        animation.save(gif_title)
        plt.show()
