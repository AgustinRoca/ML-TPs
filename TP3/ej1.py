from copy import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import datasets
import numpy as np
from support_vectors import SupportVectors
from StepSimplePerceptron import StepSimplePerceptron
from celluloid import Camera

CLASS_1 = 1
CLASS_2 = -1
class Ej1():
    def __init__(self):
        print("Ej 1")
    
        self.N = 30
      
        self.dataset,self.dataset_classes,self.N = self.generate_linear_separable_dataset(self.N,plot = True)
        self.learn_factor = 0.1
        self.limit = 29
        self.step_simple_perceptron = StepSimplePerceptron()
       

    def run(self):
        # Punto a)
        print("########### Punto a) ###########")
        print("Dataset:\n",self.dataset)
        print("Expected outputs: \n",self.dataset_classes)
        weights_min,error_min = self.step_simple_perceptron.fit(self.dataset,self.dataset_classes,self.learn_factor,self.limit,plot = True,gif_title="linear_separable.gif",last_title="linear_perceptron_simple.png")
        prediction = self.step_simple_perceptron.predict(self.dataset)
        print("prediction: ",prediction)
        error = np.sum((self.dataset_classes - prediction)**2)
        print("prediction error: ",error)
        
        # Punto b)
        print("########### Punto b) ###########")
        self.find_optimal_H(self.dataset,self.dataset_classes,weights_min)

        # Punto c)
        print("########### Punto c) ###########")
        non_linear_dataset, non_linear_classes = self.generate_almost_linear_separable_dataset(self.dataset, self.dataset_classes, weights_min,plot=True)
        weights_min,error_min = self.step_simple_perceptron.fit(non_linear_dataset,non_linear_classes,self.learn_factor,self.limit,plot = True,gif_title="not_linear_separable.gif",last_title="not_linear_perceptron_simple.png")
        prediction = self.step_simple_perceptron.predict(non_linear_dataset)
        print("prediction: ",prediction)
        error = np.sum((non_linear_classes - prediction)**2)
        print("prediction error: ",error)

        # Punto d)
        print("########### Punto d) ###########")
        self.svm_classification(self.dataset, self.dataset_classes, self.dataset, self.dataset_classes,C=10)
        self.svm_classification(self.dataset, self.dataset_classes, self.dataset, self.dataset_classes,C=1000)
        
        self.svm_classification(non_linear_dataset, non_linear_classes, non_linear_dataset, non_linear_classes,C=10)
        self.svm_classification(non_linear_dataset, non_linear_classes, non_linear_dataset, non_linear_classes,C=1000)

    def generate_almost_linear_separable_dataset(self, dataset, classes, weights, plot=False):
        out_dataset = copy(dataset)
        out_classes = copy(classes)

        entries_aux =  np.concatenate([dataset, -1*np.ones((dataset.shape[0], 1))], axis=1)
        class_1_entries = entries_aux[classes == CLASS_1]
        class_1_distances = np.abs(np.matmul(class_1_entries, weights))
        min_N_class_1_distances_idx = class_1_distances.argsort()[5:7]
        class_1_SVs = class_1_entries[min_N_class_1_distances_idx][:,:2]
        class1_indexes = []
        for sv in class_1_SVs:
            class1_indexes.append(np.where(out_dataset == sv)[0][0])

        class_2_entries = entries_aux[classes == CLASS_2]
        class_2_distances = np.abs(np.matmul(class_2_entries, weights))
        min_N_class_2_distances_idx = class_2_distances.argsort()[5:7]
        class_2_SVs = class_2_entries[min_N_class_2_distances_idx][:,:2]
        class2_indexes = []
        for sv in class_2_SVs:
      
            class2_indexes.append(np.where(out_dataset == sv)[0][0])

        for index in class1_indexes:
            out_classes[index] = CLASS_2

        for index in class2_indexes:
            out_classes[index] = CLASS_1

        if plot:
            class_1_idx  = out_classes == CLASS_1
            class_2_idx = out_classes == CLASS_2
            plt.grid(True)
            plt.plot(out_dataset[:, 0][class_1_idx], out_dataset[:, 1][class_1_idx], 'ro')
            plt.plot(out_dataset[:, 0][class_2_idx], out_dataset[:, 1][class_2_idx], 'bo')
            line_xs = np.arange(0,6)
            plt.plot(line_xs,self.y(line_xs,1,0),"g")
            plt.xlim( [0.0,5.0])
            plt.ylim( [0.0,5.0])
            plt.legend()
            plt.show()
        
        return out_dataset, out_classes
        

    def find_optimal_H(self,entries,classes,weights):
        N = 4
        
        entries_aux =  np.concatenate([entries, -1*np.ones((entries.shape[0], 1))], axis=1)
        class_1_entries = entries_aux[classes == CLASS_1]
        #print("entries class 1:\n ",class_1_entries)
        class_1_distances = np.abs(np.matmul(class_1_entries, weights))
        
        #print("class_1_distances: ",class_1_distances)
        min_N_class_1_distances_idx = class_1_distances.argsort()[:N]
        #print("min_N_class_1_distances indexes: ",min_N_class_1_distances_idx)

        class_2_entries = entries_aux[classes == CLASS_2]
        #print("entries class 2:\n ",class_2_entries)
        
        class_2_distances  = np.abs(np.matmul(class_2_entries, weights))
       
       # print("class_2_distances: ",class_2_distances)
        min_N_class_2_distances_idx = class_2_distances.argsort()[:N]
       # print("min_N_class_2_distances indexes: ",min_N_class_2_distances_idx)

        class_1_SVs = class_1_entries[min_N_class_1_distances_idx][:,:2]
        class_2_SVs = class_2_entries[min_N_class_2_distances_idx][:,:2]
        # From Class 1 to Class -1
        self.calculate_optimal_margin(class_1_SVs,CLASS_1,class_2_SVs,CLASS_2,entries,classes,plot=True,gif_title="optimal_margin_from_1.gif")

        # From Class -1 to Class 1
        self.calculate_optimal_margin(class_2_SVs,CLASS_2,class_1_SVs,CLASS_1,entries,classes,plot=True,gif_title="optimal_margin_from_2.gif")
           
    
    def calculate_optimal_margin(self,from_class_SVs,from_class,to_class_SVs,to_class,entries,classes,plot=True,gif_title="optimal_margin.gif"):
        H_SVs = []
        for i in range(len(from_class_SVs)):
            for j in range(i+1,len(from_class_SVs)):
               
                H_SVs.append([from_class_SVs[i],from_class_SVs[j]])


        SVs_list = self.calculate_Hs_from_SVs(H_SVs,to_class_SVs,to_class)

        optimal_margin = 0
        optimal_margin_idx = None
        fig, axes = plt.subplots()
        plt.grid(True)

        camera = Camera(fig)
        for idx,SVs in enumerate(SVs_list):
            #print(f"{idx}) SVs: ",SVs)

            if SVs.is_valid(entries,classes):
            
                margin = SVs.get_margin(entries)
                #print("Margin: ",margin)
                if margin > optimal_margin:
                    optimal_margin = margin
                    optimal_margin_idx = idx
     
            if plot:
                
                SVs.plot_Hs(entries,classes,axes=axes)
                camera.snap()
                      
        if plot:
            animation = camera.animate(interval=1000, repeat=False)
            animation.save(gif_title)
            plt.show()
            
        if optimal_margin_idx is not None:
            fig,axes = plt.subplots()
            plt.grid(True)
            optimal_SVs = SVs_list[optimal_margin_idx]
            print(f"From Class {from_class} to Class {to_class} Optimal margin = ",optimal_margin)
            print(f"From Class {from_class} to Class {to_class} Optimal SVs",optimal_SVs)
            optimal_SVs.plot_Hs(entries,classes,axes=axes,title = f"From Class {from_class} to Class {to_class} Optimal H")
            plt.show()
       
        else:
            print(f"From Class {from_class} to Class {to_class} Optimal H not found")

    def calculate_Hs_from_SVs(self,from_class_SVs,to_class_SVs,to_class):
        Hs_list = []
        for from_sv1, from_sv2 in from_class_SVs:
           
            for to_sv in to_class_SVs:
                Hs_list.append(SupportVectors(from_sv1,from_sv2,to_sv,to_class))
        return Hs_list

       


    def y(self,x,m,b):
        return m*x + b

    def generate_linear_separable_dataset(self,N,box_limits = [0,5],plot = False):
        print("N = ",N)

        X =  []
        for i in range(N):
            x = np.random.default_rng().random()*box_limits[1]
            y = np.random.default_rng().random()*box_limits[1]
            X.append([x,y])
        X = np.array(X)
       # we create 40 separable points
        # X, classes = datasets.make_blobs(n_samples=N, centers=2, random_state=6)
        # print("classes: ",classes)
       
        # classes = np.where(classes != 0,classes,-1)
        # print("classes: ",classes)
        # if plot:
        #     plt.scatter(X[:, 0], X[:, 1], c=classes, s=30, cmap=plt.cm.Paired)
        #     plt.show()
        
        classes = np.ones((N,))
      
        
        for i,point in enumerate(X):
            if point[1] <= self.y(point[0],1,0):
                classes[i] = CLASS_2
        if plot:
            plt.grid(True)
            class_1_idx  = classes == CLASS_1
            class_2_idx = classes == CLASS_2
        
            plt.plot(X[:, 0][class_1_idx], X[:, 1][class_1_idx], 'ro')
            plt.plot(X[:, 0][class_2_idx], X[:, 1][class_2_idx], 'bo')
            line_xs = np.arange(0,6)
            plt.plot(line_xs,self.y(line_xs,1,0),"g",alpha=0.5)
            plt.xlim( box_limits)
            plt.ylim( box_limits)
            plt.title(f"Linear Separable Dataset (N = {self.N})")
            handles = [Line2D(range(1), range(1), marker='o', markerfacecolor="red", color='white', label='1'),
                       Line2D([0], [0], marker='o', markerfacecolor="blue", color='white', label='-1')]
            plt.legend(handles=handles, loc='lower right')
            plt.show()
       
        return X,classes,N



    def svm_classification(self, x_train, y_train, x_test, y_test,C =1000):
        #Import svm model
        from sklearn import svm
        from sklearn.inspection import DecisionBoundaryDisplay
        
        #Create a svm Classifier
        clf = svm.SVC(kernel='linear',C=C) # Linear Kernel

        #Train the model using the training sets
        clf.fit(x_train, y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(x_test)

        #Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics

        # Model Accuracy: how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        # plot the decision function
        plt.grid(True)
        ax = plt.gca()
        class_1_idx  = y_train == CLASS_1
        class_2_idx = y_train == CLASS_2
        plt.plot(x_train[:, 0][class_1_idx], x_train[:, 1][class_1_idx], 'ro')
        plt.plot(x_train[:, 0][class_2_idx],x_train[:, 1][class_2_idx], 'bo')
        DecisionBoundaryDisplay.from_estimator(
            clf,
            x_train,
            plot_method="contour",
            colors=["b","g","r"],
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
            ax=ax,
        )
        # plot support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k",
        )
        plt.title(f"SVM with C = {C}")
        handles = [Line2D(range(1), range(1), marker='o', markerfacecolor="red", color='white', label='1'),
                       Line2D([0], [0], marker='o', markerfacecolor="blue", color='white', label='-1')]
        plt.legend(handles=handles, loc='lower right')
        
        plt.show()