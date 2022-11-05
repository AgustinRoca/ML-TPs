import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
class SupportVectors:

    def __init__(self,from_sv1,from_sv2,to_sv,to_class):
        self.to_class = to_class
        self.from_sv1 = from_sv1
        self.from_sv2 = from_sv2
        self.to_sv = to_sv
        self.slope = self.calculate_slope(from_sv1[:2],from_sv2[:2])
        from_intercept = self.calculate_intercept(self.slope,from_sv1[:2])
        to_intercept =  self.calculate_intercept(self.slope,to_sv[:2])
        self.class_1_intercept,self.class_2_intercept = (from_intercept,to_intercept) if self.to_class == -1 else (to_intercept,from_intercept) 
        self.mid_intercept = (from_intercept+to_intercept)/2

        self.normal_vector = np.array([-self.slope,1])/np.linalg.norm([-self.slope,1])
    

    def is_valid(self,entries,classes):

        return self._separates_correctly(entries,classes) and self._has_no_entries_between_SVs(entries[classes  == 1],entries[classes  == -1])

    def _has_no_entries_between_SVs(self, class_1_entries, class_2_entries):
      
        #print("class_1_entries: ",class_1_entries)
        #print((np.matmul(class_1_entries,[-self.slope,1])-self.class_1_intercept))
        #print("class_2_entries: ",class_2_entries)
        #print((np.matmul(class_2_entries,[-self.slope,1])-self.class_2_intercept))
        # Check if no entries are between the support vectors and the optimal H

        return np.all((np.matmul(class_1_entries,[-self.slope,1])-self.class_1_intercept) >= -1e-15) and np.all((np.matmul(class_2_entries,[-self.slope,1])-self.class_2_intercept) <= 1e-15)

    def _separates_correctly(self,entries,classes):
        return np.all(classes*(np.matmul(entries,[-self.slope,1])-self.mid_intercept) >= 0)

    def get_margin(self,points):

        return np.sort(np.abs(np.matmul(points-np.array([0,self.mid_intercept]), self.normal_vector)))[0]

    def __str__(self) -> str:
        return f"(slope = {self.slope}, class_1_intercept = {self.class_1_intercept}, mid_intercept = {self.mid_intercept}, class_2_intercept = {self.class_2_intercept})"

    def calculate_slope(self,p1,p2):

        slope = (p2[1] - p1[1])/(p2[0]-p1[0])
        #TODO chequear caso que p2[0]-p1[0] sea 0 ? ==> m es infinito y seria recta vertical
    
        return slope

    
    def calculate_intercept(self,slope,p):
        return  p[1]-slope*p[0]

    def plot_Hs(self,entries,classes,axes,box_limits=[0.0,5.0],title = None):
        #colors = list(map(lambda c:  'ro' if (c == 1.0) else 'bo',classes))
    
        #plt.plot(entries[:,0], entries[:,1], colors)
      

        axes.set_xlim( box_limits)
        axes.set_ylim( box_limits)
        handles = [Line2D(range(1), range(1), marker='o', markerfacecolor="red", color='white', label='1'),
                       Line2D([0], [0], marker='o', markerfacecolor="blue", color='white', label='-1')]
        axes.legend(handles=handles, loc='lower right')
    
        Xs,Ys = box_limits,self.slope*np.array(box_limits)+self.class_1_intercept 
        axes.plot(Xs,Ys,"r--",alpha=0.5)
        Xs,Ys = box_limits,self.slope*np.array(box_limits)+self.mid_intercept 
        axes.plot(Xs,Ys,"g-",alpha=0.5)
        Xs,Ys = box_limits,self.slope*np.array(box_limits)+self.class_2_intercept
        axes.plot(Xs,Ys,"b--",alpha=0.5)

        class_1_idx  = classes == 1
        class_2_idx = classes == -1
        
        axes.plot(entries[:, 0][class_1_idx], entries[:, 1][class_1_idx], 'ro')
        axes.plot(entries[:, 0][class_2_idx], entries[:, 1][class_2_idx], 'bo')
    

        axes.scatter(self.from_sv1[0], self.from_sv1[1], s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k")
        axes.scatter(self.from_sv2[0], self.from_sv2[1], s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k")
        axes.scatter(self.to_sv[0], self.to_sv[1], s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k")

        if title is not None:
            axes.set_title(title)
       
        