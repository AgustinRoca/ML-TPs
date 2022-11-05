
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (confusion_matrix, 
                           accuracy_score)
from sklearn.preprocessing import StandardScaler

from utils.WKNN import WKNN
import statsmodels.api as sm
class LogisticRegression():
    def __init__(self, df):
        print("Ej b")

        self.df = df
        null_columns = self.df.isnull().any()
        if np.any(null_columns) != False:
            print("Has missing values")
            print("null columnms =",null_columns)
            self.df = self.complete_missing_data(self.df,null_columns)
            self.df.to_csv('TP4/Data/acath_no_nan.csv',sep=';')
        self.target_attr = 'sigdz'
        print("Target attribute: ",self.target_attr)
        self.scaler = StandardScaler()
        


    def plot_logit(self,result,x_test,y_test,title=None):
        params = result.params
        print("Params = ",params)

        prediction = result.predict(x_test)
        prediction_outputs = list(map(round, prediction))

        is_prediction_ok = prediction_outputs == y_test
        #print("x_test = \n",x_test)
        ages = x_test[:,1]
        cholestes = x_test[:,2]
        # print("ages = \n",ages)
        # print("choleste = \n",cholestes)
        ax = plt.gca()
        colors = np.array(list(map(lambda value: 'r' if value == 1 else 'g',prediction_outputs)))

        wrong_prediction = ~is_prediction_ok
        # for i,is_ok in enumerate(is_prediction_ok):
        #     if not is_ok:
        #         print(fmts[i])
        #         fmts[i][1] = 'x'
        #     fmts[i] = "".join(fmts[i])
        # print(fmts)
        ax.scatter(ages[is_prediction_ok],cholestes[is_prediction_ok],c = colors[is_prediction_ok],marker = 'o')
        ax.scatter(ages[wrong_prediction],cholestes[wrong_prediction],c = colors[wrong_prediction],marker = 'x')

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x1 = x_lim[0]
        x2 = x_lim[1]
        m = params[1]/-params[2]
        b = params[0]/-params[2]
        print(f"m  = {m}, b = {b}")
        y1 =  m*x1 + b
        y2 = m*x2 + b

        ax.plot([x1,x2],[y1,y2],'k--')

        plt.xlabel("Age")
        plt.ylabel("Choleste")
        if title:
            plt.title(title)
        handles = [Line2D([0],[0], marker='o', markerfacecolor="red", color='white', label='Predicted sigdz = 1'),
                       Line2D([0], [0], marker='o', markerfacecolor="green", color='white', label='Predicted sigdz = 0'),
                       Line2D([0], [0], marker='X', markerfacecolor="red", color='white', label='Actual sigdz = 0'),
                       Line2D([0], [0], marker='X', markerfacecolor="green", color='white', label='Actual sigdz = 1'),
                       Line2D([0], [0], linestyle='--', color="black", label='Logit')]
        plt.legend(handles=handles, loc='lower right')
        plt.show()
   
       
    def ejC(self):
        entry_columns = ['age','choleste']

        print("\n########################## Sex = M ##########################\n")
        M_entries = self.df[self.df['sex'] == 0]
        M_expected_outputs = M_entries[self.target_attr].to_numpy()
        M_entries = M_entries.loc[:,entry_columns].to_numpy().reshape(-1,len(entry_columns))
        print("M entries len = ",len(M_entries))

        min_error,avg_errors,best_confusion_matrix,best_x_train,best_y_train,best_x_test,best_y_test,best_result = self.cross_validate(M_entries,M_expected_outputs,50)
        print("Min error = ",min_error)
        print("Best accuracy = ",1-min_error)
        print("Avg Error = ",avg_errors)
        print("Best CM  = ",best_confusion_matrix)
        print("Best params = ",best_result.params)
        print(best_result.summary())
        print(best_result.summary2())
        self.plot_logit(best_result,best_x_test,best_y_test,f"Logit con sólo hombres ({len(M_entries)})")

        print("\n########################## Sex = F ##########################\n")
        F_entries = self.df[self.df['sex'] == 1]
        F_expected_outputs = F_entries[self.target_attr].to_numpy()
        F_entries = F_entries.loc[:,entry_columns].to_numpy().reshape(-1,len(entry_columns))

        print("F entries len = ",len(F_entries))
        min_error,avg_errors,best_confusion_matrix,best_x_train,best_y_train,best_x_test,best_y_test,best_result = self.cross_validate(F_entries,F_expected_outputs,50)
        print("Min error = ",min_error)
        print("Best accuracy = ",1-min_error)
        print("Avg Error = ",avg_errors)
        print("Best CM  = ",best_confusion_matrix)
        print("Best params = ",best_result.params)
        print(best_result.summary())
        print(best_result.summary2())
        self.plot_logit(best_result,best_x_test,best_y_test,f"Logit con sólo mujeres ({len(F_entries)})")

    def ejB(self):
        print("\n########################### Ej B ###########################\n")
        expected_outputs = self.df[self.target_attr].to_numpy()
        print("Expected outputs: ",expected_outputs)
        #self.df.insert(0,"B0",np.ones(len(self.df.index)).astype(int),True)
        entry_columns = ['age','cad.dur','choleste']
        print("\n\n########################################################################################\n\n")
        print("With variables: ",entry_columns)
        entries = self.df.loc[:,entry_columns].to_numpy().reshape(-1,len(entry_columns))
        print("Entries: \n",self.df[entry_columns])
        
     
        min_error,avg_errors,best_confusion_matrix,best_x_train,best_y_train,best_x_test,best_y_test,best_result = self.cross_validate(entries,expected_outputs,50)
        print("Min error = ",min_error)
        print("Best accuracy = ",1-min_error)
        print("Avg Error = ",avg_errors)
        print("Best CM  = ",best_confusion_matrix)
        print("Best params = ",best_result.params)
        print(best_result.summary())
        print(best_result.summary2())
        #print("best x train = ",best_x_train)
        x_train  = np.delete(best_x_train,2,axis=1)
        x_test  = np.delete(best_x_test,2,axis=1)
        # print("x train without cad.dur = \n",x_train)
        print("\n############################## Without CAD.DUR ##############################\n")
        result,accuracy,cm = self.solve_logistic_regression(x_train,best_y_train,x_test,best_y_test)
        print("params = ", result.params)
        print("accuracy = ",accuracy)
        print("cm = ",cm)
        print(result.summary())
        print(result.summary2())

        self.plot_logit(result,x_test,best_y_test)



    def _get_partition_indexes(self,entries,K):
            entries_len = len(entries)      
            partition_indexes = np.arange(0, entries_len)
            np.random.shuffle(partition_indexes)
        
            return np.array_split(partition_indexes, K)

    def _get_test_and_train(self,entries,expected_outputs,partition_indexes, i):
        test_indexes = partition_indexes[i]
        x_test = entries[test_indexes]
        y_test = expected_outputs[test_indexes]
        train_indexes = np.delete(np.arange(0, len(entries)), test_indexes)
        x_train = entries[train_indexes]
        y_train = expected_outputs[train_indexes]
        return x_train,y_train,x_test,y_test

    def cross_validate(self,entries,expected_outputs,K):
       
        min_error = None
        best_x_train = None
        best_y_train = None
        best_x_test = None
        best_y_test = None
        best_result = None
        best_confusion_matrix = None
        errors = list()

        partition_indexes = self._get_partition_indexes(entries,K)
       
        for i in range(K):
            x_train,y_train,x_test,y_test =  self._get_test_and_train(entries,expected_outputs,partition_indexes,i)
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.fit_transform(x_test)
            x_train = sm.add_constant(x_train)
            x_test = sm.add_constant(x_test)
            
            # print("x_train = ",x_train)
            # print("y_test = ",y_test)
            result,accuracy,cm = self.solve_logistic_regression(x_train,y_train,x_test,y_test)
        
            error = 1-accuracy
            errors.append(error)
            if (not min_error) or (error < min_error):
                min_error = error
                best_x_train = x_train
                best_y_train = y_train
                best_x_test = x_test
                best_y_test = y_test
                best_confusion_matrix = cm
                best_result = result
                
        
        return min_error,np.mean(errors),best_confusion_matrix,best_x_train,best_y_train,best_x_test,best_y_test,best_result
        
    def solve_logistic_regression(self,x_train,y_train,x_test,y_test):
        #model = LogisticRegression(solver='liblinear', random_state=0).fit(self.entries,self.expected_outputs )
                
   
        result = sm.Logit(y_train,x_train).fit(method='newton')

        #print("result params = ",result.params)
        
        prediction = list(map(round, result.predict(x_test)))
        #print("prediction = ",prediction)
    
        cm = confusion_matrix(y_test, prediction) 
        #print ("Confusion Matrix : \n", cm) 
  
        # accuracy score of the model
        accuracy =  accuracy_score(y_test, prediction)
        #print('Test accuracy = ',accuracy)
        #print("correct prediction = ",np.array_equal(prediction,y_test))
        # print(result.summary())
        # print(result.summary2())
        return result,accuracy,cm

     


    def complete_missing_data(self,df,null_columns):
        for category_label in self.df.loc[:,null_columns]:
            print("Missing Category label:", category_label)

            wknn = WKNN(1,  category_label, np.array([0,1]))
            neighbours = df[df[category_label].notnull()]
            #print("neighbours data: \n",neighbours)
            incomplete_data_set  = df[df[category_label].isnull()]
            #print("incomplete data: \n",incomplete_data_set)
            wknn.train(neighbours)
            for i,incomplete_data in incomplete_data_set.iterrows():
                #print("incomplete data: \n",incomplete_data)
                incomplete_data  = incomplete_data.drop(category_label)
                closest_index = wknn.get_closests(incomplete_data).head(1).index
                
                df.at[i,category_label] = self.df.loc[closest_index][category_label]
                #print("New data: \n",self.df.loc[i])
        return df
        
    def run(self):
        pass
     

