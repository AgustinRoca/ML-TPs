from asyncio.windows_events import NULL
from matplotlib.style import use
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from time import process_time_ns
from functools import reduce
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class NaiveBayes:

    def __init__(self,df,attribute_values,target_class):
        self.df = df
        self.target_class = target_class
        self.target_class_series = df[target_class]
       
        # probabilidades a priori
        self.target_class_probs = self.target_class_series.value_counts() / len(self.target_class_series)
        self.target_class_values = self.target_class_series.unique()
        self.target_class_total_rows = dict()
        

        self.attribute_values = attribute_values

        self.attributes_probs = self.calculate_attributes_relative_freqs2()
  
        

    def calculate_attributes_relative_freqs(self):
        attributes_relative_freqs = dict()
     
        for target_class_value in self.target_class_values:

            #print("Calculating relative freqs for class: ",target_class_value)
            start = process_time_ns()
            attributes_relative_freqs[target_class_value] = dict()
            attributes = self.df[self.df[self.target_class] == target_class_value].drop([self.target_class],axis=1)
            #print(f"target class value: {target_class_value} ")
            total_rows = len(attributes)
               
            self.target_class_total_rows[target_class_value] = total_rows 
            #print((attributes["scones"].value_counts()+1)/(total_rows+2))
            #print((attributes.apply(pd.Series.value_counts)+1)/(total_rows+2))
       
            for attribute in attributes:
                attributes_relative_freqs[target_class_value][attribute] = dict()
                attribute_values = self.attribute_values[attribute]
                laplace_k  = len(attribute_values)           
           
                for attribute_value in attribute_values:
                    value_count = len(attributes[attributes[attribute]== attribute_value])
                
                    # Laplace correction
                    relative_freq = (value_count + 1)/(total_rows + laplace_k)
                    #print(f"value: {attribute_value}, count: {value_count}, prob: {relative_freq}")
                    attributes_relative_freqs[target_class_value][attribute][attribute_value] = relative_freq
          
        return attributes_relative_freqs

    def calculate_attributes_relative_freqs2(self):
        attributes_relative_freqs = dict()
        for target_class_value in self.target_class_values:
            
            print("Calculating relative freqs for class: ",target_class_value)
            attributes_relative_freqs[target_class_value] = dict()
            attributes = self.df[self.df[self.target_class] == target_class_value].drop(
                [self.target_class], axis=1)
            #print(f"target class value: {target_class_value} ")
            total_rows = len(attributes)    
            self.target_class_total_rows[target_class_value] = total_rows
        
            attributes_relative_freqs[target_class_value]=(attributes.apply(pd.Series.value_counts)+1)/(total_rows+2)
            attributes_relative_freqs[target_class_value]= attributes_relative_freqs[target_class_value].fillna(1/(total_rows+2))
        
        return attributes_relative_freqs

    def replace_nan(self,value,df):

        print("value: ",value, " type: ",type(value))
        if math.isnan(value):
            if math.isnan(df[0]):
                return 1- df[1]
            else:
                return 1 - df[0]
        else:
            return value
    #v_opt =  arg max {P(v_i| a_0,...,a_n)}
    def get_optimum_class_probability(self,new_instance,use_log):
        class_probabilities = dict()
        max_probability = 0
        for class_value in self.target_class_values:
            class_probability = self.class_of_given_instance_probability(
                new_instance, class_value, use_log)
           # print(f"{class_value}: proba =>{class_probability}")
            class_probabilities[str(class_probability)] = class_value
            if(class_probability > max_probability):
                max_probability = class_probability
        instance_prob = self.instance_probability(new_instance, use_log)
       # print("instance prob: ",instance_prob)
        return class_probabilities[str(max_probability)], max_probability/instance_prob


    # P(v_i| a_0,...,a_n) = P(a_0,...,a_n | v_i) * P(vi) 
    def class_of_given_instance_probability(self, new_instance, target_class_value,use_log=False):
        return self.instance_of_given_class_probability2(new_instance, target_class_value,use_log)


    # P(a_0,...,a_n)
    def instance_probability(self, new_instance,use_log=False):
        total_probability = 0
        for class_value in self.target_class_values:
            total_probability += self.instance_of_given_class_probability2(
                new_instance, class_value, use_log)
        return total_probability

    # P(a_0,...,a_n | v_i)
    def instance_of_given_class_probability(self,new_instance,target_class_value,use_log=False):
        probability = 1
        #print("new instance: \n",new_instance)
        #print("target class: ",target_class_value)
        attributes = self.attributes_probs[target_class_value]
       # print("attributes: \n",attributes)
       #print(f"new instance: {new_instance}")
        if use_log:
        
            # Al ser muchas columnas, las probabilidades son muy pequeñas y puede haber problemas de representación al multiplicar 2 numeros muy pequeños=> se calcula log de las probabilidades y se suman => se mantiene relación > y <
            for attribute in new_instance:
                if attribute == self.target_class:
                    continue
                #print(f"attribute: {attribute}")
                new_probability = attributes[attribute][new_instance[attribute]]
              
                # if math.isnan(new_probability):
                  
                #     if attribute == 0:
                #         new_probability = 1 -  attributes[attribute][1]
                #     else:
                #         new_probability = 1 - attributes[attribute][0]
                probability += math.log(new_probability)
            probability += math.log(self.target_class_probs[target_class_value])
            probability = math.exp(probability)
        else:
            #print("new instance: \n",new_instance)
            for attribute in new_instance:
                #print(f"new_instance attribute: {attribute}")
                if attribute == self.target_class:
                    continue
                #word_probs = attributes.get(attribute, None)
               
                #new_probability = 0
                # if word_probs == None:
                #     #no deberia pasar
                #     print(f"target class: {target_class_value} , attribute: {attribute}")
                #     new_probability = 1/(self.target_class_total_rows[target_class_value] + len(self.attribute_values[attribute]))
                # else:
                new_probability = attributes[attribute][new_instance[attribute]]
               
                if math.isnan(new_probability):
                  
                    if attribute == 0:
                        new_probability = 1 - attributes[attribute][1]
                    else:
                        new_probability = 1 - attributes[attribute][0]
                 
                probability *= new_probability
               
            probability *= self.target_class_probs[target_class_value]
        
        return probability

    

               
    def instance_of_given_class_probability2(self,new_instance,target_class_value,use_log=False):       
        probability = 1
   
        attributes = self.attributes_probs[target_class_value]
   
        if use_log:
            # Al ser muchas columnas, las probabilidades son muy pequeñas y puede haber problemas de representación al multiplicar 2 numeros muy pequeños=> se calcula log de las probabilidades y se suman => se mantiene relación > y <
        
            lookup = attributes.lookup(new_instance.values(), attributes.columns)
            lookup[0] = math.log(lookup[0])
            probability = reduce(lambda x, y: x+math.log(y), lookup) + math.log(self.target_class_probs[target_class_value])
    
            probability = math.exp(probability)
        else:

         
            lookup = attributes.lookup(new_instance.values(), attributes.columns)
     
            probability = reduce(lambda x, y: x*y, lookup)* self.target_class_probs[target_class_value]

        return probability
        
