from audioop import avg
import numpy as np
import pandas as pd
import re
from bayes.naive_bayes import NaiveBayes
import matplotlib.pyplot as plt
import TP2.metrics as metrics
PRESENT = 1
class Ej2:

    def __init__(self,df):
     
        #El numpy array de datos es de bytes porque sino ocupaba mucha memoria
        self.classes_to_index_dic = {  'Nacional': 0, 'Internacional': 1,
                            'Deportes': 2, 'Salud':3, 'Ciencia y Tecnologia': 4, 'Entretenimiento':5, 'Economia': 6}
        self.index_to_classes_dic = { 0: 'Nacional', 1:'Internacional',
                                 2:'Deportes',3: 'Salud',4: 'Ciencia y Tecnologia',5: 'Entretenimiento', 6:'Economia'}

        banned_words = ['', 'a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sobre', "sin", "so", "sobre", "tras", "versus", "vía", "el", "ello", "ellos", "la", "las", "ella", "ellas", "él", "los", "lo", "le", "les", "un", "uno", "una", "unas", "del", "al", "yo", "me", "mí",
                        "conmigo", "tú", "te", "ti", "contigo", "vos", "se", "sí", "consigo", "nosotros", "vosotros", "nos", "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra", "nuestros", "nuestras", "mío", "míos", "tuyo", "tuyos", "tuyas", "tuya", "suyo", "suyos", "suya", "suyas", "donde", "dónde", "que", "qué", "como", "cómo", "cuando", "cuándo", "quien", "quién", "y"]

        banned_chars = '(["\'¿?!¡[\]“:#()+-.,&])|([0-9])'
   
        self.categories = list(self.classes_to_index_dic.keys())
      
        
      
        #Hay filas con categoria nan porque esta mal el excel
        df = df[df['categoria'].isin(self.categories)]
        self.columns = None
        df = self.sample_df_by_category(df,500,self.categories)
        df = df.sample(frac=1).reset_index(drop=True)

        for category in self.categories:
            print(
                f"Category {category}: news count = {len(df[df['categoria']==category])}")

        self.df = self.get_word_df(df, banned_words, banned_chars)
  
        #K de laplace
     
        self.laplace_Ks = dict(
            zip(self.columns, np.full((len(self.columns), 2), [0, 1])))
        self.laplace_Ks["categoria"] = self.categories

    def print_confusion_matrix(self, confusion_matrix):
      
        print("Categoria            | Nacional | Internacional | Deportes | Salud | Ciencia y Tec | Entretenimiento | Economia |")
        for key in confusion_matrix.keys():
            rows = confusion_matrix[key]
            spaces = 21 - len(key)
            s = " "*spaces
            print(f"{key}{s} |    {rows['Nacional']}     |       {rows['Internacional']}       |    {rows['Deportes']}     |  {rows['Salud']}   |       {rows['Ciencia y Tecnologia']}        |        {rows['Entretenimiento']}        |     {rows['Economia']}    |\n")

    def sample_df_by_category(self,df,sample_len,categories):
        final_df = None
        for category in categories:
            category_df = df[df['categoria']== category]
            category_df_len = len(category_df)
            print(f"category df_len: {category_df_len}")
            final_len = sample_len if category_df_len > sample_len else category_df_len
            sample_df = category_df.sample(final_len)
            if final_df is None:
                final_df = sample_df
            else:
                final_df = final_df.append(sample_df, ignore_index=True)
        return final_df

        
    def initialize_confusion_matrix(self,confusion_matrix):
        for row_category in self.categories:
            confusion_matrix[row_category] = dict()
            for col_cateogory in self.categories:
                confusion_matrix[row_category][col_cateogory] = 0

    def calculate_category_rates(self,confusion_matrix,category):
        TP,TN,FP,FN = 0,0,0,0
        TP = confusion_matrix[category][category]
        other_categories = np.delete(self.categories,self.classes_to_index_dic[category])

        for other_categ1 in other_categories:
            FN += confusion_matrix[category][other_categ1]
            FP += confusion_matrix[other_categ1][category]
            for other_categ2 in other_categories:
                    TN += confusion_matrix[other_categ1][other_categ2]
        return TP,TN,FP,FN

    def run(self):
        cross_validation_K = 5
        entries_len = len(self.df.index)
        test_size = int(entries_len / cross_validation_K)
        print(f"entries len: {entries_len}, test_size:{test_size}")
      
        entry_indexes = np.arange(0, entries_len)
        test_indexes = np.arange(0, entries_len)
        np.random.shuffle(test_indexes)
     
        test_sets_indexes = np.array_split(test_indexes, cross_validation_K)
        #print("test indixes: ", test_sets_indexes)
        categories_len = len(self.categories)

 
        min_avg_error = 1000
        best_naive_bayes = None
        best_test_df = None
        confusion_matrix = None
        for i in range(cross_validation_K):
            print(f"Cross validation iteration: {i}")
            test_indexes = test_sets_indexes[i]
            test_df = self.df.iloc[test_indexes]
            train_indexes = np.delete(entry_indexes, test_indexes)
            train_df = self.df.iloc[train_indexes]
            naive_bayes = NaiveBayes(train_df, self.laplace_Ks, 'categoria')
      
            print("Categories total rows: \n",naive_bayes.target_class_total_rows)
            confusion_matrix = dict()
            self.initialize_confusion_matrix(confusion_matrix)

            for index,test_instance in test_df.iterrows():
               # print("test index: ", index)
                test_instance_class = self.index_to_classes_dic[test_instance['categoria']]
                test_instance = test_instance.drop('categoria')
       
                opt_class_index, probability = naive_bayes.get_optimum_class_probability(test_instance.to_dict(),True)
                opt_class = self.index_to_classes_dic[opt_class_index]
               
                #print(f"real class: {test_instance_class}\nopt_class: {opt_class}")
                confusion_matrix[test_instance_class][opt_class] += 1
            #print("Confusion matrix: \n",confusion_matrix)
           
     
            avg_error = 0
            for category in self.categories:
                TP, TN, FP, FN = self.calculate_category_rates(confusion_matrix,category)
                print(f"TP: {TP},TN: {TN}, FP: {FP}, FN: {FN}")
                avg_error += self.calculate_error(TP, TN, FP, FN)
            avg_error /= categories_len
            print("Average error: ", avg_error)
            print("Average accuracy ",1-avg_error)
            if avg_error < min_avg_error:
                min_avg_error = avg_error
                best_naive_bayes = naive_bayes
                best_test_df = test_df
        print("Min avg error: ", min_avg_error)
        self.print_confusion_matrix(confusion_matrix)
        metrics = self.calculate_metrics(confusion_matrix)
        self.print_metrics(metrics)
        self.calculate_ROC(best_test_df,best_naive_bayes)
    def print_metrics(self,metrics):
        categories_len = len(self.categories)
        avg_accuracy= 0
        avg_precision = 0
        avg_f1_score = 0
        avg_tp_rate = 0
        avg_fp_rate = 0
        print("###############################################")
        for category in self.categories:
            accuracy = metrics[category]["accuracy"]
            precision = metrics[category]["precision"]
            f1_score = metrics[category]["f1"]
            tp_rate = metrics[category]["TPrate"]
            fp_rate = metrics[category]["FPrate"]
            avg_accuracy +=accuracy
            avg_precision += precision
            avg_f1_score += f1_score
            avg_tp_rate += tp_rate
            avg_fp_rate += fp_rate
            print("Category: ",category)
            print("Accuracy: ",accuracy)
            print("Precision: ", precision)
            print("F1-Score: ", f1_score)
            print("TP Rate: ", tp_rate)
            print("FP Rate: ", fp_rate)
            print("###############################################")
    
        print("Average Accuracy: ", avg_accuracy/categories_len)
        print("Average Precision: ", avg_precision/categories_len)
        print("Average F1-Score: ", avg_f1_score/categories_len)
        print("Average TP Rate: ", avg_tp_rate/categories_len)
        print("Average FP Rate: ", avg_fp_rate/categories_len)
        print("###############################################")

    def calculate_metrics(self,confusion_matrix):
        
        metrics = dict()

        for category in self.categories:
            metrics[category]= dict()
            TP, TN, FP, FN = self.calculate_category_rates(confusion_matrix, category)
            print(f"TP: {TP},TN: {TN}, FP: {FP}, FN: {FN}")
            
            metrics[category]["precision"] = self.calculate_precision(TP,FP)
            metrics[category]["accuracy"] = self.calculate_accuracy(TP,TN,FP,FN)
            metrics[category]["f1"] = self.calculate_F1_score(TP,FP,FN)
            metrics[category]["TPrate"] = self.calculate_TP_rate(TP,FP)
            metrics[category]["FPrate"] = self.calculate_FP_rate(TN, FP)
        return metrics
    

    def calculate_accuracy(self,TP,TN,FP,FN):
        div = (TP+TN+FP+FN)
        if div == 0:
            return 0

        return (TP+TN)/div
    
    def calculate_precision(self,TP,FP):
        div = (TP+FP)
        if div == 0:
            return 0
        return TP/div
    
    def calculate_recall(self,TP,FN):
        div = (TP+FN)
        if div == 0:
            return 0
        return TP/div

    def calculate_F1_score(self, TP, FP, FN):

        precision = self.calculate_precision(TP,FP)
        recall = self.calculate_recall(TP,FN)
        div = (precision+recall)
        if div == 0:
            return 0
        return (2*precision*recall)/div
    
    def calculate_TP_rate(self, TP, FP):
        div = (TP+FP)
        if div == 0:
            return 0
        return TP/div
    
    def calculate_FP_rate(self, TN, FP):
        div = (FP+TN)
        if div == 0:
            return 0
        return FP/div

    def initialize_categories_confusion_matrix(self,categories_confusion_matrix):
        for category in self.categories:
            categories_confusion_matrix[category] = dict()
            categories_confusion_matrix[category]["TP"] = 0
            categories_confusion_matrix[category]["FP"] = 0
            categories_confusion_matrix[category]["TN"] = 0
            categories_confusion_matrix[category]["FN"] = 0


    def calculate_ROC(self,test_df,naive_bayes):
        thresholds = np.arange(0, 1.1, 0.1)
    
        categories_confusion_matrix = dict()
        category_ROC_points = dict()
        self.initialize_category_ROC(category_ROC_points)
        
        for threshold in thresholds:
            self.initialize_categories_confusion_matrix(categories_confusion_matrix)
            print("Threshold: ",threshold)
        
            for index, test_instance in test_df.iterrows():
                #print("test instance: \n",test_instance)
                real_category = self.index_to_classes_dic[test_instance['categoria']]
                test_instance = test_instance.drop('categoria')
                test_instance_dict = test_instance.to_dict()
                test_instance_prob = naive_bayes.instance_probability(
                    test_instance_dict,True)
                for category in self.categories:
                    prediction_probability = naive_bayes.class_of_given_instance_probability(
                        test_instance_dict, self.classes_to_index_dic[category],True)/test_instance_prob
                  
                    #print( f"real categ: {real_category}, categ: {category}, prediction_prob: {prediction_probability}")
                    if real_category == category:
                        # Positivo = True positive, Negativo = False Negative
                        if prediction_probability > threshold:
                            # True positive
                            categories_confusion_matrix[real_category]["TP"]+=1
                        else:
                            # False negative
                            categories_confusion_matrix[real_category]["FN"] += 1
                    else:
                        if prediction_probability > threshold:
                            # False positive
                            categories_confusion_matrix[category]["FP"] += 1
                        else:
                            # True negative
                            categories_confusion_matrix[category]["TN"] += 1

            for category in self.categories:
                TP = categories_confusion_matrix[category]["TP"]
                TN = categories_confusion_matrix[category]["TN"]
                FP = categories_confusion_matrix[category]["FP"]
                FN = categories_confusion_matrix[category]["FN"]
                print(f"Category: {category} = TP: {TP},TN: {TN}, FP: {FP}, FN: {FN}")
                TP_rate = 0
                if TP + FN != 0:

                    TP_rate = TP/(TP + FN)
                FP_rate = 0
                if FP + TN != 0:
                    FP_rate = FP/(FP + TN)
               # print("TP rate: ",TP_rate)
               # print("FP rate: ",FP_rate)
                category_ROC_points[category]["x"].append(FP_rate)
                category_ROC_points[category]["y"].append(TP_rate)
        
        self.calculate_AUCs(category_ROC_points)
        for category in self.categories:
            #print(f"categ: {category}\nX: {category_ROC_points[category]['x']}\nY: {category_ROC_points[category]['y']}")
            plt.plot(category_ROC_points[category]["x"],category_ROC_points[category]["y"], '-o',label=category)
              
        plt.plot([0,1],[0,1],'--b',label="Clasificador aleatorio")
        plt.title("ROC")
        plt.xlabel("False Positive's Rate")
        plt.ylabel("True Positive's Rate")
        plt.legend(loc='lower right')
        plt.show()
    def initialize_category_ROC(self,category_ROC_points):
        for category in self.categories:
            category_ROC_points[category] = dict()
            category_ROC_points[category]["x"] = list()
            category_ROC_points[category]["y"] = list()

    def calculate_AUCs(self,category_ROC_points):
        avg_auc = 0
        for category in self.categories:
            auc = metrics.auc(
                category_ROC_points[category]["x"], category_ROC_points[category]["y"])
            print(f"{category} AUC: {auc}")
            avg_auc+=auc
        print("Average AUC: ",avg_auc)
    def calculate_error(self, TP, TN, FP, FN):
        return (FP+FN)/(TP+TN+FP+FN)

    def get_word_df(self,df, banned_words, banned_chars):
        df = df.assign(words=df.apply(lambda row: self.tokenize_news(
            row['titular'], banned_words, banned_chars), axis=1))
        news_word_lists = df['words'].tolist()
        categories_list = df['categoria'].tolist()

        all_words = set([item for sublist in news_word_lists for item in sublist])
        all_words_dic = dict(zip(all_words, range(len(all_words))))
        columns_size = len(all_words) + 1
        category_index = columns_size-1
        rows = np.zeros((len(news_word_lists), columns_size), dtype='byte')
        for (i, news_word_list) in enumerate(news_word_lists):
            for word in news_word_list:
                word_index = all_words_dic[word]
                rows[i, word_index] = PRESENT
                rows[i, category_index] = self.classes_to_index_dic[categories_list[i]]

        self.columns = list(all_words)
        self.columns.append('categoria')

        new_df = pd.DataFrame(rows, columns=self.columns)

        return new_df


    def tokenize_news(self,titular, banned_words, banned_chars):
        tokens = titular.lower().split()
        tokens = [re.sub(banned_chars, '', word) for word in tokens]
        filtered = set(filter(lambda token: token not in banned_words, tokens))
        return filtered

