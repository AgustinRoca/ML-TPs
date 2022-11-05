from utils.KNN import KNN


class WKNN(KNN):
    def classify(self, test_instance):
        closests= self.get_closests(test_instance)
      
        classified  = False
        current_K = self.K
      
        while not classified:
            value_counter = dict()
            dist_zero_counter = dict()
            closest_K = closests.head(current_K)
           
            for index in closest_K.index:
                value = self.train_set.loc[index][self.category_label]
                dist = closest_K.loc[index]

                if dist == 0:
                    if value in dist_zero_counter.keys():
                        dist_zero_counter[value] += 1
                    else:
                        dist_zero_counter[value] = 1

                elif not dist_zero_counter:
                    if value in value_counter:
                        value_counter[value] += 1/(dist ** 2)
                    else:
                        value_counter[value] = 1/(dist ** 2)

            if dist_zero_counter:
                return max(dist_zero_counter,key=dist_zero_counter.get)
                
            else:
            
                max_class_list = self.get_max_class_list(value_counter)
                if len(max_class_list) == 1:
                    return max_class_list[0]
                else:
               
                    if current_K < len(closests):
                        current_K +=1
                
                    else:
                        return  max_class_list[0]
