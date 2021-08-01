# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 00:46:39 2021

@author: Jericho
"""
import pandas as pd
import numpy as np
import os

# array = ['good','bad','well','bad','good','good','well','good']

# df = pd.DataFrame(array,columns=['status'])
# status_dict = df['status'].unique().tolist()

# df['transfromed']=df['status'].apply(lambda x : status_dict.index(x))

training_data = pd.read_csv (r'train.csv')
status_dict = sorted(training_data['label'].unique().tolist())

# label2index = {label:index for index, label in enumerate(sorted(set(status_dict)))} 
label2index = {label:index for index, label in enumerate(status_dict)} 


img_labels = np.array([label2index[label] for label in training_data['label']], dtype=int)

idx = 0

image_name = training_data.iloc[idx, 1]


# label2index[image_name]

# save the dictionary 
dirs = 'dictionary.txt'
if not os.path.exists(dirs):
    with open('dictionary.txt', 'w') as f:
        for key in label2index.keys():
            f.write("%s %s\n"%(key, label2index[key]+1))
            
d = {}
with open("dictionary.txt") as f:
    for line in f:
        (key, val) = line.split()
        # print(key)
        d[key] = val
        
d[image_name]
            


        
        


		# Save a .txt file listing the numeric values for each label
# label_file = os.path.join(dataset_root,'numeric_class_labels.txt')
# if not os.path.exists(label_file):
#     with open(label_file, 'w') as f:
#         for id, label in enumerate(sorted(self.label2index)):
#             f.writelines(str(id + 1) + ' ' + label + '\n')



# training_data['transfromed']=training_data['label'].apply(lambda x : status_dict.index(x)+1)

# train = training_data.drop('label', axis = 1)




# for label in training_data['label']:
#     print(label)
#     break


