import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
from sklearn.metrics import f1_score


# Read the dataset from csv file
df = pd.read_csv(r"C:\Users\Priyanka\source\repos\ML_hw3\dataset\parkinsons.csv", header = 0)

#global var
attribute_list = list(df.iloc[:0,:-1])
print("attribute list",attribute_list)

class Node:
    def __init__(self, attribute=None, threshold=0, leaf=None):
        self.attribute = attribute
        self.children = []
        self.threshold = threshold
        self.leaf = leaf

def _entropy(column):
    total = len(column)
    unique, c = np.unique(column, return_counts=True)
    entropy = 0
    for i in range(len(c)):
        p = c[i] / total
        if p != 0:
            entropy += (-p) * math.log2(p)
    return entropy

def _info_gain(df,attribute):
    dataset_entropy = _entropy(df.iloc[:,-1])
    mean_attr = np.mean(df[attribute])
    partitions = []
    partitions.append(df.loc[df[attribute] <= mean_attr])
    partitions.append(df.loc[df[attribute] > mean_attr])
    avg_entropy=0
    for each_part in partitions:
        partitions_entropy=(_entropy(each_part.iloc[:,-1]))
        avg_entropy+= partitions_entropy * len(each_part)/len(df)
    info_gain = dataset_entropy - avg_entropy
    return info_gain



def _grow_DT(dataset,attribute_list_local,current_depth):
    #Find dataset entropy for base condition
    max_depth=15
    dataset_entropy = _entropy(dataset.iloc[:,-1])
    if dataset_entropy==0 or len(attribute_list_local)==0 or current_depth==max_depth:
        leaf=dataset.iloc[:,-1].value_counts().idxmax()
        return Node(leaf=leaf)

    m=int(math.sqrt(len(attribute_list_local)))
    random_attr_lst = random.sample(attribute_list_local,m)

    ig = [_info_gain(dataset, attribute) for attribute in random_attr_lst]
    best = random_attr_lst[np.argmax(ig)]
    
    node = Node(attribute=best)
    current_depth+=1
    #print("current depth",current_depth)
    threshold = np.mean(dataset[best])
    node.threshold=threshold
    partitions = []
    partitions.append(dataset.loc[dataset[best] <= threshold])
    partitions.append(dataset.loc[dataset[best] > threshold])

    for i in range(2):
        if partitions[i].empty:
            node.children.append(None)
        else:
            child = _grow_DT(partitions[i],attribute_list_local,current_depth)
            node.children.append(child)
    return node

def predict(data,root):
    if root == None:
        return 
    if root.leaf is not None:
        return root.leaf
    attribute = root.attribute
    threshold = root.threshold
    attribute_list = list(df.iloc[:0,:-1])
    index = attribute_list.index(attribute)
    if data[index] <= threshold:
        return predict(data,root.children[0])
    else:
        return predict(data,root.children[1])
    

def _create_bootstraps(dataset):
    bootstrap = dataset
    num_of_rows = dataset.shape[0]
    num_drop_rows = int(num_of_rows/3)
    seed = np.random.randint(100)
    rows_to_drop = bootstrap.sample(n=num_drop_rows,random_state=seed)
    bootstrap=bootstrap.drop(rows_to_drop.index)
    bootstrap.reset_index(drop=True,inplace=True)
    rows_to_add = bootstrap.sample(n=num_drop_rows,random_state=seed)
    bootstrap = pd.concat([bootstrap,rows_to_add])
    bootstrap.reset_index(drop=True,inplace=True)
    return bootstrap

def build_random_forest(ntree,dataset):
    bootstraps_lst = [_create_bootstraps(dataset) for i in range(ntree)]
    root_nodes_of_ntrees = [_grow_DT(bootstrap,attribute_list,0) for bootstrap in bootstraps_lst]
    return root_nodes_of_ntrees


split_dataset = []
ratios = []
dataset_length = df.shape[0]
unique_target_vals = df.iloc[:,-1].unique()
for vals in unique_target_vals:
    rows = df.loc[df['Diagnosis'] == vals]
    split_dataset.append(rows)
for i in range(len(unique_target_vals)):
    ratios.append(split_dataset[i].shape[0]/dataset_length)
k=10
ntrees=[1,5,10,20,30,40,50]
each_fold_instances = int(dataset_length/k)
sub_folds = [int(ratio*each_fold_instances) for ratio in ratios]    
dataset_copy = df
k_fold_datasets = []
for fold in range(k):
    kth_dataset = pd.DataFrame()
    seed = np.random.randint(100)
    rows_to_add = [split_dataset[i].sample(n=sub_folds[i],random_state=seed) for i in range(len(split_dataset))]
    for row in range(len(rows_to_add)):
        kth_dataset = pd.concat([kth_dataset,rows_to_add[row]])
        split_dataset[row] = split_dataset[row].drop(rows_to_add[row].index)
    k_fold_datasets.append(kth_dataset)

def k_fold_cross_validation(k_fold_datasets,ntree):
    k_acc=[]
    k_f1=[]
    for i in range(len(k_fold_datasets)):
        each_fold_predictions=[]
        kfold_ds_copy = k_fold_datasets
        test_data = pd.DataFrame()
        test_data = pd.concat([test_data,kfold_ds_copy[i]])
        train_data = pd.DataFrame()
        for j in range(len(kfold_ds_copy)):
            if j!=i:
                train_data = pd.concat([train_data,kfold_ds_copy[j]])
        train_data.reset_index(drop=True,inplace=True)
        true_values = list(test_data.iloc[:,-1])
        test_data = test_data.drop(test_data.columns[-1],axis=1)
        test_data.reset_index(drop=True,inplace=True)
        #print("test data",test_data)
        #print("train data",train_data)
        root_nodes_of_ntrees = build_random_forest(ntree,train_data)
        for index, test_row in test_data.iterrows():
            ntrees_prediction = [predict(test_row.tolist(),each_tree) for each_tree in root_nodes_of_ntrees]
            pred=(max(ntrees_prediction,key=ntrees_prediction.count))
            #print("predictions {0} true values {1}".format(pred,true_values[index]))
            each_fold_predictions.append(pred)
        match = np.equal(each_fold_predictions,true_values)
        k_acc.append(np.mean(match))
        f1 = f1_score(true_values, each_fold_predictions, average = 'macro')
        k_f1.append(f1)
        #print("each fold accuracy",k_acc)
    return np.mean(k_acc), np.mean(k_f1)

    
acc_ntree=[]
f1_ntree=[]
for ntree in ntrees:
    acc,f1Score=k_fold_cross_validation(k_fold_datasets,ntree)
    acc_ntree.append(acc)
    f1_ntree.append(f1Score)
    print("accuracy",acc)
    print("F1 score",f1Score)


print(acc_ntree)
print(f1_ntree)

plt.xlabel('number of trees')
plt.ylabel('Accuracy')
plt.title('number of tree vs accuracy plot')
plt.plot(ntrees,acc_ntree,marker='o')
plt.show()

plt.xlabel('number of trees')
plt.ylabel('F1 score')
plt.title('number of tree vs F1 score plot')
plt.plot(ntrees,f1_ntree,marker='o')
plt.show()
