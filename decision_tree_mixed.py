import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import statistics
import pandas as pd
import numpy as np
import random
class Node:

    def __init__(self,col=None,rows=None):
        self.rows = rows
        self.split_col = col
        self.children = dict()
        self.label = None
        self.leaf = False



# Todo
'''
# Add numerical split
# Run on wine dataset
# Adjust parameters
# Check accuracies while training a random forest and how it is affected
'''
class decisionTree:
    def __init__(self,cols,classes,column_type,class_label,rows,df):
        self.root =None
        self.df = df
        self.rows = rows
        self.columns = cols
        self.classes = classes
        self.column_type = column_type
        self.depth = 0
        # hyperparameters
        self.th_split = 3
        self.info_th = .000001
        self.max_depth = 20
        self.M = int(math.sqrt(len(self.columns)))
        self.class_label = class_label
        #print(self.th_split,self.max_depth)


    # Calculate probability for a given set of rows
    def find_entropy(self,row_list):
        prob = 0
        subset = self.df.loc[row_list]
        tot_size = subset.shape[0]
        # For each class find the probability of the class
        for cl in self.classes:
            ins = subset.loc[subset[self.class_label] == cl].shape[0]
            p = ins/tot_size
            if p==0:
                p = 1
            prob += -p*math.log2(p)
        return prob
    '''
    Finds the average entropy and returns the entropy and dictionary with
    values and keys:rows for splits
    '''

    # Todo :Currently only deals with categorical values
    def average_entropy(self,rows,column,is_numerical):
        sub_df = self.df.loc[rows]
        if is_numerical:
            avg = sub_df[column].mean()
            less_than = sub_df[sub_df[column] < avg]
            greater_than = sub_df[sub_df[column] >= avg]
            less_than_entropy = 0
            if less_than.shape[0] > 0:
                for _, group in less_than.groupby(self.class_label):
                    c = group[self.class_label].count()
                    p = c / len(less_than)
                    less_than_entropy += -p * math.log2(p)
            greater_than_entropy = 0
            if greater_than.shape[0] > 0:
                for _, group in greater_than.groupby(self.class_label):
                    c = group[self.class_label].count()
                    p = c / len(greater_than)
                    greater_than_entropy += -p * math.log2(p)
            weighted_avg_entropy = len(less_than) * less_than_entropy / len(rows) + len(greater_than) * greater_than_entropy / len(rows)
        else:
            # Compute the entropy of the child nodes
            child_entropies = sub_df.groupby(column)[self.class_label].apply(lambda x: -(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True))).sum())
            # Compute the proportion of data in each child node
            proportions = sub_df[column].value_counts(normalize=True)
            # Calculate the weighted average entropy of the child nodes
            weighted_avg_entropy = (child_entropies * proportions).sum()
        return weighted_avg_entropy

    def informationGain(self,rows,old_entropy):
        gain = -float('inf')
        split_col = 0
        new_en = 0
        subset_df = self.df.loc[rows]
        # Pick m columns in random
        cols_picked = random.sample(self.columns,k=self.M)
        for col in cols_picked:
            # To check is the current split_col is numerical or not
            is_numerical = self.column_type[col]
            new_entropy = self.average_entropy(rows,col,is_numerical)
            new_gain = old_entropy - new_entropy
            if new_gain > gain:
                split_col = col
                gain = new_gain
                new_en = new_entropy
        if self.column_type[split_col]:
            avg = subset_df[split_col].mean()
            unique_values = {avg: subset_df[subset_df[split_col] < avg].index,
                             avg + 1: subset_df[subset_df[split_col] >= avg].index}
        else:
            unique_values = self.df.loc[rows].groupby(split_col).groups
        return (split_col,new_en,unique_values)


    # Check if all instances are same
    def all_instances_same(self,rows):
        subset = self.df.loc[rows]
        if subset[self.class_label].nunique() == 1:
            return True
        return False

    # Return the majority class
    def calculate_class(self,rows):
        subset = self.df.loc[rows]
        return subset[self.class_label].value_counts().idxmax()

    def count_majority(self,rows):
        #print('rows1',rows)
        subset = self.df.loc[rows]
        cl = subset[self.class_label].value_counts().idxmax()
        return cl

    def traverse_tree(self,node):

        if node.leaf:
            print(node.label)
            return
        print(node.split_col)
        for child in node.children:
            print('child',child)
            self.traverse_tree(node.children[child])
        print('done')

    # Stopping criteria
    def minimum_size_for_split(self,rows):
        if len(rows) < self.th_split:
            return True,self.count_majority(rows)
        return False,0

    def minimal_gain_criteria(self,new_en,old_en):
        if (old_en - new_en)/old_en < self.info_th:
            return True
        return False

    def maximal_depth(self,d):
        return d >= self.max_depth



    def get_split(self,old_en,rows,prev_rows,stop_cri,depth):
        self.depth = max(self.depth,depth)
        newNode = Node(rows=rows)
       # print(self.df.loc[rows])
        # Exit conditions
        # If the current branch rows are empty
        if len(rows) == 0:
            newNode.leaf = True
            newNode.label = self.calculate_class(prev_rows)
            return newNode
        # If all the instances are the same
        if self.all_instances_same(rows):
            newNode.leaf = True
            newNode.label = self.calculate_class(rows)
            return newNode
        # Number of instances are small
        if 'minimum_size_for_split' in stop_cri:
            dec,cl = self.minimum_size_for_split(rows)
            if dec:
                newNode.leaf = True
                newNode.label = cl
                return newNode
        # If depth exceeds
        if 'maximal_depth' in stop_cri:
            if self.maximal_depth(depth):
                newNode.leaf = True
                cl = self.count_majority(rows)
                newNode.label = cl
                return newNode


        #Find the next split column and corresponding rows for each split value

        split_col, new_en, split_values = self.informationGain(rows,old_en)
        if 'minimal_gain_criteria' in stop_cri:
            if self.minimal_gain_criteria(new_en,old_en):
                newNode.leaf = True
                cl = self.count_majority(rows)
                newNode.label = cl
                return newNode

        newNode.split_col = split_col
        for key in split_values:
            newNode.children[key] = self.get_split(new_en,split_values[key],rows,stop_cri,depth+1)
        return newNode

    def predict(self,test_data):
        temp = self.root
        while temp.leaf !=True:
            col = temp.split_col
            if self.column_type[col]:
                value = test_data[col]
                # If the value is not present in the current class
                cur_key = list(temp.children.keys())[0]
                if value < cur_key:
                    temp = temp.children[cur_key]
                else:
                    temp = temp.children[cur_key + 1]
            else:
                value = test_data[col]
                # If the value is not present in the current class
                if value not in temp.children:
                    return self.calculate_class(temp.rows)
                temp = temp.children[value]
        return temp.label
    def find_label(self,test_data):
        pred_list = []
        for i, row in test_data.iterrows():
            row_list = row.values.tolist()
            header_list = test_data.columns.tolist()
            result = dict(zip(header_list, row_list))
            pred_list.append(self.predict(result))
        return pred_list

    def accuracy(self,test_data):
        cnt = 0
        for i, row in test_data.iterrows():
            row_list = row.values.tolist()
            header_list = test_data.columns.tolist()
            result = dict(zip(header_list, row_list))
            pred = self.predict(result)
            if pred == result[self.class_label]:
                cnt+=1
        return (cnt)/test_data.shape[0]

    def trainDecTree(self,stop_criteria):
        old_en = self.find_entropy(self.rows)
        self.root = self.get_split(old_en,self.rows,[],stop_criteria,0)
        return self.root









'''
def plot_histograms(l,label,color):
    plt.hist(l, bins='auto',label=label,color = color)
    plt.title(f'Accuracy histogram for {label}')
    plt.legend()
    plt.xlabel('(Accuracy)')
    plt.ylabel(f'(Accuracy frequency on {label})')
    plt.show()
'''
