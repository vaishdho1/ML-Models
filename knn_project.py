import copy
import csv
import math
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

def generate_data(df,col,num_list,cat_list):
    print(df.head())
    print(df.columns)
    class_col = df[col]
    num_classes = df[col].nunique()
    # Doing this for categorical data
    encoder = OneHotEncoder(categories='auto', sparse=False)
    data_encoded = encoder.fit_transform(df.loc[:, df.columns != col].loc[:, cat_list])
    df1 = pd.DataFrame(data_encoded)
    # Doing this for numerical dataset
    df = df.loc[:, df.columns != col].loc[:, num_list].astype(float).apply(normalise)
    df = pd.concat([pd.concat([df, df1], axis=1), class_col], axis=1)
    print(df.head())
    class_groups = df.groupby(col)
    tot_classes = []
    data_l = dict()
    final_kfold = []
    for cl, group in class_groups:
        tot_classes.append(cl)
        group_df = pd.DataFrame(group)
        group_df = group_df.sample(frac=1)
        len_cl = group_df.shape[0]
        len_fold = len_cl // 10
        left = len_cl % 10
        data_l[cl] = [group_df.iloc[i:i + len_fold] for i in range(0, len_cl, len_fold)][:10]
        for i in range(1, left + 1):
            data_l[cl][i - 1] = pd.concat([data_l[cl][i - 1], group_df.iloc[len_cl - i:len_cl - i + 1, :]],
                                          ignore_index=True)
    for i in range(10):
        pd_old = pd.DataFrame()
        for cl, _ in class_groups:
            pd_old = pd.concat([pd_old, data_l[cl][i]], ignore_index=True)
        final_kfold.append(pd_old)
    return final_kfold, num_classes


def normalise(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def load_dataset(input_csv,col):
    df = pd.read_csv(input_csv)

    #df[col] = df[col].replace({'Y': 1, 'N': 0})
    class_col = df[col]
    confmatrix = {key: [0] * 4 for key in class_col.unique()}
    df.drop(columns=[col], inplace=True)
    print(df.nunique())
    print(df.info())
    print(df.head())
    cat_list = []
    num_list = []
    # Seprate numerical categorical columns
    for col in df.columns:
        if df[col].nunique() > 8:
            num_list.append(col)
        else:
            cat_list.append(col)
    print('num_list', num_list)
    print('cat_list', cat_list)
    df = pd.concat([df, class_col], axis=1)
    return df,cat_list,num_list,confmatrix


def loadData(dataset):
    data = []
    for row in dataset:
        data.append(row.tolist())
    #with open ('iris_dataset.csv','r') as f:
    #    reader = csv.reader(f)
    #    for row in reader:
    #        row = [float(x) for x in row[:-1]] + [row[-1]]
    #        data.append(row)
    #print(digits_dataset_X)
    #data = shuffle(data)
    print(data)
    return data
#Find euclidian distance
def euclidian_distance(train_row,test_row):
        dis = 0
        for tr,tst in zip(train_row,test_row):
            dis += (tr-tst)**2
        return math.sqrt(dis)

#Find minimum and maximum in training data for normalization
def find_min_max(train,istrain):
    k = 1 if istrain==1 else 0
    min_val = [float('inf') for _ in range(len(train[0])-k)]
    max_val = [0 for _ in range(len(train[0])-k)]
    for row in train:
        for j in range(len(row)-k):
            min_val[j] = min(min_val[j], row[j])
            max_val[j] = max(max_val[j], row[j])
    return min_val,max_val

#Normalizing parameters
def normalise_parameters(train,test,istrain):
    k=1 if istrain==1 else 0
    min_val,max_val = find_min_max(train,istrain)
    for row in test:
        for i in range(len(row)-k):
            row[i] = (row[i]-min_val[i])/(max_val[i]-min_val[i])

#Find distances and sort them in ascending order
def find_distance(train,test):
    dist =[]
    for row in train:
        dist.append([row,euclidian_distance(row[:-1],test[:-1])])
    dist.sort(key=lambda x: x[1])
    return dist
#Find prediction for data
def find_predictions(dist,k):
    freqcount = dict()
    for val in dist[:k]:
        key = val[0][-1]
        freqcount[key] =1 if key not in freqcount else freqcount[key]+1
    return max(freqcount,key = freqcount.get)

#Run knn algorithm
def knn(train,test,k):
    pred_out = []
    actual_out = []
    for data in test:
        dist = find_distance(train,data)
        res = find_predictions(dist,k)
        pred_out.append(res)
        actual_out.append(data[-1])

    pred_df = pd.DataFrame({'actual_class': actual_out, 'pred_class': pred_out})
    return pred_df

def update_conf_matrix(confmatrix,pred_df):
    classes = [key for key in confmatrix]
    for i in range(len(classes)):
        pos_class = classes[i]
        neg_class = classes[:i] + classes[i + 1:]
        true_pos = len(pred_df[(pred_df['pred_class'] == pos_class) & (pred_df['actual_class'] == pos_class)])
        true_neg = len(pred_df[(pred_df['pred_class'].isin(neg_class)) & (pred_df['actual_class'].isin(neg_class))])
        false_pos = len(pred_df[(pred_df['pred_class'] == pos_class) & (pred_df['actual_class'].isin(neg_class))])
        false_neg = len(pred_df[(pred_df['pred_class'].isin(neg_class)) & (pred_df['actual_class'] == pos_class)])
        tot = true_pos + true_neg + false_pos + false_neg
        confmatrix[pos_class][0] += true_pos
        confmatrix[pos_class][1] += false_neg
        confmatrix[pos_class][2] += false_pos
        confmatrix[pos_class][3] += true_neg

def initial_setup(data):
    #data = loadData()
    x_train, x_test = train_test_split(data, test_size=.2)
    train = copy.deepcopy(x_train)
    #normalise_parameters(train,x_train,1)
    #normalise_parameters(train, x_test, 1)

    return x_train,x_test
def plot_functions(output,color,label):
    x_dir = [i for i in range(1, 52, 2)]
    plt.errorbar(x_dir, output, marker='o', color=color,label=label)
    plt.xlabel('K')
    plt.ylabel('Accuracy for Parkinsons dataset')
    plt.legend()
    plt.show()
def calculate_metrics(conf_matrix):
    tot_len = len(conf_matrix)
    prec, recall, acc, f1score = 0, 0, 0, 0
    for key in conf_matrix:
        tot = sum(conf_matrix[key])
        cur_recall = conf_matrix[key][0]/(conf_matrix[key][0]+conf_matrix[key][1])
        cur_prec = conf_matrix[key][0]/(conf_matrix[key][0] + conf_matrix[key][2])
        prec += cur_prec
        recall += cur_recall
        acc += (conf_matrix[key][0]+conf_matrix[key][3])/tot
        f1score += (2*cur_recall*cur_prec)/(cur_recall+cur_prec)
    prec /=tot_len
    recall/=tot_len
    acc/=tot_len
    f1score/=tot_len
    return prec,recall,acc,f1score

def run_knn(final_kfold,confmatrix):
    prec_list=[]
    recall_list=[]
    accuracy_list=[]
    f1_list =[]
    output_train = []
    output_test = []
    st_dev_train = []
    st_dev_test = []
    for i in range(1,52,2):
        res1 = []
        res2 = []
        for j in range(10):
            x_train = pd.concat(final_kfold[0:j] + final_kfold[j + 1:]).values
            x_test = final_kfold[j].values
            knn(x_train, x_train, i)
            pred_df = knn(x_train, x_test, i)
            update_conf_matrix(confmatrix,pred_df)
        #st_dev_train.append(statistics.stdev(res1))
        #st_dev_test.append(statistics.stdev(res2))
        prec,recall,accuracy,f1 = calculate_metrics(confmatrix)
        prec_list.append(prec)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
        f1_list.append(f1)
        print(i)
        print('prec',prec)
        print('recall',recall)
        print('accuracy',accuracy)
        print('f1',f1)
    plot_functions(accuracy_list,'forestgreen','testing_set')
    #plot_functions(output_test,st_dev_test,'teal','test_set')


if __name__ == '__main__':
    df,cat_list,num_list,confmatrix = load_dataset('parkinsons.csv','Diagnosis')
    final_kfold,num_classes = generate_data(df,'Diagnosis',num_list,cat_list)
    #loadData(dataset)
    run_knn(final_kfold,confmatrix)
