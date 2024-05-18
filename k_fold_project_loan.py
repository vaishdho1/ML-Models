import pandas as pd
#Change this to decision_tree_ginni to run ginni code
from decision_tree_mixed import *
import random
from sklearn import datasets
import matplotlib.pyplot as plt


# Todo: Join both numerical and categorical in one code
class RandomForest:

    def __init__(self, ntree,tot_classes, column_type, stop_criteria):
        self.ntree = ntree
        self.stop_criteria = stop_criteria
        # Choose positive and negative classes
        self.pos_class = tot_classes[0]
        self.neg_class = tot_classes[1:]
        self.classes = tot_classes
        self.column_type = column_type
        self.confmatrix = {key: [0]*4 for key in self.classes}

    def generate_bootstraps(self, df):
        train_samp = []
        N = df.shape[0]
        for i in range(self.ntree):
            train_indices = np.random.choice(df.index, size=N, replace=True)
            train_data = df.loc[train_indices].reset_index(drop=True)
            train_samp.append(train_data)
        return train_samp
    def create_confmatrix(self,pred_df):

        for i in range(len(self.classes)):
            pos_class = self.classes[i]
            neg_class = self.classes[:i]+self.classes[i+1:]
            true_pos = len(pred_df[(pred_df['pred_class'] == pos_class) & (pred_df['actual_class'] == pos_class)])
            true_neg = len(pred_df[(pred_df['pred_class'].isin(neg_class)) & (pred_df['actual_class'].isin(neg_class))])
            false_pos = len(pred_df[(pred_df['pred_class'] == pos_class) & (pred_df['actual_class'].isin(neg_class))])
            false_neg = len(pred_df[(pred_df['pred_class'].isin(neg_class)) & (pred_df['actual_class'] == pos_class)])
            tot = true_pos+true_neg+false_pos+false_neg
            self.confmatrix[pos_class][0]+=true_pos
            self.confmatrix[pos_class][1]+=false_neg
            self.confmatrix[pos_class][2] +=false_pos
            self.confmatrix[pos_class][3] +=true_neg



    def majority_vote(self, pred_labels, test_sample,col):
        predicted_array = np.array(pred_labels)
        output = test_sample[col].to_list()
        maj_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predicted_array)
        pred_df = pd.DataFrame({'actual_class': test_sample[col], 'pred_class': maj_predictions})
        self.create_confmatrix(pred_df)


    def initial_setup(self, df, test_samp,col):
        train_samp = self.generate_bootstraps(df)
        #print('train_samp')
        #print(train_samp)
        columns = list(df.columns)
        #print(columns)
        columns.remove(col)
        classes = df[col].unique().tolist()
        train_acc = 0
        pred_labels = []
        for i in range(self.ntree):
            rows = list(train_samp[i].index)
            tree = decisionTree(columns, classes,self.column_type,col,rows, train_samp[i])
            tree.trainDecTree(self.stop_criteria)
            train_acc += tree.accuracy(train_samp[i])
            pred_labels.append(tree.find_label(test_samp))
        self.majority_vote(pred_labels, test_samp,col)
        return train_acc/self.ntree


class KCrossValidation:

    def __init__(self, k):
        self.k = k

    def find_columntype(self, df):
        '''
        Holds the value of whether the column is categorical or numerical
        1: numerical
        0:categorical
        '''
        column_type = dict()
        for col in list(df.columns):
            if df[col].nunique() > 5:
                column_type[col] = 1
            else:
                column_type[col] = 0
        print(column_type)
        return column_type

    def generate_data(self, input_csv,col):
        #digits = datasets.load_digits()
        #digits_dataset_X = digits.data
        #digits_dataset_y = digits.target.reshape(-1,1)
        #N = len(digits_dataset_X)
        # print(digits_dataset_X)
        # Convert the dataset into a pandas dataframe
        # print(digits_dataset_X)
        #df = pd.DataFrame(digits_dataset_X)
        #df['class'] = digits_dataset_y
        df = pd.read_csv(input_csv,delimiter=',',header=0)
        #print(df.head())
        #df.columns = map(str.lower, df.columns)
        #Loan dataset only
        df[col] = df[col].replace({'Y':1,'N':0})

        #print(df.head())
        df.drop(columns=['Loan_ID'], inplace=True)
        class_groups = df.groupby(col)
        # Finds whether the column is categorical or numerical
        column_type = self.find_columntype(df)
        tot_classes = []
        data_l = dict()
        final_kfold = []
        for cl, group in class_groups:
            tot_classes.append(cl)
            group_df = pd.DataFrame(group)
            group_df = group_df.sample(frac=1)
            len_cl = group_df.shape[0]
            len_fold = len_cl // self.k
            left = len_cl % self.k
            data_l[cl] = [group_df.iloc[i:i + len_fold] for i in range(0, len_cl, len_fold)][:self.k]
            for i in range(1, left + 1):
                data_l[cl][i - 1] = pd.concat([data_l[cl][i - 1], group_df.iloc[len_cl - i:len_cl - i + 1, :]],ignore_index=True)
        for i in range(self.k):
            pd_old = pd.DataFrame()
            for cl, _ in class_groups:
                pd_old = pd.concat([pd_old, data_l[cl][i]], ignore_index=True)
            final_kfold.append(pd_old)

        return column_type, tot_classes, final_kfold

def calculate_metrics(conf_matrix,tot_len):
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


k = 10
obj = KCrossValidation(k)
column_type, tot_classes, final_kfold = obj.generate_data(r'dataset\loan.csv','Loan_Status')
stopping = ['maximal_depth']
trees = [1, 5, 10, 20, 30, 40, 50]
tot_acc = []
tot_prec = []
tot_recall = []
tot_f1 = []

for tr in trees:
    rf = RandomForest(tr,tot_classes, column_type, stopping)
    train_acc = 0
    for i in range(k):
        current_dfs = final_kfold[0:i] + final_kfold[i + 1:]
        train_samples = pd.concat(current_dfs).reset_index(drop=True)
        test_samples = final_kfold[i]
        train_acc += rf.initial_setup(train_samples, test_samples,'Loan_Status')
        #print(rf.confmatrix)
    prec,recall,acc,f1score = calculate_metrics(rf.confmatrix,len(tot_classes))
    tot_acc.append(acc)
    tot_prec.append(prec)
    tot_recall.append(recall)
    tot_f1.append(f1score)
    print(f"accuracy:{acc},{train_acc / k}")
    print(f"precision:{prec}")
    print(f"recall:{recall}")
    print(f"f1score:{f1score}")


def plot_values(x_co, y_co, label):
    plt.plot(x_co, y_co, marker='o', color='brown', label='test_data')
    plt.xlabel('trees')
    plt.ylabel(label)
    plt.legend()
    plt.title('Digits(Info gain)')
    plt.show()


plot_values(trees, tot_acc, 'accuracy')
plot_values(trees, tot_prec, 'precision')
plot_values(trees, tot_recall, 'recall')
plot_values(trees, tot_f1, 'f1score')


