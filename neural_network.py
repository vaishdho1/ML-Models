import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
def normalise(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))



class Neural_network:

    def __init__(self,layers,neurons:list):
        # neurons are the total number of neurons including the input and output
        self.layers = layers
        self.neurperlayer = neurons
        self.weights =[]
        self.activation=[]


    # To apply sigmoid function to a matrix
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def generate_weights(self):
        for i in range(self.layers+1):
            # Find the number of neurons in layer i and layer i+1
            m = self.neurperlayer[i]
            n = self.neurperlayer[i+1]
            # Create a random weighted array m+1 as adding the bias weight also
            self.weights.append(np.array([[random.gauss(0,1) for _ in range(m+1)] for _ in range(n)]))

    def sum_squares_weight(self):
        tot_sum = 0
        for weight in self.weights:
            # pick all columns except thr first column
            without_col1 = weight[:,1:]
            tot_sum += np.sum(np.square(without_col1))
        return tot_sum

    # inputs of size 1*number of features in input
    def forward_propogation(self,inputs:list,activationList):

        # Convert input into a column vector and add bias term
        activation = np.array(inputs).reshape((-1, 1))
        activation = np.vstack(([1],activation))
        activationList.append(activation)

        for i in range(self.layers):
            z = self.weights[i]@activation

            activation = self.sigmoid(z)
            # Adding bias term
            activation = np.vstack(([1],activation))

            activationList.append(activation)
        # For the last layer compute without adding the bias term
        z = self.weights[-1]@activation

        activation = self.sigmoid(z)
        activationList.append(activation)

        return activation

    # Send both x and y as a tuple to the cost function
    def cost_function(self,inputs,lamb):
        sumJ =0
        # Size of the training instances
        n = len(inputs)
        for x,y in inputs:
            y = np.array(y).reshape(-1,1)
            pred_out = self.forward_propogation(x,[])
            J = -y*np.log(pred_out) - (1-y)*np.log(1-pred_out)
            sumJ += np.sum(J)
        sumJ/=n
        S = self.sum_squares_weight()
        S = (lamb/(2*n))*S
        sumJ += S
        return sumJ

    def back_propogation(self,inputs,outputs,lambd,alpha,iterations):
        D_l=[]
        J =[]
        J1=[]
        for _ in range(iterations):
            n = len(inputs)
            for i in range(len(self.neurperlayer)-1):
                D_l.append(np.zeros((self.neurperlayer[i+1],self.neurperlayer[i]+1)))
            for x,y in inputs:
                delta_l = [0 for _ in range(self.layers + 2)]
                activationlist=[]
                y = np.array(y).reshape(-1,1)
                pred_out = self.forward_propogation(x,activationlist)
                delta_l[-1]=pred_out-y
                for k in range(self.layers,0,-1):
                    delta_l[k] = (np.transpose(self.weights[k])@delta_l[k+1])*activationlist[k]*(1-activationlist[k])
                    delta_l[k] = delta_l[k][1:]

                for k in range(self.layers,-1,-1):
                    D_l[k]+=delta_l[k+1]@activationlist[k].T
            for k in range(self.layers,-1,-1):
                P = lambd*self.weights[k]
                P[:,0] = 0
                D_l[k] = (1/n)*(D_l[k]+P)
            for k in range(self.layers,-1,-1):
                self.weights[k] -= alpha*D_l[k]
            J_new = self.cost_function(inputs, lambd)
            J_test = self.cost_function(outputs,lambd)
            J.append(J_new)
            J1.append(J_test)
        #print(J)
        #print(J1)
        return J,J1
    def predict(self,inputs,conf_matrix,num_classes):
        count =0
        output_classes=[]
        for x,y in inputs:
            y = np.array(y).reshape(-1, 1)
            pred_out = self.forward_propogation(x,[])
            pred_class = np.argmax(np.max(pred_out, axis=1))
            ac_class = np.argmax(np.max(y, axis=1))
            output_classes.append((pred_class,ac_class))
            if pred_class == ac_class:
                count += 1
        cl = [i for i in range(num_classes)]
        for val in cl:
            pos_class = val
            neg_class = cl[:val]+cl[val+1:]
            for pred,ac in output_classes:
                conf_matrix[val][0] += pred==pos_class and ac == pos_class
                conf_matrix[val][2] += pred==pos_class and ac in neg_class
                conf_matrix[val][3] += pred in neg_class and ac in neg_class
                conf_matrix[val][1] += pred in neg_class and ac==pos_class



def create_inputs_to_NN(df,num_classes,col):
    inputs = []
    y = df[col].to_list()
    df = df.drop([col], axis=1)
    feature_list = df.values.tolist()
    for i in range(len(feature_list)):
        # Generate output class labels
        output = [0 for _ in range(num_classes)]
        inp = list(feature_list[i])
        # changing for loan dataset
        output[y[i]] = 1
        inputs.append((inp, list(output)))
    return inputs

# Takes confusion matrix and computes metrics
def calculate_metrics(num_classes,conf_mat_train):
    tot,prec,recall,acc = 0,0,0,0
    for i in range(num_classes):
        tot = conf_mat_train[i][0]+conf_mat_train[i][1]+conf_mat_train[i][2]+conf_mat_train[i][3]
        prec += conf_mat_train[i][0]/(conf_mat_train[i][0]+conf_mat_train[i][2])
        recall += conf_mat_train[i][0]/(conf_mat_train[i][0]+conf_mat_train[i][1])
        acc += (conf_mat_train[i][0]+conf_mat_train[i][3])/tot
    f1_score = 2*prec*recall/(prec+recall)
    return prec/num_classes,recall/num_classes,acc/num_classes,f1_score/num_classes

def plot_metrics(X,metric,label_list):
    num_plots = len(metric)
    rows = 2
    cols = (num_plots + 1) // 2
    # Create the subplots
    fig, axs = plt.subplots(rows, cols)
    # Iterate over the data and plot each set of values
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            index = i * cols + j
            if index < num_plots:
                y= metric[index]
                ax.plot(X, y,marker='o')
                ax.set_title(f"Metric {label_list[index]}")
                ax.set_xlabel(f"Number of iterations")
                ax.set_ylabel(f"J")
            else:
                # Remove any unused subplots
                fig.delaxes(ax)

    # Adjust spacing and display the plot
    fig.tight_layout()
    plt.show()
    '''
    plt.plot(iterations,prec_train,marker='o',label='prec_train')
    plt.plot(iterations, prec_test, marker='o', label='prec_test')
    plt.plot(iterations, acc_train, marker='o', label='acc_train')
    plt.plot(iterations, acc_test, marker='o', label='acc_test')
    plt.plot(iterations, recall_train, marker='o', label='recall_train')
    plt.plot(iterations, recall_test, marker='o', label='recall_test')
    plt.show()
    '''
def plot_error_function(input_csv,col):
    J_tot = []
    X =[ i for i in range(500)]
    df = pd.read_csv(input_csv, delimiter='\t', header=0)
    num_classes = df[col].nunique()
    class_col = df[col]
    # Doing this for numerical dataset
    df = df.loc[:, df.columns != col].astype(float).apply(normalise)
    df = pd.concat([df, class_col], axis=1)
    #final_kfold, num_classes = generate_data(input_csv, col)
    obj = Neural_network(1, [13, 4, 3])
    obj.generate_weights()
    train_inputs = create_inputs_to_NN(df,num_classes,col)
    #test_inputs = create_inputs_to_NN(test_samples, num_classes, col)
    J = obj.back_propogation(train_inputs, 0, 1, 500)
    print(J)
    #print(J1)
    plt.plot(X,J,label='Cost_function')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()
'''
def train_test(input_csv,col):
    df = pd.read_csv(input_csv, delimiter='\t', header=0)
    print(df.columns)
    class_col = df[col]
    # Do this for categorical data, fit and transform the data
    #encoder = OneHotEncoder(categories='auto', sparse=False)
    #data_encoded = encoder.fit_transform(df.loc[:,df.columns != col])
    #df = pd.DataFrame(data_encoded)
    #df = pd.concat([df, class_col], axis=1)
    num_classes = df[col].nunique()
    # Doing this for numerical dataset
    df = df.loc[:, df.columns != col].astype(float).apply(normalise)
    df = pd.concat([df, class_col], axis=1)
    df = df.sample(frac=1, random_state=42)
    X_train, X_test = train_test_split(df,test_size=0.2)
    #print(X_train)
    #print(X_test)
    obj = Neural_network(3, [9,32,64,64,2])
    obj.generate_weights()
    conf_mat_train = {i: [0 for _ in range(4)] for i in range(num_classes)}
    conf_mat_test = {i: [0 for _ in range(4)] for i in range(num_classes)}
    train_inputs = create_inputs_to_NN(X_train, num_classes, col)
    test_inputs = create_inputs_to_NN(X_test, num_classes, col)
    J,J1 = obj.back_propogation(train_inputs,test_inputs,0.25, 1, 500)
    print(J)
    print(J1)
    #plt.plot([i for i in range(500)], J, label='Cost_function')
    plt.plot([i for i in range(500)], J1,label='Cost_function')
    plt.title('Error function for test data')
    plt.xlabel(f'Number of iterations *{len(X_train)}')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()
    #J1 = obj.back_propogation(train_inputs, test_inputs, 0.25, 1, 500)
    #obj.predict(train_inputs, conf_mat_train, num_classes)
    #obj.predict(test_inputs, conf_mat_test, num_classes)
    # print(f'J1: {J1[-1]}')
    #print(f'cost: {obj.cost_function(test_inputs, 0.25)}')
'''

def find_cost(input_csv,col):
    #X_train, X_test = train_test_split( test_size=0.2)
    final_kfold, num_classes = generate_data(input_csv, col)
    #print(final_kfold[9])
    test_samples = final_kfold[9]
    for i in range(9):
        obj = Neural_network(3, [9,32,64,2,2])
        obj.generate_weights()
        conf_mat_train = {i: [0 for _ in range(4)] for i in range(num_classes)}
        conf_mat_test = {i: [0 for _ in range(4)] for i in range(num_classes)}
        conf_test = final_kfold[0:i+1]
        train_samples = pd.concat(conf_test).reset_index(drop=True)
        #print('shape',train_samples.shape)
        train_inputs = create_inputs_to_NN(train_samples, num_classes, col)
        test_inputs = create_inputs_to_NN(test_samples, num_classes, col)
        J1 = obj.back_propogation(train_inputs,test_inputs,0.25, 1, 500)
        obj.predict(train_inputs,conf_mat_train,num_classes)
        obj.predict(test_inputs,conf_mat_test,num_classes)
        #print(f'J1: {J1[-1]}')
        print(f'cost: {obj.cost_function(test_inputs,0.25)}')





# plots the metrics with varying alpha and iterations
def metric_plot(input_csv,col):
    final_kfold, num_classes = generate_data(input_csv,col)

    # Stores the TP,FN,FP,TN for the different classes
    # Not plotting error function,need to check this
    alphas = [0.5]
    iterations = [1000]
    lambd = [0]
    '''
    Network_architectures = {5:[13,4,6,2,8,6,3],
                            9:[13,8,8,6,6,6,4,4,2,2,3]}
    '''

    Network_architectures =[(2, [9,2,4,2])]


    for layer, arch in Network_architectures:
        print(f'----------Architecture layers :{layer}-------')
        print(f'----------configs {arch}------------------')
        for al in alphas:
            print(f'----------alpha {al}----------')
            for lam in lambd:
                print(f'----------lambda:{lam}-------------')
                conf_mat_train = {i: [0 for _ in range(4)] for i in range(num_classes)}
                conf_mat_test = {i: [0 for _ in range(4)] for i in range(num_classes)}
                for i in range(10):
                    obj = Neural_network(layer, arch)
                    obj.generate_weights()
                    current_dfs = final_kfold[0:i] + final_kfold[i + 1:]
                    train_samples = pd.concat(current_dfs).reset_index(drop=True)
                    test_samples = final_kfold[i]
                    train_inputs = create_inputs_to_NN(train_samples, num_classes,col)
                    test_inputs = create_inputs_to_NN(test_samples, num_classes,col)
                    # To find the back prop,lambda = 0.25 is given
                    obj.back_propogation(train_inputs,test_inputs,lam, al, 500)
                    obj.predict(train_inputs, conf_mat_train, num_classes)
                    obj.predict(test_inputs, conf_mat_test, num_classes)
                    #print(obj.cost_function(train_inputs,lam))
                prec_tr, rec_tr, acc_tr, f1_tr = calculate_metrics(num_classes, conf_mat_train)
                prec_ts,rec_ts,acc_ts,f1_ts = calculate_metrics(num_classes,conf_mat_test)
                print('-------Metrics-----')
                print('Training')
                print('Precision',prec_tr)
                print('Recall',rec_tr)
                print('Accuracy',acc_tr)
                print('F1',f1_tr)
                print('Test')
                print('Precision',prec_ts)
                print('Recall',rec_ts)
                print('Accuracy',acc_ts)
                print('F1',f1_ts)




# Using all samples form training
def initial_processing(input_csv):
    df = pd.read_csv(input_csv, delimiter='\t', header=0)
    # print(df.dtypes)
    num_classes = df['Class'].nunique()
    class_col = df['Class']
    print(class_col.unique())
    df = df.loc[:, df.columns != 'Class'].astype(float).apply(normalise)
    df = pd.concat([df, class_col], axis=1)
    inputs = []
    y = df['Class'].to_list()
    df = df.drop(['Class'], axis=1)
    feature_list = df.values.tolist()
    # print('feature_list',feature_list)
    for i in range(len(feature_list)):
        # Generate output class labels
        output = [0 for _ in range(num_classes)]
        inp = list(feature_list[i])
        # changing for wine dataset
        output[y[i]-1] = 1
        inputs.append((inp, list(output)))



#input_csv = 'datasets/hw3_cancer.csv'
#find_cost(input_csv,'class')
#train_test(input_csv,'Class')
#plot_error_function(input_csv,'class')
#metric_plot(input_csv,'class')
#initial_processing('datasets/hw3_wine.csv')



























