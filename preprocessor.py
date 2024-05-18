import pandas as pd
from neural_network import *


def generate_data(df, col, num_list, cat_list):
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


def process_data(input_csv, col):
    df = pd.read_csv(input_csv)
    class_col = df[col]
    print(df.nunique())
    # df = pd.read_csv(input_csv)
    df.drop(columns=[col], inplace=True)
    print('here')
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
    return df, cat_list, num_list


def metric_plot(df, col, num_list, cat_list):
    final_kfold, num_classes = generate_data(df, col, num_list, cat_list)

    # Stores the TP,FN,FP,TN for the different classes
    # Not plotting error function,need to check this
    alphas = [1]
    lambd = [0, 0.25]
    '''
    Network_architectures = {5:[13,4,6,2,8,6,3],
                            9:[13,8,8,6,6,6,4,4,2,2,3]}
    '''

    Network_architectures = [(2, [22, 256, 256, 2]), (3, [22, 256, 256, 128, 2])]

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
                    train_samples = pd.concat(current_dfs)
                    test_samples = final_kfold[i]
                    train_inputs = create_inputs_to_NN(train_samples, num_classes, col)
                    test_inputs = create_inputs_to_NN(test_samples, num_classes, col)
                    # To find the back prop,lambda = 0.25 is given
                    J = obj.back_propogation(train_inputs, test_inputs, lam, al, 500)
                    obj.predict(train_inputs, conf_mat_train, num_classes)
                    obj.predict(test_inputs, conf_mat_test, num_classes)
                prec_tr, rec_tr, acc_tr, f1_tr = calculate_metrics(num_classes, conf_mat_train)
                prec_ts, rec_ts, acc_ts, f1_ts = calculate_metrics(num_classes, conf_mat_test)
                print('-------Metrics-----')
                print('Training')
                print('Precision', prec_tr)
                print('Recall', rec_tr)
                print('Accuracy', acc_tr)
                print('F1', f1_tr)
                print('Test')
                print('Precision', prec_ts)
                print('Recall', rec_ts)
                print('Accuracy', acc_ts)
                print('F1', f1_ts)


def train_test(df, col, cat_list, num_list):
    class_col = df[col]
    num_classes = df[col].nunique()
    # Doing this for categorical data
    encoder = OneHotEncoder(categories='auto', sparse=False)
    data_encoded = encoder.fit_transform(df.loc[:, df.columns != col].loc[:, cat_list])
    df1 = pd.DataFrame(data_encoded)
    # Doing this for numerical dataset
    df = df.loc[:, df.columns != col].loc[:, num_list].astype(float).apply(normalise)
    df = pd.concat([pd.concat([df, df1], axis=1), class_col], axis=1)
    df = df.sample(frac=1, random_state=42)

    X_train, X_test = train_test_split(df, test_size=0.2)
    obj = Neural_network(2, [22, 64, 128, 2])
    obj.generate_weights()
    conf_mat_train = {i: [0 for _ in range(4)] for i in range(num_classes)}
    conf_mat_test = {i: [0 for _ in range(4)] for i in range(num_classes)}
    train_inputs = create_inputs_to_NN(X_train, num_classes, col)
    # print(train_inputs[:5])
    test_inputs = create_inputs_to_NN(X_test, num_classes, col)
    J, J1 = obj.back_propogation(train_inputs, test_inputs, 0, 1, 500)
    obj.predict(train_inputs, conf_mat_train, num_classes)
    obj.predict(test_inputs, conf_mat_test, num_classes)
    # print(obj.cost_function(train_inputs,lam))
    prec_tr, rec_tr, acc_tr, f1_tr = calculate_metrics(num_classes, conf_mat_train)
    prec_ts, rec_ts, acc_ts, f1_ts = calculate_metrics(num_classes, conf_mat_test)
    print('-------Metrics-----')
    print('Training')
    print('Precision', prec_tr)
    print('Recall', rec_tr)
    print('Accuracy', acc_tr)
    print('F1', f1_tr)
    print('Test')
    print('Precision', prec_ts)
    print('Recall', rec_ts)
    print('Accuracy', acc_ts)
    print('F1', f1_ts)
    plt.plot([i for i in range(500)], J1, label='Cost_function')
    plt.title(f'Number of iterations *{len(X_train)}(Parkinsons dataset)')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()


df, cat_list, num_list = process_data('parkinsons.csv', 'Diagnosis')
print(df.head())
train_test(df, 'Diagnosis', cat_list, num_list)