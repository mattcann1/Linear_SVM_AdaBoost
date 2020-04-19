#SYDE 675
#Matthew Cann
#20863891
#Assignment 3 Question 1


#Linear SVM for Two-class Problem

#.......................................................................IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#.....................................................................CONSTANTS
C_list = [0.1, 1 , 10 , 100]
C_best = 1
#.....................................................................FUNCTIONS
def get_classA_dataset():
    '''Reads in dataset 1'''
    os.chdir('E:\Documents\Waterloo-Masters\SYDE 675\Assignment 3\Data')
    df = pd.read_csv('classA.csv', delimiter=',', header = None)

    df.columns=['x1','x2']
    return df

def get_classB_dataset():
    '''Reads in dataset 2'''
    os.chdir('E:\Documents\Waterloo-Masters\SYDE 675\Assignment 3\Data')
    df = pd.read_csv('classB.csv', delimiter=',', header = None)

    df.columns=['x1','x2']

    return df

def plot_boundary(model, dataframe, label):
    '''Plots the SVM linear line from the fitted model'''
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace((dataframe.iloc[:,0].values).min(),(dataframe.iloc[:,0].values).max())
    yy = a * xx - model.intercept_[0] / w[1]
    plt.plot(xx, yy, label = label)
    return
def plot_boundary_2(model, x_min, x_max, label):
    '''Plots the SVM linear line from the fitted model'''
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(x_min,x_max)
    yy = a * xx - model.intercept_[0] / w[1]
    plt.plot(xx, yy, label = label)
    return

def SVM_classifier(C_list, dataframe):
    '''Classifies the dataframe using a fitted SVM model with the C values in C_list, plots the decision boundary of the varying C values'''    
    X_train = dataframe.iloc[:,0:2].values
    y_train = dataframe.iloc[:,2].values
    
    df_0 = dataframe[dataframe['class'] == 0]
    df_1 = dataframe[dataframe['class'] == 1]
    
    plt.scatter(df_0.iloc[:,0].values, df_0.iloc[:,1].values, label = 'Class A')
    plt.scatter(df_1.iloc[:,0].values, df_1.iloc[:,1].values, label = 'Class B')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    for C_i in C_list:
        #svc = LinearSVC(C = C_i, max_iter = 200000)
        svc = svm.SVC(kernel='linear', C = C_i) # linear SVM
        svc.fit(X_train, y_train)
        plot_boundary_2(svc, 180, 425, C_i)
                
    plt.legend()
    plt.savefig('Q2_part2.png')    
    plt.show()
    return

def accuracy(y_true, y_predicted):
    '''Reports the accuracy of two lists of true and predicted values'''
    correct = 0
    for true, pred in zip(y_true, y_predicted):        
        if float(true) == float(pred):
            correct += 1
    accuracy = correct/len(y_predicted)*100
    print('Accuracy of classifer {:0.2f} %' .format(accuracy))
    return accuracy
def fold_cross_val(dataframe, num_folds, C):
    '''10 fold cross validation'''
    
    #Scaled features for faster processing

    
    scaled_features = StandardScaler().fit_transform(dataframe.values)
    scaled_features_df = pd.DataFrame(scaled_features, index=dataframe.index, columns=dataframe.columns)
    scaled_features_df['class'] = dataframe['class']
    
    #Shuffle Dataframe
    df_shuffle = scaled_features_df.iloc[np.random.permutation(len(scaled_features_df))]
    df_shuffle = df_shuffle.reset_index(drop=True) #Reset the index to begin at 0
    

    folds = num_folds    #Calls number of folds
    fold_size = int(df_shuffle.shape[0]/folds) # Determines the size of the folds
    
    accuracy_list = [] #makes empty list to store accuracy values
    #y_pred_master = []
    #y_test_master = []
    
    start = 0 # initalize the start
    end = fold_size # initalize the end
    for i in range(folds):
        print('\t Calculating fold number {} of {} number if folds...'.format(i+1, folds))
        
        #For the final cut, if the fold makes a sliver of data left over, the test data will take the extra data. 
        len_dataframe = len(df_shuffle)
        if (len_dataframe - end) < fold_size:
            end = len_dataframe
            
        #Test Dataframe    
        df_test = df_shuffle.iloc[start:end] #dataframe of test values from the fold
        y_test = df_test.iloc[:,-1] #True values labeled
        df_test = df_test.drop(labels='class', axis=1) # removes the label column from df_test
        X_test = df_test.iloc[:,0:2].values

        #print(X_test)
        #Train Dataframe
        drop_index = list(range(start,end))
        df_train = df_shuffle.drop(drop_index) #, axis = 0)
        
        start += fold_size
        end += fold_size
        
        X_train = df_train.iloc[:,0:2].values
        y_train = df_train.iloc[:,2].values
        
        #Train SVM
        #svc = LinearSVC(C = C, max_iter = 100000)
        svc = svm.SVC(kernel='linear', C = C)
        svc.fit(X_train, y_train)
        
        #Predict
        y_pred = svc.predict(X_test)
        
        #Accuracy
        accuracy_i = accuracy(y_test, y_pred)
        accuracy_list.append(accuracy_i)
        
    return accuracy_list


def cross_validation(times,dataframe, C):
    '''Performs 10 fold cross validation 'times' number of times'''
    master_acc = []
    for i in range(times):
        print('Calculating {} of {} times - 10 fold cross validation...'.format(i, times))
        accuracy_list = fold_cross_val(dataframe, 10, C)
        master_acc.append(accuracy_list)
    accuracy_flat = [y for x in master_acc for y in x]
    return accuracy_flat

def stats_info(list_accuracies):
    '''Detemines the statistical quantities of mean, varience, and std from the list of accuracies'''
    mean = sum(list_accuracies) / len(list_accuracies)
    variance = sum([((x - mean) ** 2) for x in list_accuracies]) / len(list_accuracies)
    std = variance ** 0.5
    print('Mean Cross-Validation Accuracy: \t\t\t{:.2f}'.format(mean))
    print('Standard Deviation of Cross-Validation Accuracy: \t{:.2f}'.format(std))
    print('Variance of Cross-Validation Accuracy: \t\t\t{:.2f}'.format(variance))

    return mean, variance, std

def update_weights(y_weak_pred, y_train, beta_t, sample_weight):
    '''Updates the sample weights using the correct predictions'''
    
    correct = y_weak_pred == y_train #Array of boolean values if the predicted values matches the training values
    update = []
    for x in correct:
        if x == 1:
            update.append(beta_t) #If correct, update list with beta
        else:
            update.append(1) #If incorrecet, update list with 1
            
    sample_weight = (sample_weight)*np.array(update)
    Z = sample_weight.sum() # Used to re-distribute the weights
    sample_weight = sample_weight/Z
    return sample_weight

def m1_algorithm(N_samples, X_train, y_train, C, T):
    '''Adaboost m1 algorithm'''
    
    N = len(y_train) #Len of the training set
    #Initialize the sample weights
    sample_weight = np.ones(N)/N
    
    condition = "continue" # Initilze the loop
    model_list, alpha_list = [], []

    while (condition == "continue"):
        row_i = np.random.choice(X_train.shape[0], N_samples, p=sample_weight)
        X_train_sample = X_train[row_i]
        y_train_sample = y_train[row_i]
        
        #Weak learner linear SVM fit and predict
        weak_learner = SVC(kernel='linear', C=C)
        weak_learner.fit(X_train_sample, y_train_sample)
        y_weak_pred = weak_learner.predict(X_train)
        
        #Step 3: Hypothesis error
        incorrect = y_weak_pred != y_train
        error_t = np.dot(incorrect, sample_weight)#/sum(sample_weight)
        
        #Step 4:
        beta_t =   error_t / (1 - error_t)
         #Hypothesis weight
        alpha_t = np.log(1/beta_t)
        
        if error_t >= 0.5:
            continue    
        else:
            sample_weight = update_weights(y_weak_pred, y_train, beta_t, sample_weight)
            model_list.append(weak_learner)
            alpha_list.append(alpha_t)
        
        if len(alpha_list) == T:
            condition = 'break'
    return alpha_list, model_list


def predict_ensemble(X_test, alpha_list, model_list):
    '''Combines the ensemble of weak learner predictions into one prediction for the class labels'''
    
    y_pred_list = []
    for alpha, model in zip(alpha_list, model_list):
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
        
    y_pred_list = np.asarray(y_pred_list)
    y_pred_ensemble = []
    
    for point in range(X_test.shape[0]):
        points = y_pred_list[:,point] # Takes the column which is the y values for each model at a singel point.
        index_0 = np.where(points == 0)[0] #Index values of 0 class
        index_1 = np.where(points == 1)[0] #Index values of 1 class

        alpha_array = np.array(alpha_list)
        alpha_0 = alpha_array[index_0].sum() #summation of alpha values when class label is 0 for the weak learners
        alpha_1 = alpha_array[index_1].sum() #summation of alpha values when class label is 1 for the weak learners
        

        if alpha_0 >= alpha_1:
            y_pred_ensemble.append(0)
        else:
            y_pred_ensemble.append(1)
        
    return y_pred_ensemble
        
def fold_cross_val_model(dataframe, num_folds):
    '''10 fold cross validation'''

    #Shuffle Dataframe
    df_shuffle = dataframe.iloc[np.random.permutation(len(dataframe))]
    df_shuffle = df_shuffle.reset_index(drop=True) #Reset the index to begin at 0
    

    folds = num_folds    #Calls number of folds
    fold_size = int(df_shuffle.shape[0]/folds) # Determines the size of the folds
    
    accuracy_list = [] #makes empty list to store accuracy values    
    start = 0 # initalize the start
    end = fold_size # initalize the end
    
    for i in range(folds):
        print(i)
        print('\t Calculating fold number {} of {} number if folds...'.format(i+1, folds))
        
        #For the final cut, if the fold makes a sliver of data left over, the test data will take the extra data. 
        len_dataframe = len(df_shuffle)
        if (len_dataframe - end) < fold_size:
            end = len_dataframe
            
        #Test Dataframe    
        df_test = df_shuffle.iloc[start:end] #dataframe of test values from the fold
        y_test = df_test.iloc[:,-1] #True values labeled
        df_test = df_test.drop(labels='class', axis=1) # removes the label column from df_test
        X_test = df_test.iloc[:,0:2].values
        
        #Train Dataframe
        drop_index = list(range(start,end))
        df_train = df_shuffle.drop(drop_index) #, axis = 0)
        X_train = df_train.iloc[:,0:2].values #Training set X
        y_train = df_train.iloc[:,2].values # training set y class labels
        
        #M1 Algortihm
        alpha_list, model_list = m1_algorithm(100, X_train, y_train, C_best, 50)
        
        #Ensemble prediction of y 
        y_pred_ensemble = predict_ensemble(X_test, alpha_list, model_list)
        
        y_test = list(y_test)

        accuracy_i = accuracy(y_test, y_pred_ensemble)
        accuracy_list.append(accuracy_i)
        
        start += fold_size
        end += fold_size

    return accuracy_list


def cross_validation_model(times,dataframe):
    '''10 times cross validation'''
    master_acc = []
    for i in range(times):
        print('Calculating {} of {} times - 10 fold cross validation...'.format(i, times))
        accuracy_list = fold_cross_val_model(dataframe, 10)
        master_acc.append(accuracy_list)        
    accuracy_flat = [y for x in master_acc for y in x]
    return accuracy_flat

def main():
    Directory = "E:\Documents\Waterloo-Masters\SYDE 675\Assignment 3"
    os.chdir(Directory)
    
    dfA = get_classA_dataset()
    dfB = get_classB_dataset()
    
    #PART 1 .......................................................................
    plt.scatter(dfA.iloc[:,0].values, dfA.iloc[:,1].values, label = 'Class A')
    plt.scatter(dfB.iloc[:,0].values, dfB.iloc[:,1].values, label = 'Class B')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.savefig('Q2.part1.png')
    plt.show()
    
    dfA['class'] = [0]*dfA.shape[0]
    dfB['class'] = [1]*dfB.shape[0]
    
    
    df = pd.concat([dfA, dfB],ignore_index=True)
    
    #PART 2........................................................................
    
    X_train = df.iloc[:,0:2].values
    y_train = df.iloc[:,2].values
    
    SVM_classifier(C_list, df)
    
    mean_list = []
    for C_i in C_list:
        accuracy_list = cross_validation(10, df, C_i)
        mean, variance, std = stats_info(accuracy_list)
        mean_list.append(mean)
    
    print(C_list)
    print('{:0.2f}%,  {:0.2f}%,  {:0.2f}%,  {:0.2f}%'.format(mean_list[0], mean_list[1],mean_list[2],mean_list[3]))
    
    #PART 3.......................................................................
    #AdaBoost.M1.................................................................
    
    #PART 4........................................................................
    X_train = df.iloc[:,0:2].values #Training set X
    y_train = df.iloc[:,2].values # training set y class labels
            
    N_samples = 100
    C_best = 1
    accuracy_list_2 = cross_validation_model(10, df)
    mean, variance, std = stats_info(accuracy_list_2)
    print(mean)
    
    #PART 5........................................................................
    
    x_min, x_max = df.iloc[:, 0].min() - .1, df.iloc[:, 0].max() + .1
    y_min, y_max = df.iloc[:, 1].min() - .1, df.iloc[:, 1].max() + .1
    
    xx, yy = np.meshgrid( np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    X_grid = np.array([xx.ravel(), yy.ravel()]).T 
    
    X_train = df.iloc[:,0:2].values #Training set X
    y_train = df.iloc[:,2].values # training set y class labels
            
    
    alpha_list, model_list = m1_algorithm(N_samples , X_train, y_train, C_best, 50)
    
    
    y_pred_grid = predict_ensemble(X_grid, alpha_list, model_list)
    
    #Plots ensemble boundary condiion
    X_train = df.iloc[:,0:2].values
    y_train = df.iloc[:,2].values
    
    df_0 = df[df['class'] == 0]
    df_1 = df[df['class'] == 1]
    
    
    lo = plt.scatter(df_0.iloc[:,0].values, df_0.iloc[:,1].values, label = 'Class A')
    ll = plt.scatter(df_1.iloc[:,0].values, df_1.iloc[:,1].values, label = 'Class B')
    
    plt.contour(xx,yy, np.array(y_pred_grid).reshape(100,100), colors = 'black', linewidths =0.5, label = 'Ensemble Classifier')
    from matplotlib.lines import Line2D
    
    legend_element = [Line2D([0], [0], linestyle='-', color = 'black', label = 'Ensemble Classifier'), lo, ll]
    
    plt.legend(handles=legend_element)
    plt.savefig('Q2_part_final.png')
    plt.show()
    return

#..........................................................................MAIN

main()


#%%


















