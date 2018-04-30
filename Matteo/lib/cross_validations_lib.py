import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, neighbors,datasets
from sklearn import svm
from lib.helpers import *


def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_logistic_regularized(Y,X, degrees, lambdas, k_fold, seed, max_iters):
    
    # Get the indices so that we get the k'th subgroup in test, others in train, for each k
    k_indices = build_k_indices(Y, k_fold, seed)
    
    # Initialize matrix of computed accuracies for each degree and each fold
    accuracies_train_by_fold = np.zeros([len(degrees), len(lambdas), k_fold])
    accuracies_test_by_fold = np.zeros([len(degrees), len(lambdas), k_fold])

    
    for k in range(k_fold):
        print('--- Fold', k, '---')
        # Create the testing set for this fold number
        k_index = k_indices[k] # Indices of the testing set for fold k
        Y_cross_val_test = Y[k_index]
        X_cross_val_test = X[k_index,:]
        
        
        # Create the training set for this fold number
        mask = np.ones(len(Y), dtype=bool) # set all elements to True
        mask[k_index] = False # set test elements to False
        Y_cross_val_train = Y[mask] # select only True elements (ie train elements)
        X_cross_val_train = X[mask,:]
       
        # Compute the accuracies for each degree
        accuracies_train_by_fold[:,:,k], accuracies_test_by_fold[:,:,k] = cross_validation_one_fold_logistic_regularized\
            (Y_cross_val_train, Y_cross_val_test, X_cross_val_train, X_cross_val_test, \
                                 degrees, lambdas,max_iters)
    # Compute the mean accuracies over the folds, for each degree
    mean_accuracies_train_by_deg = np.mean(accuracies_train_by_fold, axis=2)
    mean_accuracies_test_by_deg = np.mean(accuracies_test_by_fold, axis=2)
    
    # Get the index of the best accuracy in the testing set
    max_id_deg_test,max_id_lambda = \
        np.unravel_index(mean_accuracies_test_by_deg.argmax(), mean_accuracies_test_by_deg.shape)
    
    # Find the optimal degree and the corresponding accuracies in the training and testing sets
    best_deg = degrees[max_id_deg_test]
    best_lambda=lambdas[max_id_lambda]
    best_accuracy_test = mean_accuracies_test_by_deg[max_id_deg_test,max_id_lambda]
    corresponding_accuracy_train = mean_accuracies_train_by_deg[max_id_deg_test,max_id_lambda]
    
    print('Best accuracy test =', best_accuracy_test, 'with degree =', best_deg , 'lambda=',best_lambda)
    print('Corresponding accuracy train =', corresponding_accuracy_train)
    
    return best_deg, best_lambda, best_accuracy_test, corresponding_accuracy_train                        


def cross_validation_one_fold_logistic_regularized(y_cross_val_train, y_cross_val_test, tx_cross_val_train, tx_cross_val_test, \
                                 degrees, lambdas, max_iters):
    
    accuracies_train_by_deg = np.zeros([len(degrees),len(lambdas)])
    accuracies_test_by_deg = np.zeros([len(degrees),len(lambdas)])
    
    # For each degree, compute the least squares weights, the predictions and the accuracies
    for deg_id, deg in enumerate(degrees):
        print('++ Degree', deg, '++')
                
        # Add powers of the chosen columns
        len_data = tx_cross_val_train.shape[1]
        tx_cross_val_train = add_powers(tx_cross_val_train,deg )
       
        
        tx_cross_val_test = add_powers(tx_cross_val_test,deg)
  
        
        
        for lambda_id, single_lambda in enumerate(lambdas):
                
                print('>> Lambda', single_lambda, '<<')
                # Compute the best weights on the training set
                logreg = linear_model.LogisticRegression(C=1/single_lambda, class_weight="balanced",max_iter=max_iters)
                logreg.fit(tx_cross_val_train,y_cross_val_train )

                # Compute the predictions
                y_predicted_cross_val_train = logreg.predict(tx_cross_val_train)
                y_predicted_cross_val_test = logreg.predict(tx_cross_val_test)



                # Compute the accuracies for each degree
                accuracies_train_by_deg[deg_id,lambda_id] = \
                    np.sum(y_predicted_cross_val_train == y_cross_val_train)/len(y_cross_val_train)
                accuracies_test_by_deg[deg_id,lambda_id] = \
                    np.sum(y_predicted_cross_val_test == y_cross_val_test)/len(y_cross_val_test)


                print(accuracies_test_by_deg[deg_id,lambda_id])
        
        
    return accuracies_train_by_deg, accuracies_test_by_deg





def cross_validation_SVM(Y,X, C_parameters, kernel_types,k_fold, seed, max_iters):
    
    # Get the indices so that we get the k'th subgroup in test, others in train, for each k
    k_indices = build_k_indices(Y, k_fold, seed)
    
    # Initialize matrix of computed accuracies for each degree and each fold
    accuracies_train_by_fold = np.zeros([len(C_parameters),len(kernel_types), k_fold])
    accuracies_test_by_fold = np.zeros([len(C_parameters),len(kernel_types), k_fold])

    
    for k in range(k_fold):
        print('--- Fold', k, '---')
        # Create the testing set for this fold number
        k_index = k_indices[k] # Indices of the testing set for fold k
        Y_cross_val_test = Y[k_index]
        X_cross_val_test = X[k_index,:]
        
        
        # Create the training set for this fold number
        mask = np.ones(len(Y), dtype=bool) # set all elements to True
        mask[k_index] = False # set test elements to False
        Y_cross_val_train = Y[mask] # select only True elements (ie train elements)
        X_cross_val_train = X[mask,:]
       
        # Compute the accuracies for each degree
        accuracies_train_by_fold[:,:,k], accuracies_test_by_fold[:,:,k] = cross_validation_one_fold_SVM\
            (Y_cross_val_train, Y_cross_val_test, X_cross_val_train, X_cross_val_test, \
                                 C_parameters, kernel_types,max_iters)
    # Compute the mean accuracies over the folds, for each degree
    mean_accuracies_train = np.mean(accuracies_train_by_fold, axis=2)
    mean_accuracies_test = np.mean(accuracies_test_by_fold, axis=2)
    
    # Get the index of the best accuracy in the testing set
    
    
    max_id_C_parameter, max_id_kernel= \
        np.unravel_index(mean_accuracies_test.argmax(), mean_accuracies_test.shape)
    
    # Find the optimal degree and the corresponding accuracies in the training and testing sets
    best_C_parameter=C_parameters[max_id_C_parameter]
    best_kernel_type=kernel_types[max_id_kernel]
    best_accuracy_test = mean_accuracies_test[max_id_C_parameter,max_id_kernel]
    corresponding_accuracy_train = mean_accuracies_train[max_id_C_parameter,max_id_kernel]
    
    print('Best accuracy test =', best_accuracy_test , 'Penalty parameter=',best_C_parameter, 'kernel type=',best_kernel_type)
    print('Corresponding accuracy train =', corresponding_accuracy_train)
    
    return best_C_parameter, best_kernel_type, best_accuracy_test, corresponding_accuracy_train
    


def cross_validation_one_fold_SVM(y_cross_val_train, y_cross_val_test, tx_cross_val_train, tx_cross_val_test, \
                                  C_parameters, kernel_types, max_iters):
    
    accuracies_train = np.zeros([len(C_parameters),len(kernel_types)])
    accuracies_test = np.zeros([len(C_parameters),len(kernel_types)])
    
    # For each degree, compute the least squares weights, the predictions and the accuracies
 

        
        
    for C_id, single_C in enumerate(C_parameters):
                
        print('>> Penalty parameter C', single_C, '<<')
                
        for kernel_id, single_kernel in enumerate(kernel_types):
            print('>> Type of Kernel ',single_kernel, '<<')
                
           
                
                # Compute the best weights on the training set
            clf = svm.SVC(C=single_C, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel=single_kernel,
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
            clf.fit(tx_cross_val_train,y_cross_val_train )  

                # Compute the predictions
            y_predicted_cross_val_train = clf.predict(tx_cross_val_train)
            y_predicted_cross_val_test =clf.predict(tx_cross_val_test)



                # Compute the accuracies for each degree
            accuracies_train[C_id,kernel_id] = \
                        np.sum(y_predicted_cross_val_train == y_cross_val_train)/len(y_cross_val_train)
            accuracies_test[C_id,kernel_id] = \
                        np.sum(y_predicted_cross_val_test == y_cross_val_test)/len(y_cross_val_test)


            print(accuracies_test[C_id,kernel_id],accuracies_train[C_id,kernel_id])
        
        
    return accuracies_train, accuracies_test

