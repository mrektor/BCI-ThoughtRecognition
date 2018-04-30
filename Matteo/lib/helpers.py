import numpy as np

def add_ones(tx):
    """
	Add column of ones to the dataset tx
    """
    return np.concatenate((tx, np.ones([tx.shape[0],1])), axis=1)

def standardize(x):
    """Standardize the data set x."""
    # Compute the mean for each column
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    # Compute the standard deviation for each column
    std_x = np.std(x, axis=0)
    x = x / std_x
    return np.array(x)



def build_poly(x, degree):
    """ Returns the polynomial basis functions for input data x, for j=2 up to j=degree."""
    new_cols=np.array([x**p for p in range(2,degree+1)]).T;
    return new_cols

def add_powers(tx, degree):
    for col in range(0,tx.shape[1]): 
            tx = np.concatenate((tx, build_poly(tx[:,col], degree)), axis=1)
    return tx




def get_accuracy(predicted_labels, true_labels):
     if (predicted_labels.size == true_labels.size):
        return  np.sum(predicted_labels ==  true_labels )/len( true_labels)
   

   