def standardize(x):
    """Standardize the data set x."""
    # Compute the mean for each column
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    # Compute the standard deviation for each column
    std_x = np.std(x, axis=0)
    x = x / std_x
    return np.array(x), mean_x, std_x


def get_accuracy(predicted_labels, true_labels):
     if (predicted_labels.size == true_labels.size):
        return np.count_nonzero((predicted_labels-true_labels))/predicted_labels.size
   