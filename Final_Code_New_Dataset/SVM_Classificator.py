import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, neighbors, datasets
from sklearn import svm
import scipy.signal as signal
import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import LeavePOut
from sklearn.preprocessing import StandardScaler
import sys
import time
import os
import shutil
from sklearn.externals import joblib
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


##############  INPUT PARAMETERS ####################################
Model_Name='Day6'  # Write the name of the model
Input_Path_Folder='./DataDay6'
 # Select the Input folder of the files 

num_experiment = 5000 # number of experiments to validate the data

#####################################################################




########################################################   HELPERS FUNCTIONS   ###################################################################################



def split_matrix_two_blocks(y, percentage1, percentage2, seed):
    """Build k indices for k-fold."""
    if(percentage1+percentage2==1):
        num_row = len(y)
        #print(num_row)
        interval_1 = int(percentage1*num_row);
        
        np.random.seed(seed)
        indices = np.random.permutation(num_row);
        first_indices = indices[0:interval_1];
        second_indices = indices[interval_1:num_row];
        return [np.array(first_indices),np.array(second_indices)]
    else:
        print('>>>>>>>>>>>ERROR:Not valid splitting percentage')
        
        
##
## This function reutrn a list of matrices. Each matrix correspond to a question instance in which each row is a channel, and in the coloumn it develop the signal in time
## The function also manage to standardize the time length
def channels_to_vector(channels): 
    time_instances=[];
    dim=channels.shape;
    #find the length min of the signal in the specified temporal instance
    length_min=len(channels[0,1]);
    for i in range (0,dim[1]):
        single_measurement=channels[0,i];
        single_length=single_measurement.shape[0]
        if(single_length<length_min):
                length_min=single_length;
    #export the signals
    for i in range (0,dim[1]):
        single_measurement=channels[0,i];
        dim1=single_measurement.shape;
        time_instance=[];
        for j  in range (0,dim1[1]):
            if(len(single_measurement[:,j])>length_min):
                single_signal=single_measurement[:,j][0:length_min]
            else:
                single_signal=single_measurement[:,j]
            #put in a list 
            time_instance.append(np.asarray(single_signal).reshape(len(single_signal),1).T);
       # create the matrix of the signals per a single time instance 
        time_instance=np.concatenate(time_instance);
        time_instances.append(time_instance);   
    return time_instances;


##
# Create the train data matrix
##
## usage
def get_feature_matrix_and_labels(channel_structure,label,features_extracted,connectivity_feature):
    list_train=[]
    list_labels=[]
    cont=0;
    index_connectivity=0;
    list_row=[]
    
    for time_instance in channel_structure:
        dim1=time_instance.shape
        #indipendent_components=extract_ICs(time_instance,n_ICA_components);
        for j  in range (0,dim1[0]):
           
            features=features_extracted[cont,:];
            list_row.append(features);
            cont=cont+1;
        list_row.append(connectivity_feature[index_connectivity,:]);
        index_connectivity=index_connectivity+1;
        labels=get_labels(1,label);
        feature_row=np.concatenate(list_row);
        list_train.append(feature_row.reshape(len(feature_row),1).T)
        list_labels.append(labels);
        list_row=[]
        
    train_TX=np.concatenate(list_train)
    labels=np.concatenate(list_labels,axis=0)
    
    return train_TX,labels.T.reshape(labels.size)


### Description
def get_labels(number, string):
    if(string=="No"):
        return np.zeros(number)    
    if(string=="Yes"):
        return np.ones(number)
    
## description
def select_features(weights,matrix,th):
    cont=0;
    i=0;
    while(cont<len(weights)):
        if(weights[cont]<th):

            mask = np.ones(matrix.shape[1], dtype=bool)
            mask[i] = False
            matrix=matrix[:,mask]
        else:
            i=i+1;
        cont=cont+1;
    return matrix


def get_accuracy(predicted_labels, true_labels):
     if (predicted_labels.size == true_labels.size):
        return  np.sum(predicted_labels ==  true_labels )/len( true_labels)
    
    

def classification_SVM_experiments_std(X, Y, classifier, fraction_train_test, num_experiments,Model_Name,save_model):
    
    seed=range(num_experiments)
    svm_total_acc_test  = []
    svm_total_acc_train = [] 
    dataset_length=X.shape[0];
    
    for single_seed in seed:
        scaler = StandardScaler()
        
        [i1,i2]=split_matrix_two_blocks(X, fraction_train_test, 1-fraction_train_test,single_seed)
        
        
        train =X[i1,:]
        labels_train=Y[i1]
        
        test = X[i2,:]
        labels_test=Y[i2]
        
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
  
        clf=classifier
        clf.fit(train, labels_train)  

        
        #Accuracy on test
        predicted_labels_test = clf.predict(test)
        SVM_accuracy_test = get_accuracy(predicted_labels_test, labels_test)
        svm_total_acc_test.append(SVM_accuracy_test)
        
        if(save_model==1):
            old_path=os.getcwd()
            os.chdir('Model'+Model_Name)
            joblib.dump(clf, 'Classifier_'+Model_Name+'.pkl') 
            os.chdir(old_path)
    
        #Accuracy on train
        predicted_labels_train = clf.predict(train)
        SVM_accuracy_train = get_accuracy(predicted_labels_train, labels_train)
        svm_total_acc_train.append(SVM_accuracy_train)
        #print("Accuracy: "+ str(SVM_accuracy) + "; iteration  " + str(single_seed) )
    return svm_total_acc_test, svm_total_acc_train

def performance_assesment_fraction_std(X, Y, num_experiment, classifier,Model_Name):
    fracs = np.linspace(0.2,0.9,X.shape[0])
    accuracy_test_mean  = []
    accuracy_test_std   = []
    accuracy_train_mean = []
    accuracy_train_std  = []

    for frac_tr_te in fracs:
        print("Evaluation progress: " + str(int((frac_tr_te-fracs[0])/(fracs[-1]-fracs[0])*100)) + " %")
        acc_test, acc_train = classification_SVM_experiments_std(X, Y, classifier, frac_tr_te, num_experiment,Model_Name,0)
        #saving of metrics of interest
        accuracy_test_mean.append(np.mean(acc_test))
        accuracy_test_std.append(np.std(acc_test))
        accuracy_train_mean.append(np.mean(acc_train))
        accuracy_train_std.append(np.std(acc_train))
    
    frac_tr_te=0.95;
    acc_test, acc_train =classification_SVM_experiments_std(X, Y, classifier, frac_tr_te, 1,Model_Name,1)
    #plot the figure
    plt.figure(figsize=(10, 7), dpi=80)
    plt.errorbar(np.floor(fracs*60), accuracy_test_mean, yerr=accuracy_test_std, label="Error bars plot", fmt="s-",  linewidth=3)
    plt.errorbar(np.floor(fracs*60), accuracy_train_mean, yerr=accuracy_train_std, label="Error bars plot", fmt="s-",  linewidth=3)
    plt.grid(b=True, which='major', color='k', linestyle='--', alpha = 0.4)
    plt.minorticks_on()
    plt.title('SVM perfomances over different train/test dataset of reduced features')
    plt.ylabel('Accuracy')
    plt.xlabel('Train instances considered')

    plt.legend(['Test Accuracy', 'Train Accuracy'], loc=4)
    name='train_test_acc_fine_tuned_IID_'+Model_Name+'.eps'
    plt.savefig(name, format='eps')
    plt.show()
    
def split_train_test_temporal(X, y, n):
    #n is the number of instances of yes and no, n>1
    half = int(X.shape[0]/2)
    if (n>half): n=half
    X_tr = np.concatenate(( X[0:n,:] , X[half:half+n,:] ))
    X_te = np.concatenate(( X[n:half,:], X[half+n:,:] ))
    y_tr = np.concatenate(( y[0:n]   , y[half:half+n]   ))
    y_te = np.concatenate(( y[n:half]  , y[half+n:]   ))
    
    return X_tr, X_te, y_tr, y_te
###########################################################################################################################################

# STARTING THE CODE



dir_name='Model'+Model_Name;
if os.path.exists(dir_name):
    shutil.rmtree(dir_name)
os.mkdir('Model'+Model_Name)

old_path=os.getcwd()
os.chdir(Input_Path_Folder)


# LOAD DATA

#Import data from mat files
yes_EEG_contents = sio.loadmat('EEGyes.mat')
no_EEG_contents = sio.loadmat('EEGno.mat')

channels_no_EEG=no_EEG_contents["EEGno"]
channels_yes_EEG=yes_EEG_contents["EEGyes"]

#Features Loading
features_extracted_yes   = sio.loadmat('FeaturesYes.mat')['FeaturesYes']
features_extracted_no    = sio.loadmat('FeaturesNO.mat')['FeaturesNo']
connectivity_feature_yes = sio.loadmat('ConnectivityFeaturesYes.mat')['ConnectivityFeaturesYes']
connectivity_feature_no  = sio.loadmat('ConnectivityFeaturesNo.mat')['ConnectivityFeaturesNo']

channels_structure_yes_EEG = channels_to_vector(channels_yes_EEG)
channels_structure_no_EEG  = channels_to_vector(channels_no_EEG)

##Structuring of the data:
#the code below create the train matrix with respect to the signal given in "channel_structure" but using the features contained in "features_extracted*" and in "connettivity_feature*".
feature_dataset_yes_EEG, EEG_yes_labels = get_feature_matrix_and_labels(channels_structure_yes_EEG,"Yes",features_extracted_yes,connectivity_feature_yes);

feature_dataset_no_EEG, EEG_no_labels = get_feature_matrix_and_labels(channels_structure_no_EEG,"No",features_extracted_no,connectivity_feature_no);

#Merge the labeled data
feature_dataset_full = np.concatenate((feature_dataset_yes_EEG, feature_dataset_no_EEG), axis=0 )
labels = np.concatenate((EEG_yes_labels,EEG_no_labels), axis=0)
os.chdir(old_path)




############## SELECT THE BEST K-Features #############################

print("Selecting the best k features:");
time.sleep(2) ;
tot_perf = []
end_feat = 100
for k in range(1,end_feat):
    #print report
    #sys.stdout.write("\033[F") # Cursor up one line
    # sys.stdout.write("\033[K") # Clear to the end of line
    print('progress: ' + str(k/float(end_feat)*100) + ' %');
    
    
    #reduce with anova f-test score the data in k features
    reducer = SelectKBest(f_classif, k)
    feature_reduced = reducer.fit_transform(feature_dataset_full, labels)
    
    
    clf = svm.SVC(C = 1, kernel = 'linear', gamma = 'auto') #Parameters of the SVM
    fraction_train_test = 0.30 #Division tran/test of the full dataset
    num_experiments = 1000 #Number of different partition for better evaluating the performances
    
    #Function that actually computes the performances on the k features 
    perf_temp = classification_SVM_experiments_std(feature_reduced, labels, clf, fraction_train_test, num_experiments,Model_Name,0) #Note: this method scale the data to have 0-mean and unit variance
    tot_perf.append(np.mean(perf_temp)) #Save the result


best_k = np.argmax(tot_perf)+1
print("smallest k that gives best low train results:");
print(best_k)
print()
print("wich have lead to the top performance:");
print(np.max(tot_perf))


#What are those features?
reducer = SelectKBest(f_classif, best_k)
feature_dataset_reduced = reducer.fit_transform(feature_dataset_full, labels)
boolean_vec = reducer.get_support()
idx =[]
for i in range(len(boolean_vec)): 
    if boolean_vec[i] == True: idx.append(i)
print()
print("index of features retained:");
print(idx)


#save the model 
old_path=os.getcwd()
os.chdir('Model'+Model_Name)
joblib.dump(reducer, 'reducer_'+Model_Name+'.pkl')
os.chdir(old_path)

######################################################


###### Evaluate different train/test fraction ######################
print("Evaluating on different train/test fraction");
k=best_k
reducer = SelectKBest(f_classif, k)
feature_dataset_reduced = reducer.fit_transform(feature_dataset_full, labels)
clf = svm.SVC(C = 1, kernel = 'linear', gamma = 'auto')
performance_assesment_fraction_std(feature_dataset_reduced, labels, num_experiment, clf,Model_Name)

####################################################################




