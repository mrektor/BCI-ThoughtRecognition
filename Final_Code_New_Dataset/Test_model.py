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
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)







######################### Helpers Functions ##########################

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
 
    
    
def extract_dataset(destination_folder):
    #Import data from mat files
    old_path=os.getcwd()
    os.chdir(destination_folder)
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


    return feature_dataset_full,labels
    
    
   
   # Load Data


# Load Data

#################### DAY 1  ###########################
#Select the traine  classifier

classifier_folder= 'ModelDay1'
classifier_name='Classifier_Day1'
reducer_name='reducer_Day1'

#Select the input folders to test the model
Input_Path_Folders=['./DataDay1','./DataDay2','./DataDay4','./DataDay5','./DataDay6']



old_path=os.getcwd()
os.chdir(classifier_folder)
clf = joblib.load(classifier_name+'.pkl') 
reducer=joblib.load(reducer_name+'.pkl') 
os.chdir(old_path)


print('Model trained: '+classifier_name)
for single_input in Input_Path_Folders:
    print('Accuracy on dataset in:' + single_input)
    [feature_dataset_full,labels]=extract_dataset(single_input)
    dataset_reduced=reducer.fit_transform(feature_dataset_full,labels)
    scaler= StandardScaler()
    dataset_reduced = scaler.fit_transform(dataset_reduced)
    predicted_label=clf.predict(dataset_reduced)
    print(get_accuracy(predicted_label,labels))



 ################# DAY 2 ####################################

#Select the traine  classifier

classifier_folder= 'ModelDay2'
classifier_name='Classifier_Day2'
reducer_name='reducer_Day2'

#Select the input folders to test the model
Input_Path_Folders=['./DataDay1','./DataDay2','./DataDay4','./DataDay5','./DataDay6']

old_path=os.getcwd()
os.chdir(classifier_folder)
clf = joblib.load(classifier_name+'.pkl') 
reducer=joblib.load(reducer_name+'.pkl') 
os.chdir(old_path)





print('Model trained: '+classifier_name)
for single_input in Input_Path_Folders:
    print('Accuracy on dataset in:' + single_input)
    [feature_dataset_full,labels]=extract_dataset(single_input)
    dataset_reduced=reducer.fit_transform(feature_dataset_full,labels)
    scaler= StandardScaler()
    dataset_reduced = scaler.fit_transform(dataset_reduced)
    predicted_label=clf.predict(dataset_reduced)
    print(get_accuracy(predicted_label,labels))


 ################# DAY 4 ####################################

# INPUT PARAMETERS 

#Select the traine  classifier

classifier_folder= 'ModelDay4'
classifier_name='Classifier_Day4'
reducer_name='reducer_Day4'

#Select the input folders to test the model
Input_Path_Folders=['./DataDay1','./DataDay2','./DataDay4','./DataDay5','./DataDay6']

old_path=os.getcwd()
os.chdir(classifier_folder)
clf = joblib.load(classifier_name+'.pkl') 
reducer=joblib.load(reducer_name+'.pkl') 
os.chdir(old_path)




print('Model trained: '+classifier_name)
for single_input in Input_Path_Folders:
    print('Accuracy on dataset in:' + single_input)
    [feature_dataset_full,labels]=extract_dataset(single_input)
    dataset_reduced=reducer.fit_transform(feature_dataset_full,labels)
    scaler= StandardScaler()
    dataset_reduced = scaler.fit_transform(dataset_reduced)
    predicted_label=clf.predict(dataset_reduced)
    print(get_accuracy(predicted_label,labels))



 ################# DAY 5 ####################################

classifier_folder= 'ModelDay5'
classifier_name='Classifier_Day5'
reducer_name='reducer_Day5'

#Select the folders to test the model
Input_Path_Folders=['./DataDay1','./DataDay2','./DataDay4','./DataDay5','./DataDay6']

old_path=os.getcwd()
os.chdir(classifier_folder)
clf = joblib.load(classifier_name+'.pkl') 
reducer=joblib.load(reducer_name+'.pkl') 
os.chdir(old_path)



print('Model trained: '+classifier_name)
for single_input in Input_Path_Folders:
    print('Accuracy on dataset in:' + single_input)
    [feature_dataset_full,labels]=extract_dataset(single_input)
    dataset_reduced=reducer.fit_transform(feature_dataset_full,labels)
    scaler= StandardScaler()
    dataset_reduced = scaler.fit_transform(dataset_reduced)
    predicted_label=clf.predict(dataset_reduced)
    print(get_accuracy(predicted_label,labels))

# INPUT PARAMETERS 


 ################# DAY 6 ####################################

classifier_folder= 'ModelDay6'
classifier_name='Classifier_Day6'
reducer_name='reducer_Day6'

#Select the input folders to test the modle
Input_Path_Folders=['./DataDay1','./DataDay2','./DataDay4','./DataDay5','./DataDay6']

old_path=os.getcwd()
os.chdir(classifier_folder)
clf = joblib.load(classifier_name+'.pkl') 
reducer=joblib.load(reducer_name+'.pkl') 
os.chdir(old_path)





print('Model trained: '+classifier_name)
for single_input in Input_Path_Folders:
    print('Accuracy on dataset in:' + single_input)
    [feature_dataset_full,labels]=extract_dataset(single_input)
    dataset_reduced=reducer.fit_transform(feature_dataset_full,labels)
    scaler= StandardScaler()
    dataset_reduced = scaler.fit_transform(dataset_reduced)
    predicted_label=clf.predict(dataset_reduced)
    print(get_accuracy(predicted_label,labels))
    
    
