clear all
close all
clc
load('EEGyes.mat')

addpath('/Users/giuliacorniani/Desktop/Universita/BCI/data_students/qEEG_feature_set-master/IBI_features');
addpath('/Users/giuliacorniani/Desktop/Università/BCI/data_students/qEEG_feature_set-master/utils')
%
Fs=500;
for i=1:size(EEGyes,2)
    for c=1:size(EEGyes{i},2)
     x=EEGyes{i}(:,c)';
        EEGyesIBIlengthMaxr{i}(:,c)=IBI_features(x,Fs,'IBI_length_max');
    end
end
