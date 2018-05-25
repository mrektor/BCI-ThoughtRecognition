clear all
close all

%% filtering  EEG row signal
run('filtroEEG.m')

%% extracting features without windowing
run('EEGyesAllFeatures.m')
run('EEGnoAllFeatures.m')

%% extracting features with windowing

run('windowNoFeatures.m')
run('windowYesFeatures.m')