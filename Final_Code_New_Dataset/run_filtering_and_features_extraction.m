% run this code to process the all the data and extract the features

%filtering
run('filtroEEG.m');

%extract features

run('EEGyesAllFeatures.m');
run('EEGnoAllFeatures.m');
