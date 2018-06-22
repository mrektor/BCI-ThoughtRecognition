clear all
close all
clc
addpath(genpath('lib_feature_extraction'))
load EEGyes.mat
load ConnectivityParameters.mat
load rangeParameters.mat
load amplitudeParameters.mat
load('spectralParameters.mat')

%%
Fs=500;
nd=6;
for d=1:nd
    dir=strcat('DataDay', num2str(d));
    addpath(genpath(dir))
    load (fullfile(dir,'EEGyes.mat'))

%% connectivity
channel_labels={'FC5', 'FC6', 'C5', 'C6'};
for i=1:size(EEGyes,2)
        x=EEGyes{i}(:,1:4)';
        EEGyesBSI(i,:)=connectivity_features(x,Fs,'connectivity_BSI',ConnectivityParameters, channel_labels);
        EEGyesCoherenceMean(i,:)=connectivity_features(x,Fs,'connectivity_coh_mean',ConnectivityParameters, channel_labels);
        EEGyesCoherenceMax(i,:)=connectivity_features(x,Fs,'connectivity_coh_max',ConnectivityParameters, channel_labels);
        EEGyesCorrelation(i,:)=connectivity_features(x,Fs,'connectivity_corr',ConnectivityParameters, channel_labels);
  
end
clear x

%%

for i=1:size(EEGyes,2)
    for c=1:size(EEGyes{i},2)
        x=EEGyes{i}(:,c)';
        
%range
        EEGyesRangeMean(i,:,c)=rEEG(x,Fs,'rEEG_mean',rangeParameters);
        EEGyesRangeMedian(i,:,c)=rEEG(x,Fs,'rEEG_median',rangeParameters);
        EEGyesRangeLMargin(i,:,c)=rEEG(x,Fs,'rEEG_lower_margin',rangeParameters);
        EEGyesRangeUMargin(i,:,c)=rEEG(x,Fs,'rEEG_upper_margin',rangeParameters);
        EEGyesRangeWidth(i,:,c)=rEEG(x,Fs,'rEEG_width',rangeParameters);
        EEGyesRangeSD(i,:,c)=rEEG(x,Fs,'rEEG_SD',rangeParameters);
        EEGyesRangeCV(i,:,c)=rEEG(x,Fs,'rEEG_CV',rangeParameters);
        EEGyesRangeAssymetry(i,:,c)=rEEG(x,Fs,'rEEG_asymmetry',rangeParameters);


% amplitude

        EEGyesAmplitudePower(i,:,c)=amplitude_features(x,Fs,'amplitude_total_power', amplitudeParameters);
        EEGyesAplitudeSD(i,:,c)=amplitude_features(x,Fs,'amplitude_SD', amplitudeParameters);  
        EEGyesAmplitudeSkew(i,:,c)=amplitude_features(x,Fs,'amplitude_skew', amplitudeParameters); 
        EEGyesAmplitudeKurtosis(i,:,c)=amplitude_features(x,Fs,'amplitude_kurtosis', amplitudeParameters); 
        EEGyesAmplitudeEnvelopeMean(i,:,c)=amplitude_features(x,Fs,'amplitude_env_mean',amplitudeParameters);
        EEGyesAmplitudeEnvelopeSd(i,:,c)=amplitude_features(x,Fs,'amplitude_env_SD',amplitudeParameters);

%spectral

        EEGyesSpectralPower(i,:,c)=spectral_features(x,Fs,'spectral_power');
        EEGyesSpectralRelativePower(i,:,c)=spectral_features(x,Fs,'spectral_relative_power', spectralParameters);
        EEGyesSpectralFlatness(i,:,c)=spectral_features(x,Fs,'spectral_flatness', spectralParameters);
        EEGyesSpectralEntropy(i,:,c)=spectral_features(x,Fs,'spectral_entropy', spectralParameters);
        EEGyesSpectralDiff(i,:,c)=spectral_features(x,Fs,'spectral_edge_frequency', spectralParameters);
        EEGyesSpectralEdgeFreq(i,:,c)=spectral_features(x,Fs,'spectral_edge_frequency', spectralParameters);
    end
end
%%
%nelle colonne le diverse features, in riga i 5 canalix30 istanze
yesRangeFeatures=[ EEGyesRangeMean(:,:,1) EEGyesRangeMedian(:,:,1)  EEGyesRangeLMargin(:,:,1) EEGyesRangeUMargin(:,:,1) EEGyesRangeWidth(:,:,1) EEGyesRangeSD(:,:,1) EEGyesRangeCV(:,:,1)  EEGyesRangeAssymetry(:,:,1)];
for i=2:5
    yesRangeFeatures=[yesRangeFeatures;
        EEGyesRangeMean(:,:,i) EEGyesRangeMedian(:,:,i)  EEGyesRangeLMargin(:,:,i) EEGyesRangeUMargin(:,:,i) EEGyesRangeWidth(:,:,i) EEGyesRangeSD(:,:,i) EEGyesRangeCV(:,:,i)  EEGyesRangeAssymetry(:,:,i)];
end
yesAmplitudeFeatures=[EEGyesAmplitudePower(:,:,1) EEGyesAplitudeSD(:,:,1) EEGyesAmplitudeSkew(:,:,1) EEGyesAmplitudeKurtosis(:,:,1) EEGyesAmplitudeEnvelopeMean(:,:,1)  EEGyesAmplitudeEnvelopeSd(:,:,1)];
for i=2:5
    yesAmplitudeFeatures=[yesAmplitudeFeatures;
        EEGyesAmplitudePower(:,:,i) EEGyesAplitudeSD(:,:,i) EEGyesAmplitudeSkew(:,:,i) EEGyesAmplitudeKurtosis(:,:,i) EEGyesAmplitudeEnvelopeMean(:,:,i)  EEGyesAmplitudeEnvelopeSd(:,:,i)];
end

yesSpectralFeatures=[EEGyesSpectralPower(:,:,1) EEGyesSpectralRelativePower(:,:,1) EEGyesSpectralFlatness(:,:,1) EEGyesSpectralEntropy(:,:,1) EEGyesSpectralDiff(:,:,1) EEGyesSpectralEdgeFreq(:,:,1)];
for i=2:5
    yesSpectralFeatures=[yesSpectralFeatures;
    EEGyesSpectralPower(:,:,i) EEGyesSpectralRelativePower(:,:,i) EEGyesSpectralFlatness(:,:,i) EEGyesSpectralEntropy(:,:,i) EEGyesSpectralDiff(:,:,i) EEGyesSpectralEdgeFreq(:,:,i)];
end

ConnectivityFeaturesYes=[EEGyesBSI  EEGyesCoherenceMean EEGyesCoherenceMax EEGyesCorrelation];
%%

FeaturesYes=[yesRangeFeatures yesAmplitudeFeatures yesSpectralFeatures];
%%

save(fullfile(dir,'FeaturesYes.mat'), 'FeaturesYes')
save(fullfile(dir,'ConnectivityFeaturesYes.mat'), 'ConnectivityFeaturesYes')
clearvars -except amplitudeParameters ConnectivityParameters rangeParameters spectralParameters nd Fs
end
