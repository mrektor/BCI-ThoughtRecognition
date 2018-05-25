clear all
close all
clc
addpath(genpath('./lib_feature_extraction'))
load EEGno.mat
load ConnectivityParameters.mat
load rangeParameters.mat
load amplitudeParameters.mat
load('spectralParameters.mat')


%%
Fs=500;
%% windowing
window =1500;
slide=500;
for i=1:size(EEGno,2)
    x=EEGno{i}(:,1:4)';
    cont=1;
    for k=1:slide:4000
        windx{cont,i}(:,:)=x( :, k:k+window);
        cont=cont+1;
    end
end
for i=1:8*30
    flatten{i} = windx{i};
end
%% Connectivity
channel_labels={'FC5', 'FC6', 'C5', 'C6'};
for i=1:240
        x=flatten{i}(1:4,:);
        EEGnoBSI(i,:)=connectivity_features(x,Fs,'connectivity_BSI',ConnectivityParameters, channel_labels);
        EEGnoCoherenceMean(i,:)=connectivity_features(x,Fs,'connectivity_coh_mean',ConnectivityParameters, channel_labels);
        EEGnoCoherenceMax(i,:)=connectivity_features(x,Fs,'connectivity_coh_max',ConnectivityParameters, channel_labels);
        EEGnoCorrelation(i,:)=connectivity_features(x,Fs,'connectivity_corr',ConnectivityParameters, channel_labels);
  
end
%%
clear x
clear flatten
clear windx
%%
for i=1:size(EEGno,2)
    x=EEGno{i}(:,1:5)';
    cont=1;
    for k=1:slide:4000
        windx{cont,i}(:,:)=x( :, k:k+window);
        cont=cont+1;
    end
end
for i=1:8*30
    flatten{i} = windx{i};
end

%%

for i=1:size(flatten,2)
    for c=1:size(flatten{i},1)
        x=flatten{i}(c,:);
        
%range
        EEGnoRangeMean(i,:,c)=rEEG(x,Fs,'rEEG_mean',rangeParameters);
        EEGnoRangeMedian(i,:,c)=rEEG(x,Fs,'rEEG_median',rangeParameters);
        EEGnoRangeLMargin(i,:,c)=rEEG(x,Fs,'rEEG_lower_margin',rangeParameters);
        EEGnoRangeUMargin(i,:,c)=rEEG(x,Fs,'rEEG_upper_margin',rangeParameters);
        EEGnoRangeWidth(i,:,c)=rEEG(x,Fs,'rEEG_width',rangeParameters);
        EEGnoRangeSD(i,:,c)=rEEG(x,Fs,'rEEG_SD',rangeParameters);
        EEGnoRangeCV(i,:,c)=rEEG(x,Fs,'rEEG_CV',rangeParameters);
        EEGnoRangeAssymetry(i,:,c)=rEEG(x,Fs,'rEEG_asymmetry',rangeParameters);


% amplitude

        EEGnoAmplitudePower(i,:,c)=amplitude_features(x,Fs,'amplitude_total_power', amplitudeParameters);
        EEGnoAplitudeSD(i,:,c)=amplitude_features(x,Fs,'amplitude_SD', amplitudeParameters);  
        EEGnoAmplitudeSkew(i,:,c)=amplitude_features(x,Fs,'amplitude_skew', amplitudeParameters); 
        EEGnoAmplitudeKurtosis(i,:,c)=amplitude_features(x,Fs,'amplitude_kurtosis', amplitudeParameters); 
        EEGnoAmplitudeEnvelopeMean(i,:,c)=amplitude_features(x,Fs,'amplitude_env_mean',amplitudeParameters);
        EEGnoAmplitudeEnvelopeSd(i,:,c)=amplitude_features(x,Fs,'amplitude_env_SD',amplitudeParameters);

%spectral

        EEGnoSpectralPower(i,:,c)=spectral_features(x,Fs,'spectral_power');
        EEGnoSpectralRelativePower(i,:,c)=spectral_features(x,Fs,'spectral_relative_power', spectralParameters);
        EEGnoSpectralFlatness(i,:,c)=spectral_features(x,Fs,'spectral_flatness', spectralParameters);
        EEGnoSpectralEntropy(i,:,c)=spectral_features(x,Fs,'spectral_entropy', spectralParameters);
        EEGnoSpectralDiff(i,:,c)=spectral_features(x,Fs,'spectral_edge_frequency', spectralParameters);
        EEGnoSpectralEdgeFreq(i,:,c)=spectral_features(x,Fs,'spectral_edge_frequency', spectralParameters);
    end
end
%%
%nelle colonne le diverse features, in riga i 5 canalix30 istanze
noRangeFeatures=[ EEGnoRangeMean(:,:,1) EEGnoRangeMedian(:,:,1)  EEGnoRangeLMargin(:,:,1) EEGnoRangeUMargin(:,:,1) EEGnoRangeWidth(:,:,1) EEGnoRangeSD(:,:,1) EEGnoRangeCV(:,:,1)  EEGnoRangeAssymetry(:,:,1)];
for i=2:5
    noRangeFeatures=[noRangeFeatures;
        EEGnoRangeMean(:,:,i) EEGnoRangeMedian(:,:,i)  EEGnoRangeLMargin(:,:,i) EEGnoRangeUMargin(:,:,i) EEGnoRangeWidth(:,:,i) EEGnoRangeSD(:,:,i) EEGnoRangeCV(:,:,i)  EEGnoRangeAssymetry(:,:,i)];
end
noAmplitudeFeatures=[EEGnoAmplitudePower(:,:,1) EEGnoAplitudeSD(:,:,1) EEGnoAmplitudeSkew(:,:,1) EEGnoAmplitudeKurtosis(:,:,1) EEGnoAmplitudeEnvelopeMean(:,:,1)  EEGnoAmplitudeEnvelopeSd(:,:,1)];
for i=2:5
    noAmplitudeFeatures=[noAmplitudeFeatures;
        EEGnoAmplitudePower(:,:,i) EEGnoAplitudeSD(:,:,i) EEGnoAmplitudeSkew(:,:,i) EEGnoAmplitudeKurtosis(:,:,i) EEGnoAmplitudeEnvelopeMean(:,:,i)  EEGnoAmplitudeEnvelopeSd(:,:,i)];
end

noSpectralFeatures=[EEGnoSpectralPower(:,:,1) EEGnoSpectralRelativePower(:,:,1) EEGnoSpectralFlatness(:,:,1) EEGnoSpectralEntropy(:,:,1) EEGnoSpectralDiff(:,:,1) EEGnoSpectralEdgeFreq(:,:,1)];
for i=2:5
    noSpectralFeatures=[noSpectralFeatures;
    EEGnoSpectralPower(:,:,i) EEGnoSpectralRelativePower(:,:,i) EEGnoSpectralFlatness(:,:,i) EEGnoSpectralEntropy(:,:,i) EEGnoSpectralDiff(:,:,i) EEGnoSpectralEdgeFreq(:,:,i)];
end

%%
WindowConnectivityFeaturesNo=[EEGnoBSI  EEGnoCoherenceMean EEGnoCoherenceMax EEGnoCorrelation];

WindowFeaturesNo=[noRangeFeatures noAmplitudeFeatures noSpectralFeatures];

save('WindowFeaturesNo.mat', 'WindowFeaturesNo')
save('WindowConnectivityFeaturesNo.mat', 'WindowConnectivityFeaturesNo')