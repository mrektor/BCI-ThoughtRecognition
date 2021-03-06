clear all
close all
clc
warning off
%% BLOCK 1
load('EEG_Block1_Training.mat')
data=eeg.data(5:9,:)';
%% filter
data=detrend(data);
t=eeg.timePoints;
Nc=length(data');
Fs=1/(t(2)-t(1));                                           % Sampling Frequency (Hz)
Fn = Fs/2;                                                  % Nyquist Frequency (Hz)
Wp = [0.1 30]/Fn;                                           % Passband Frequency (Normalised)
Ws = [ 0.01 32]/Fn;                                         % Stopband Frequency (Normalised)
Rp =   1;                                                   % Passband Ripple (dB)
Rs = 150;                                                   % Stopband Ripple (dB)

[n,Ws] = cheb2ord(Wp,Ws,Rp,Rs);                             % Filter Order
[z,p,k] = cheby2(n,Rs,Ws);                                  % Filter Design
[sosbp,gbp] = zp2sos(z,p,k);                                % Convert To Second-Order-Section For Stability
freqz(sosbp, 2^16, Fs)                                      % Filter Bode Plot
title('Filter Bode Plot')

filtered = filtfilt(sosbp, gbp, double(data));    % Filter Signal
%% segmentation
yes_start=find(eeg.onset==4);
yes_end=find(eeg.onset==1);
for k=1:length(yes_start)
    yes_signal1{k}=filtered(yes_start(k):yes_end(k),:);
end  
no_start=find(eeg.onset==8);
no_end=find(eeg.onset==2);
for k=1:length(no_start)
    no_signal1{k}=filtered(no_start(k):no_end(k),:);
end  
%% save block 1
save('block1.mat', 'no_signal1', 'yes_signal1')
clear all
%% BLOCK 2
load('EEG_Block2_Feedback.mat')
data=eeg.data(5:9,:)';
%% filter
data=detrend(data);
t=eeg.timePoints;
Nc=length(data');
Fs=1/(t(2)-t(1));                                           % Sampling Frequency (Hz)
Fn = Fs/2;                                                  % Nyquist Frequency (Hz)
Wp = [0.1 30]/Fn;                                            % Passband Frequency (Normalised)
Ws = [ 0.01 32]/Fn;                                            % Stopband Frequency (Normalised)
Rp =   1;                                                   % Passband Ripple (dB)
Rs = 150;                                                   % Stopband Ripple (dB)

[n,Ws] = cheb2ord(Wp,Ws,Rp,Rs);                             % Filter Order
[z,p,k] = cheby2(n,Rs,Ws);                                  % Filter Design
[sosbp,gbp] = zp2sos(z,p,k);                                % Convert To Second-Order-Section For Stability


filtered = filtfilt(sosbp, gbp, double(data));    % Filter Signal
%% segmentation
yes_start=find(eeg.onset==4);
yes_end=find(eeg.onset==1);
for k=1:length(yes_start)
    yes_signal2{k}=filtered(yes_start(k):yes_end(k),:);
end  

no_start=find(eeg.onset==8);
no_end=find(eeg.onset==2);
for k=1:length(no_start)
    no_signal2{k}=filtered(no_start(k):no_end(k),:);
end  
%% save block 2
save('block2.mat', 'no_signal2', 'yes_signal2')
clear all
%% BLOCK 3
load('EEG_Block3_Feedback.mat')
data=eeg.data(5:9,:)';
%% filter
data=detrend(data);
t=eeg.timePoints;
Nc=length(data');
Fs=1/(t(2)-t(1));                                           % Sampling Frequency (Hz)
Fn = Fs/2;                                                  % Nyquist Frequency (Hz)
Wp = [0.1 30]/Fn;                                            % Passband Frequency (Normalised)
Ws = [ 0.01 32]/Fn;                                            % Stopband Frequency (Normalised)
Rp =   1;                                                   % Passband Ripple (dB)
Rs = 150;                                                   % Stopband Ripple (dB)

[n,Ws] = cheb2ord(Wp,Ws,Rp,Rs);                             % Filter Order
[z,p,k] = cheby2(n,Rs,Ws);                                  % Filter Design
[sosbp,gbp] = zp2sos(z,p,k);                                % Convert To Second-Order-Section For Stability


filtered = filtfilt(sosbp, gbp, double(data));    % Filter Signal
%% segmentation
yes_start=find(eeg.onset==4);
yes_end=find(eeg.onset==1);
for k=1:length(yes_start)
    yes_signal3{k}=filtered(yes_start(k):yes_end(k),:);
end 

no_start=find(eeg.onset==8);
no_end=find(eeg.onset==2);
for k=1:length(no_start)
    no_signal3{k}=filtered(no_start(k):no_end(k),:);
end  
%% save block 3
save('block3.mat', 'no_signal3', 'yes_signal3')
clear all

%% save all
load block1.mat
load block2.mat
load block3.mat

EEGyes=yes_signal1;
EEGno=no_signal1;
for i=1:10
EEGyes{i+10}=yes_signal2{i};
EEGno{i+10}=no_signal2{i};
end
for i=1:10
    EEGyes{i+20}=yes_signal3{i};
    EEGno{i+20}=no_signal3{i};
end

save('EEGyes.mat', 'EEGyes')
save('EEGno.mat', 'EEGno')
