clear all 
close all
clc
addpath(genpath('6DayDataset'))
%%
% the only parameters to modify if the dataset changes are nd, nb, the name of
% the input files and the folder where to save the data
% 
% the output are two file EEGyes.mat and EEGno.mat containing the signals
% corresponding to the yes/no answer and a file order.mat containing a vector of 0/1 representing the
% order of the answers
%%
disp('Filtering all the data'); 
nd=6; %number of days
nb=[4 4 2 4 3 4]; %number of blocks/day
order=[];
%%
for d=1:nd

    for b=1:nb(d)
        %% load data
        file=strcat('EEG_Day', num2str(d),'_Block', num2str(b), '_Training.mat');
        if exist(file)==0
            file=strcat('EEG_Day', num2str(d),'_Block', num2str(b), '_Feedback.mat');
        end
        load(file)
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
        
        filtered = filtfilt(sosbp, gbp, double(data));    % Filter Signal
        %% segmentation
        yes_start=find(eeg.onset==4);
        yes_end=find(eeg.onset==1);
        for k=1:length(yes_start)
            yes_signal{k}=filtered(yes_start(k):yes_end(k),:);
        end
        no_start=find(eeg.onset==8);
        no_end=find(eeg.onset==2);
        for k=1:length(no_start)
            no_signal{k}=filtered(no_start(k):no_end(k),:);
        end
        
        for i=1:10
            EEGyes{i+(b-1)*10}=yes_signal{i};
            EEGno{i+(b-1)*10}=no_signal{i};
        end
        
        %% 
        ord=[yes_start no_start; ones(length(yes_start),1)' zeros(length(no_start),1)' ]'; %yes=1 no=0
        ord=sortrows(ord, 1);
        ord=ord(:,2);
        order=[ord; order];
        
    end
    
    savdir = strcat('DataDay', num2str(d));   %name of the folder where save the files
    save(fullfile(savdir,'EEGyes.mat'),'EEGyes');
    save(fullfile(savdir,'EEGno.mat'),'EEGno');
    save(fullfile(savdir,'order.mat'),'order');
    clear EEGyes
    clear EEGno
    clear no_signal
    clear yes_signal
    clear order
    order=[];
end
