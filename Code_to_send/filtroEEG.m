clear all
close all

load('EEG_Block1_Training.mat')
data=eeg.data(5:9,:)';
%% Block1
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
%figure(3)
%freqz(sosbp, 2^16, Fs)                                      % Filter Bode Plot
filtered = filtfilt(sosbp, gbp, double(data));    % Filter Signal

w=50/(500/2);
bw=w;
[num,den]=iirnotch(w,bw); % notch filter implementation 
for i=1:4
filtered(:,i)=filter(num,den,filtered(:,i));
end

% for i=1
%  plot(data(i:i+5000,2)) 
%  hold on
%  plot(filtered(i:i+5000,2),'k'),
%  hold off
%  pause(1)
%  legend('data', 'filtered', 'filteredHP')
% end

yes_start=find(eeg.onset==4);
yes_end=find(eeg.onset==1);
% figure
for k=1:length(yes_start)
    yes_signal1{k}=filtered(yes_start(k):yes_end(k),:);
    yes_signalpre{k}=data(yes_start(k):yes_end(k),:);  
   % subplot(5,2,k), plot(yes_signal1{k}(:,1), 'k'), suptitle('yes filtrato 1')
end  
% figure
% for k=1:length(yes_start)
% subplot(5,2,k), plot(yes_signalpre{k}(:,1)), suptitle('yes grezzo 1')
% end
%% no
no_start=find(eeg.onset==8);
no_end=find(eeg.onset==2);
%figure
for k=1:length(no_start)
    no_signal1{k}=filtered(no_start(k):no_end(k),:);
    no_signalpre{k}=data(no_start(k):no_end(k),:);
    %subplot(5,2,k), plot(no_signal1{k}(:,1), 'k'), suptitle('no filtrato 1')
end  
% figure
% for k=1:length(no_start)
% subplot(5,2,k), plot(no_signalpre{k}(:,1)), suptitle('no grezzo 1')
% end
%%
% figure
% plot(data(:,1))
% hold on
% plot(filtered(:,1))
%%
save('block1.mat', 'no_signal1', 'yes_signal1')
clear all
close all
%% block 2
close all

load('EEG_Block2_Feedback.mat')
data=eeg.data(5:9,:)';
%% 
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
% figure(3)
% freqz(sosbp, 2^16, Fs)                                      % Filter Bode Plot
filtered = filtfilt(sosbp, gbp, double(data));    % Filter Signal

w=50/(500/2);
bw=w;
[num,den]=iirnotch(w,bw); % notch filter implementation 
for i=1:4
filtered(:,i)=filter(num,den,filtered(:,i));
end

% for i=1
%  plot(data(i:i+5000,2)) 
%  hold on
%  plot(filtered(i:i+5000,2),'k'),
%  hold off
%  pause(1)
%  legend('data', 'filtered', 'filteredHP')
% end

yes_start=find(eeg.onset==4);
yes_end=find(eeg.onset==1);
% figure
for k=1:length(yes_start)
    yes_signal2{k}=filtered(yes_start(k):yes_end(k),:);
    yes_signalpre{k}=data(yes_start(k):yes_end(k),:);  
%     subplot(5,2,k), plot(yes_signal2{k}(:,1), 'k'), suptitle('yes filtrato 1')
end  
% figure
% for k=1:length(yes_start)
% subplot(5,2,k), plot(yes_signalpre{k}(:,1)), suptitle('yes grezzo 1')
% end
%% no
no_start=find(eeg.onset==8);
no_end=find(eeg.onset==2);
% figure
for k=1:length(no_start)
    no_signal2{k}=filtered(no_start(k):no_end(k),:);
    no_signalpre{k}=data(no_start(k):no_end(k),:);
%     subplot(5,2,k), plot(no_signal2{k}(:,1), 'k'), suptitle('no filtrato 1')
end  
% figure
% for k=1:length(no_start)
% subplot(5,2,k), plot(no_signalpre{k}(:,1)), suptitle('no grezzo 1')
% end
%%
% figure
% plot(data(:,1))
% hold on
% plot(filtered(:,1))

%% block 3
save('block2.mat', 'no_signal2', 'yes_signal2')
clear all
close all

load('EEG_Block3_Feedback.mat')
data=eeg.data(5:9,:)';
%% 
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
% figure(3)
% freqz(sosbp, 2^16, Fs)                                      % Filter Bode Plot
filtered = filtfilt(sosbp, gbp, double(data));    % Filter Signal

w=50/(500/2);
bw=w;
[num,den]=iirnotch(w,bw); % notch filter implementation 
for i=1:4
filtered(:,i)=filter(num,den,filtered(:,i));
end

% for i=1
%  plot(data(i:i+5000,2)) 
%  hold on
%  plot(filtered(i:i+5000,2),'k'),
%  hold off
%  pause(1)
%  legend('data', 'filtered', 'filteredHP')
% end

yes_start=find(eeg.onset==4);
yes_end=find(eeg.onset==1);
% figure
for k=1:length(yes_start)
    yes_signal3{k}=filtered(yes_start(k):yes_end(k),:);
    yes_signalpre{k}=data(yes_start(k):yes_end(k),:);  
%     subplot(5,2,k), plot(yes_signal3{k}(:,1), 'k'), suptitle('yes filtrato 1')
end  
% figure
% for k=1:length(yes_start)
% subplot(5,2,k), plot(yes_signalpre{k}(:,1)), suptitle('yes grezzo 1')
% end
%% no
no_start=find(eeg.onset==8);
no_end=find(eeg.onset==2);
% figure
for k=1:length(no_start)
    no_signal3{k}=filtered(no_start(k):no_end(k),:);
    no_signalpre{k}=data(no_start(k):no_end(k),:);
%     subplot(5,2,k), plot(no_signal3{k}(:,1), 'k'), suptitle('no filtrato 1')
end  
% figure
% for k=1:length(no_start)
% subplot(5,2,k), plot(no_signalpre{k}(:,1)), suptitle('no grezzo 1')
% end
%%
% figure
% plot(data(:,1))
% hold on
% plot(filtered(:,1))
%%
save('block3.mat', 'no_signal3', 'yes_signal3')
clear all

%% save
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
