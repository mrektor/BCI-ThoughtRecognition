
 
fc=1/0.128040973111396; % NIRS

% NIRSdxy_no_signal=load('NIRSdxy_no_signal.mat');
% NIRSdxy_yes_signal=load('NIRSdxy_yes_signal.mat');
% 
NIRSoxy_no_signal=load('./segnali_belli/NIRSoxy_noFasato.mat');
NIRSoxy_yes_signal=load('./segnali_belli/NIRSoxy_yesFasato.mat');

%EEG_no_signal=load('Good_signals/EEGno_signal.mat');
%EEG_yes_signal=load('Good_signals/EEGyes_signal.mat');

% pick_cell=peak_detect(NIRSdxy_no_signal.no_signal,fc,'pick_NIRSdxy_no_signal');
% pick_cell=peak_detect(NIRSdxy_yes_signal.yes_signal,fc,'pick_NIRSdxy_yes_signal');
pick_cell=peak_detect(NIRSoxy_no_signal.nofasato,fc,'peak_NIRSoxy_no_signal');
pick_cell=peak_detect(NIRSoxy_yes_signal.yesfasato,fc,'peak_NIRSoxy_yes_signal');
%pick_cell=peak_detect(EEG_no_signal.no_signal,fc,'pick_EEG_no_signal');
%pick_cell=peak_detect(EEG_yes_signal.yes_signal,fc,'pick_EEG_yes_signal');


function [f_cell,coeff_cell]=peak_detect(signal,fc,name)
size_=size(signal);

cont=1;
for i=1:size_(2)
   single_slot=signal{1, i};
   channel_number=size(single_slot);
   for j=1:channel_number(2)
      string=[name,num2str(cont),'.png'];
      disp(string)
      [f_pick,coeff_pick]=find_peak(single_slot(:,j),fc,3,string);
      hold off;
      f_cell{cont}=f_pick;
      coeff_cell{cont}=coeff_pick;
      cont=cont+1;
   end
end
save([name,'.mat'],'f_cell','coeff_cell');
end


function [f_pick,coeff_pick]=find_peak(signal,fc,number_of_pick,string)

f=linspace(0, fc/2, floor(length(signal)/2));
spectrum= abs(fft(signal.^2));
spectrum=spectrum(1:floor(length(spectrum)/2));
[pks,locs]= findpeaks(spectrum);

[B,I] = sort(pks,'descend');

if(length(I)<number_of_pick)
    indexes=locs(I(1:length(I)));
else
indexes=locs(I(1:number_of_pick));
end

fft_pick_frequencies=f(indexes);
fft_pick_coefficents=spectrum(indexes);
% 
% h = figure;
% set(h, 'Visible', 'off');
% plot(f,spectrum)
% hold on
% plot(fft_pick_frequencies, fft_pick_coefficents, '*')        
% 
% saveas(h,string);


if(length(fft_pick_frequencies)<number_of_pick)
    temp=zeros(1,number_of_pick);
    temp(1:length(fft_pick_frequencies))=fft_pick_frequencies ;
    f_pick=temp;
    temp=zeros(1,number_of_pick);
    temp(1:length(fft_pick_frequencies))=fft_pick_coefficents ;
    coeff_pick=temp;
   
else
    f_pick=fft_pick_frequencies;
    coeff_pick=fft_pick_coefficents;
end

end
