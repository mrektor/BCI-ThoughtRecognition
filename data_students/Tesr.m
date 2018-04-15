%%
figure()
for k=1:10

plot(mean(yes_signal{k},2),'--')
end
%%
test = yes_signal{1}

%%
yes_80 = yes_signal

%%
tot_yes = yes_80{1};
for k=2:length(yes_80)
    tot_yes = [tot_yes; yes_80{k}];
end


%%
[coeff,score,latent,tsquared,explained,mu] = pca(tot_yes)

bar(explained); grid on
figure();
bar(cumsum(explained)); grid minor