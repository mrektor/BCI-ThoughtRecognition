%%

vediamo = nirs.oxy

%%

plot(nirs.timePoints,vediamo(:,1)); hold on;
plot(nirs.timePoints,nirs.dxy(:,1))


%%
mean(nirs.timePoints,vediamo(:,1))
