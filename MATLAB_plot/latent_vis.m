% visualize latent space
set(0,'DefaultAxesFontSize',10);
%% read data
load('CVAE_pulse_phase_mixed_Poisson_johnathon_sample4_no_MSEY_5-100_zp.mat');
%% error bar plot for each dimension
figure;
mup_list=mup0(1:10,:);
sigmap_list=sigmap0(1:10,:);
errorbar(mup_list',sigmap_list','b.');
hold on
mup_list=mup0(101:110,:);
sigmap_list=sigmap0(101:110,:);
errorbar(mup_list',sigmap_list','r.');
hold on
mup_list=mup0(201:210,:);
sigmap_list=sigmap0(201:210,:);
errorbar(mup_list',sigmap_list','g.');
hold off
axis([-10,128,-10,17]);
set(gcf,'Position',[680 679 408 299]);

%% read data
load('CVAE_pulse_phase_mixed_Poisson_no_MSEY_5-110_zp.mat');
%% error bar plot for each dimension
figure;
mup_list=mup0(1:10,:);
sigmap_list=sigmap0(1:10,:);
errorbar(mup_list',sigmap_list','b.');
hold on
mup_list=mup0(101:110,:);
sigmap_list=sigmap0(101:110,:);
errorbar(mup_list',sigmap_list','r.');
hold on
mup_list=mup0(201:210,:);
sigmap_list=sigmap0(201:210,:);
errorbar(mup_list',sigmap_list','g.');
hold off
axis([-10,128,-10,17])
set(gcf,'Position',[680 679 408 299]);