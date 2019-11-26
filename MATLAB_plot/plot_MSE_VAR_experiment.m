set(0,'DefaultAxesFontSize',14);
% load('CVAE_pulse_phase_mixed_Poisson_johnathon_1_statistics.mat');
% load('CVAE_pulse_phase_mixed_Poisson_johnathon_sample4_no_MSEY_10-100_experiment_statistics.mat');
% load('CVAE_pulse_phase_Poisson_100counts_statistics.mat');
load('sample4_plot.mat');
delay=delay*1e15;
% view sample reconstruction
inds_plot=[1];
inds=[1,3,5,7,9];
% inds2=[12,14,18,19]; % locate ambiguity copies
% plot the figure showing real and imaginary parts
figure;
imagesc(delay,energy,squeeze(I_recon_group(inds(3),inds_plot(1),:,:)))
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);
figure;
imagesc(delay,energy,squeeze(I_ideal_group(1,1,:,:)));
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);

figure;
plot(tmat,abs(xuv_Et_recon(inds(1),:)),'b');
hold on;
plot(tmat,abs(xuv_Et_recon(inds(3),:)),'r');
hold on;
plot(tmat,abs(xuv_Et_recon(inds(5),:)),'g');
hold off;
set(gcf,'Position',[680 679 408 299]);

Et_mean=mean(abs(xuv_Et_recon),1);
scale=max(Et_mean);
Et_mean=Et_mean/scale;
ind_t=1:100:2048;
Et_std=std(abs(xuv_Et_recon(:,ind_t)),1)/scale;
figure;
plot(tmat,Et_mean,'k','LineWidth',1.5);
hold on
errorbar(tmat(ind_t),Et_mean(ind_t),Et_std,'k.','LineWidth',1.5);
hold off;
axis([-60 60 0 1.1]);
set(gcf,'Position',[680 679 408 299]);

Ef_abs_mean=mean(abs(xuv_Ef_recon),1);
Ef_angle=unwrap(angle(xuv_Ef_recon),[],2);
Ef_angle_mean=mean(Ef_angle,1);
Ef_angle_mean=Ef_angle_mean-min(Ef_angle_mean(:));
scale_angle=max(Ef_angle_mean(:));
ind_f=1:20:176;
Ef_angle_std=std(Ef_angle(:,ind_f),1);
figure;
yyaxis left;
plot(fmat_ev_cropped,Ef_abs_mean,'k','LineWidth',1.5);
hold on
yyaxis right;
plot(fmat_ev_cropped,Ef_angle_mean,'g','LineWidth',1.5);
hold on
yyaxis right;
errorbar(fmat_ev_cropped(ind_f),Ef_angle_mean(ind_f),Ef_angle_std,'g.','LineWidth',1.5);
hold off;
axis([70 310 -inf inf]);
set(gcf,'Position',[680 679 408 299]);

% MSE, VAR decomposition
MSE_avg_plot=mean(MSE_trace,1);
MSE_std_plot=std(MSE_trace,1);
var_plot=squeeze(mean(mean(var(I_recon_group,1),3),4));
bias_plot=squeeze(mean(mean((mean(I_recon_group,1)-mean(I_ideal_group,1)).^2,3),4));
% figure;
% plot(20:10:110,MSE_avg_plot,'k-');
% hold on;
% errorbar(20:10:110,MSE_avg_plot,MSE_std_plot,'ko');
% hold on;
% plot(20:10:110,var_plot,'ro-');
% hold on;
% plot(20:10:110,bias_plot,'bo-');
% hold off;
% set(gcf,'Position',[680 679 408 299]);

std_fwhm=std(fwhm,1);
mean_fwhm=mean(fwhm,1);
% figure;
% plot(20:10:110,MSE_fwhm,'k-');
% hold on;
% errorbar(20:10:110,MSE_fwhm,MSE_fwhm_std,'ko');
% hold on;
% plot(20:10:110,var_fwhm,'ro-');
% hold on;
% plot(20:10:110,bias_fwhm,'bo-');
% hold off;
% set(gcf,'Position',[680 679 408 299]);

%% load individual noise models
counts_list=[2,5,10,20:20:80];
MSE_avg_plot=zeros(1,length(counts_list));
MSE_std_plot=zeros(1,length(counts_list));
var_plot=zeros(1,length(counts_list));
bias_plot=zeros(1,length(counts_list));
MSE_fwhm=zeros(1,length(counts_list));
MSE_fwhm_std=zeros(1,length(counts_list));
for ind_counts=1:length(counts_list)
load(['CVAE_pulse_phase_Poisson_',num2str(counts_list(ind_counts)),'counts_statistics.mat']);
MSE_avg_plot(ind_counts)=mean(MSE_trace(:));
MSE_std_plot(ind_counts)=std(MSE_trace(:));
var_plot(ind_counts)=squeeze(mean(mean(var(reshape(I_recon_group,[],256,38)),2),3));
bias_plot(ind_counts)=squeeze(mean(mean((mean(reshape(I_recon_group,[],256,38),1)-mean(reshape(I_ideal_group,[],256,38),1)).^2,2),3));
MSE_fwhm(ind_counts)=mean((fwhm(:)-fwhm_true(:)).^2);
MSE_fwhm_std(ind_counts)=std((fwhm(:)-fwhm_true(:)).^2);
end
figure;
semilogy(counts_list,MSE_avg_plot,'k-');
hold on;
errorbar(counts_list,MSE_avg_plot,MSE_std_plot,'ko');
hold on;
plot(counts_list,var_plot,'ro-');
hold on;
plot(counts_list,bias_plot,'bo-');
hold off;