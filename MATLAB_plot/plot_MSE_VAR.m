set(0,'DefaultAxesFontSize',14);
% load('CVAE_pulse_phase_mixed_Poisson_johnathon_1_statistics.mat');
load('CVAE_pulse_phase_mixed_Poisson_johnathon_sample4_no_MSEY_5-100_statistics.mat');
% load('../Pulse_retrieval/dataset/XUV_IR_test_Poisson_johnathon_sample4_mixed_5-100_myA.mat')
load('sample4_plot.mat');
delay=delay*1e15;
% load('CVAE_pulse_phase_Poisson_100counts_statistics.mat');
counts=[5,7,10,15,21,32.5,43,55,77.5,100];
% view sample reconstruction
inds_plot=[1,3,5,7,9];
inds=[2,4,6,8,10];
% inds2=[12,14,18,19]; % locate ambiguity copies
% plot the figure showing real and imaginary parts
figure;
imagesc(delay,energy,squeeze(mean(I_recon_group(:,161,:,:),1)));
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);
title(['MSE=',num2str(mean(MSE_trace(:,161),1))])
figure;
imagesc(delay,energy,squeeze(mean(I_recon_group(:,165,:,:),1)));
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);
title(['MSE=',num2str(mean(MSE_trace(:,165),1))])
figure;
imagesc(delay,energy,squeeze(mean(I_recon_group(:,170,:,:),1)));
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);
title(['MSE=',num2str(mean(MSE_trace(:,170),1))])
figure;
imagesc(delay,energy,squeeze(I_test(161,:,:)));
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);
figure;
imagesc(delay,energy,squeeze(I_test(165,:,:)));
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);
figure;
imagesc(delay,energy,squeeze(I_test(170,:,:)));
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);
figure;
imagesc(delay,energy,squeeze(I_ideal_group(1,161,:,:)));
colormap jet;axis tight;
set(gcf,'Position',[680 679 408 299]);

% MSE, VAR and bias of straking trace
MSE_avg_plot=mean(MSE_trace,1);%./mean(mean(mean(I_ideal_group,1).^2,3),4);
MSE_avg_plot=reshape(MSE_avg_plot,10,[]);
MSE_avg_plot=mean(MSE_avg_plot,2);
MSE_std_plot=std(MSE_trace,1);%./mean(mean(mean(I_ideal_group,1).^2,3),4);
MSE_std_plot=reshape(MSE_std_plot,10,[]);
MSE_std_plot=mean(MSE_std_plot,2);
var_plot=squeeze(mean(mean(var(I_recon_group,1),3),4));%./mean(mean(mean(I_ideal_group,1).^2,3),4);
var_plot=reshape(var_plot,10,[]);
var_plot=mean(var_plot,2);
bias_plot=squeeze(mean(mean((mean(I_recon_group,1)-mean(I_ideal_group,1)).^2,3),4));%./mean(mean(mean(I_ideal_group,1).^2,3),4);
bias_plot=reshape(bias_plot,10,[]);
bias_plot=mean(bias_plot,2);
figure;
plot(counts,MSE_avg_plot,'ko-','MarkerSize',4,'LineWidth',1.5);
hold on;
errorbar(counts,MSE_avg_plot,MSE_std_plot/2,'ko','MarkerSize',4);
% hold on;
% plot(counts,var_plot,'ro-','MarkerSize',4,'LineWidth',1.5);
% hold on;
% plot(counts,bias_plot,'bo-','MarkerSize',4,'LineWidth',1.5);
hold off;
set(gcf,'Position',[680 679 408 299]);

% MSE, var and bias of pulse duration
MSE_fwhm=mean((fwhm-fwhm_true).^2,1)./mean(fwhm_true,1).^2;
MSE_fwhm=reshape(MSE_fwhm,10,[]);
MSE_fwhm=mean(MSE_fwhm,2);
MSE_fwhm_std=std((fwhm-fwhm_true).^2,1)./mean(fwhm_true,1).^2;
MSE_fwhm_std=reshape(MSE_fwhm_std,10,[]);
MSE_fwhm_std=mean(MSE_fwhm_std,2);
bias_fwhm=(mean(fwhm,1)-mean(fwhm_true,1)).^2./mean(fwhm_true,1).^2;
bias_fwhm=reshape(bias_fwhm,10,[]);
bias_fwhm=mean(bias_fwhm,2);
var_fwhm=var(fwhm,1)./mean(fwhm_true,1).^2;
var_fwhm=reshape(var_fwhm,10,[]);
var_fwhm=mean(var_fwhm,2);
figure;
plot(counts,MSE_fwhm,'ko-','MarkerSize',4,'LineWidth',1.5);
hold on;
errorbar(counts,MSE_fwhm,MSE_fwhm_std/2,'ko');
% hold on;
% plot(counts,var_fwhm,'ro-','MarkerSize',4,'LineWidth',1.5);
% hold on;
% plot(counts,bias_fwhm,'bo-','MarkerSize',4,'LineWidth',1.5);
hold off;
set(gcf,'Position',[680 679 408 299]);

%% plot individual xuv time and freq domain pulses
ind_xuv_plot=170;
xuv_Et_recon_plot=squeeze(xuv_Et_recon(:,ind_xuv_plot,:));
Et_mean=mean(abs(xuv_Et_recon_plot),1);
scale=max(Et_mean);
Et_mean=Et_mean/scale;
ind_t=1:100:2048;
Et_std=std(abs(xuv_Et_recon_plot(:,ind_t)),1)/scale;
figure;
plot(tmat,Et_mean,'k','LineWidth',1.5);
hold on
errorbar(tmat(ind_t),Et_mean(ind_t),Et_std,'k.','LineWidth',1.5);
hold on;
plot(tmat,abs(xuv_Et_test(ind_xuv_plot,:))/max(abs(xuv_Et_test(ind_xuv_plot,:))),'r--','LineWidth',1.5)
hold off;
axis([-60 60 0 1.1]);
set(gcf,'Position',[680 679 408 299]);

xuv_Ef_recon_plot=squeeze(xuv_Ef_recon(:,ind_xuv_plot,:));
Ef_abs_mean=mean(abs(xuv_Ef_recon_plot),1);
Ef_angle=unwrap(angle(xuv_Ef_recon_plot),[],2);
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
hold on
xuv_Ef_test_angle=unwrap(angle(xuv_Ef_test(ind_xuv_plot,:)));
plot(fmat_ev_cropped,xuv_Ef_test_angle-min(xuv_Ef_test_angle),'r--','LineWidth',1.5)
hold off;
axis([70 310 -inf inf]);
set(gcf,'Position',[680 679 408 299]);

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