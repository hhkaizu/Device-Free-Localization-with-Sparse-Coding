%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Single-target localization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% It shows a demo of the proposed Sparse Coding via Iterative Shrinkage-Thresholding Algorithm (SC-ISTA) 
% for single-target localization.
% The data of constructing the dictionary and the test signal are from the
% SPAN Lab of the University of Utah.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

%%
%%%%%%%%%%%%%%% Procedure of processing the dictionary data %%%%%%%%%%%%%
load matrix_dictionary.mat % Input the data of constructing dictionary 
% add noise to the dictionary
Dictionary_noise = awgn(matrix_dictionary,20,'measured'); % SNR=20dB. The modified parameters 
                                                          % are possibly required when SNR is low.

% Normalization of the noisy dictionary
for i=1:size(Dictionary_noise,2)
    Dictionary_noise(:,i)=(Dictionary_noise(:,i)-mean(Dictionary_noise(:,i)))...
        /std(Dictionary_noise(:,i));
end
Dictionary_noise_norm = Dictionary_noise*diag(1./sqrt(sum(Dictionary_noise.*...
    Dictionary_noise)));

%%%%%%%%%% Parameter initialization for ISTA algorithm %%%%%%%%%%
D = Dictionary_noise_norm;
[info.m,info.n]=size(D);
info.maxiter = 300;
info.tol=1e-1;
lambda = 0.7; %3e-4;%120,200% 0.005; %30,50% Regularization parameter limit
% initialize alpha
eigv=eig(D'*D);
info.alpha=max(eigv(:))*1.002;%max(max(eigv(:)),0.5);
% initialize H W t
info.t=lambda/info.alpha;info.W=D'/info.alpha;
info.H=eye(info.n)-D'*D/info.alpha;
info.Zchange=zeros(1,info.maxiter);

%%
%%%%%%%%%%%%%%%% Single-target localization %%%%%%%%%%%%%%%%%%%%%%%
load matrix_test.mat
index_of_RP = 1; % Reference Posiiton (RP) is the ground truth position. 
                 % The any other RPs in the monitoring area can be chose from 1 to 35.
fprintf('Ture position is at RP %d\n', index_of_RP);
test_matrix = matrix_test(:,(5*index_of_RP-1):5*index_of_RP); % Use 5 samples of each RP for target localization.
                                                              % Also, you can just choose one of the 5 samples for target localization.
% add noise to the test data
Data_noise = awgn(test_matrix,20,'measured'); % SNR=20dB. The modified parameters are possibly 
                                              % required when SNR is low.
Data_noise = mean(Data_noise,2);

% normalize the noisy test data
for h1=1:size(Data_noise,2)
    Data_noise(:,h1)=(Data_noise(:,h1)-mean(Data_noise(:,h1)))/std(Data_noise(:,h1));
end
Data_noise_norm=Data_noise*diag(1./sqrt(sum(Data_noise.*Data_noise)));

%%%%%%%%%%%%%%%%%%%%%%%%  Sparse coding stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_res, ~, info]=ISTAl_residChange(Dictionary_noise_norm, Data_noise_norm, lambda, info);
res = abs(x_res(:,:,end)); % Sparse solution

%%%%%%% The modified decision rule %%%%%%%
for i_no = 1:1:35
    pos_res (i_no, :) = sum (res((i_no*25-24):i_no*25,:));
end
[~, pos_sort] = max(pos_res);
fprintf('Estimated location is at RP %d\n', pos_sort); % Show the localizaiton result 

%%% Plot the modified sparse solution to check the localization result %%%
figure; 
h1=stem (pos_res);
h1=legend('Estamited result of the proposed SC-ISTA','location', 'best');
h1=xlabel('Index of reference position'); 
h1=ylabel('Estimated strength of targets');
text(6,(0.9*max(pos_res)),'Ground-truth location of the target: RP-1');
