%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Single-target localization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%It shows a demo of the proposed Subspace-based Sparse Coding via Iterative Shrinkage-Thresholding Algorithm (SSC-ISTA) 
%for multi-target localization.
% The data of constructing the dictionary and the test signal are from the
% SPAN Lab of the University of Utah.
% Here, we take an example for single-target Localization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear;
%%%%%%%%%%%%%%% Procedure of processing the dictionary data %%%%%%%%%%%%%
load matrix_dictionary.mat % load the dictionary which is constructed only by the data of single-target locations
Dictionary = matrix_dictionary;
% add noise to the dictionary
Dictionary_noise = awgn(Dictionary,20,'measured'); % SNR=20dB. The modified parameters 
                                                          % are possibly required when SNR is low.
% Normalize the noisy dictionary
for i=1:size(Dictionary_noise,2)
    Dictionary_noise(:,i)=(Dictionary_noise(:,i)-mean(Dictionary_noise(:,i)))/std(Dictionary_noise(:,i));
end
Dictionary_noise_norm = Dictionary_noise*diag(1./sqrt(sum(Dictionary_noise.*Dictionary_noise)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% compute subspace matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
X = Dictionary_noise_norm; % just for convience
[m_oriDic, n_oriDic] = size(X);

%%%% 1) dimensionality reduction on the column-dimension %%%%%%%
Numof_RP = 35; % The total number of reference-positions (RPs) is 35
ni= n_oriDic / Numof_RP; % Sample number of each RP
B_dic = zeros(m_oriDic, Numof_RP ); % Parameter initialization

for i_cov = 1:1:Numof_RP
    
    Cov1_dic = 1 / ni * (X(:,(i_cov*ni-19):i_cov*ni) * X(:,(i_cov*ni-19):i_cov*ni)');
    [U1, ~, ~] = svd(Cov1_dic);
    B_dic(:, i_cov ) = U1(:, 1);

end
    fprintf('Column-dimension reduction is done.\n');
    
%%%% 2) dimensionality reduction on the row-dimension %%%%%%%
[m_B_dic, n_B_dic] = size(B_dic);
Cov_B_dic = 1 / n_B_dic * (B_dic * B_dic');  %%%%%%%%%this 1/m should be replaced by 1/n %%%%%%%%
[U_Cov_B_dic, S_Cov_B_dic, V_Cov_B_dic] = svd(Cov_B_dic);

S1_Cov_B_dic = S_Cov_B_dic * ones(size(S_Cov_B_dic,2),1);
RoC = cumsum(S1_Cov_B_dic)./sum(S1_Cov_B_dic); % RoC is short for the ratio of cumulative distribution; 
                                               % see the equation (26) of our manuscript. According our
                                               % experimental result, when the row-dimension k is 25,RoC 
                                               % can be reached to 99% which can meet the requirement of
                                               % 2-target localization.


k = 25; % It is the lowest row-dimensions that can reducedcan meet the requirement of 2-target localization.

% reduction
Uk_redn = U_Cov_B_dic(:, 1:k); % Uk_redn is the final subspace-matrix
Dic_reduction = Uk_redn' * B_dic;
fprintf('Row-dimension reduction is done.\n');

%%%%%%%%%%% pre-calculate some parameters for ISTA %%%%%%%%%%%
D = Dic_reduction;
[info.m,info.n]=size(D);
info.maxiter = 300;
info.tol=1e-1;
lambda = 0.82; % Good result when it is 0.5 for 2 targets localization; Regularization parameter limit
% initialize alpha
eigv=eig(D'*D);
info.alpha=max(eigv(:))*1.02;%max(max(eigv(:)),0.5);
% initialize H W t
info.t=lambda/info.alpha;info.W=D'/info.alpha;
info.H=eye(info.n)-D'*D/info.alpha;
info.Zchange=zeros(1,info.maxiter);
%%%%%%%
%%%%%%%%%%%%%%%% Test stage of single-target localization %%%%%%%%%%%%%%%%%%%%%%%
load matrix_test.mat
index_of_RP = 1; % Reference Posiiton (RP) is the ground truth position. 
                 % The any other RPs in the monitoring area can be chose from 1 to 35.
fprintf('Ground-ture position is at RP %d\n', index_of_RP);
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
Data_reduction =  Uk_redn' * Data_noise_norm;
%%%%%%%%%%%%%%%%%%%%%%%%  Sparse coding stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_res, ~, info]=ISTAl_residChange(Dic_reduction, Data_reduction, lambda, info);
pos_res = abs(x_res(:,:,end)); % Sparse solution

[~, pos_sort] = max(pos_res);
fprintf('Estimated location is at RP %d\n', pos_sort); % Show the localizaiton result 

%%% Plot the modified sparse solution to check the localization result %%%
figure; 
h1=stem (pos_res);
h1=legend('Estamited result of the proposed SSC-ISTA','location', 'best');
h1=xlabel('Index of reference position'); 
h1=ylabel('Estimated strength of targets');
text(6,(0.9*max(pos_res)),'Ground-truth location of the target: RP-1');
