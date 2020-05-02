%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Multi-target localization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%It shows a demo of the proposed Subspace-based Sparse Coding via Iterative Shrinkage-Thresholding Algorithm (SSC-ISTA) 
%for multi-target localization.
% The data of constructing the dictionary and the test signal are from the
% SPAN Lab of the University of Utah.
% Here, we take an example for 2-target Localization. The ground-truth locations of the two targets are
% in the following six cases:
% 1-st case: one target is at RP-25; another target is at RP-24
% 2-nd case: one target is at RP-26; another target is at RP-24
% 3-rd case: one target is at RP-27; another target is at RP-24
% 4-th case: one target is at RP-26; another target is at RP-24
% 5-th case: one target is at RP-29; another target is at RP-24
% 6-th case: one target is at RP-22; another target is at RP-24
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear;
%%%%%%%%%%%%%%% Procedure of processing the dictionary data %%%%%%%%%%%%%
load matrix_dictionary.mat % load the dictionary which is constructed only by the data of single-target locations
% Normalize the Dictionary
Dictionary=matrix_dictionary;
for i=1:size(Dictionary,2)
    Dictionary(:,i)=(Dictionary(:,i)-mean(Dictionary(:,i)))/std(Dictionary(:,i));
end
Dictionary_norm = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% compute subspace matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
X = Dictionary_norm; % just for convience
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
lambda = 0.91; % Good result when it is 0.5 for 2 targets localization; Regularization parameter limit
% initialize alpha
eigv=eig(D'*D);
info.alpha=max(eigv(:))*1.02;%max(max(eigv(:)),0.5);
% initialize H W t
info.t=lambda/info.alpha;info.W=D'/info.alpha;
info.H=eye(info.n)-D'*D/info.alpha;
info.Zchange=zeros(1,info.maxiter);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% Procedure of processing the test data %%%%%%%%
load TestSignalForTwoTargets.mat % load the test data
CasesFor2tars = [25, 24; 26, 24; 27, 24; 28, 24; 29, 24; 22, 24]; % The six cases of the ground-truth locations of the two targets
index_of_cases = 1; % Here, we take one case as an example. The other cases are also optional.
fprintf('The ground-truth locations of 2 targets: RP-%d, RP-%d\n', ...
    CasesFor2tars(index_of_cases,1), CasesFor2tars(index_of_cases,2));
test_2tars = Matrix_2targets_tol (: , index_of_cases);

Data = test_2tars;
% Normalization of Data_noise 
for h1=1:size(Data,2)
    Data(:,h1)=(Data(:,h1)-mean(Data(:,h1)))/std(Data(:,h1));
end
Data_norm=Data*diag(1./sqrt(sum(Data.*Data)));
Data_reduction =  Uk_redn' * Data_norm;
%%%%%%%%%%%%%%%%%%%%%%%%  Sparse coding stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [x_res1, ~, info]=ISTAl_residChange(Dic_reduction, Data_reduction, lambda, info);
            pos_res = abs(x_res1(:,:,end)); % Sparse solution

[~, pos_sort1] = sort(pos_res,'descend');
[est_locations, ~] = sort(pos_sort1(1:2),'descend');

fprintf('Estimated locations of 2 targets: RP-%d, RP-%d\n', ...
    est_locations(1), est_locations(2));

figure;

h1=stem (pos_res);
h1=legend('Estamited result of the proposed SSC-ISTA','location', 'northwest');
h1=xlabel('Index of reference-position (RP)');
h1=ylabel('Estimated strength of targets');
text(0.5,(0.9*max(pos_res)),'Ground-truth locations of 2 targets:RP-25, RP-24');