%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Multi-target localization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%It shows a demo of the proposed Sparse Coding via Iterative Shrinkage-Thresholding Algorithm (SC-ISTA) 
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
Dictionary = matrix_dictionary; 
% Normalization of the noisy dictionary
for i=1:size(Dictionary,2)
    Dictionary(:,i)=(Dictionary(:,i)-mean(Dictionary(:,i)))...
        /std(Dictionary(:,i));
end
Dictionary_norm = Dictionary*diag(1./sqrt(sum(Dictionary.*...
    Dictionary)));

%%%%%%%%%%% pre-calculate some parameters for ISTA %%%%%%%%%%%
    D = Dictionary_norm;
    [info.m,info.n]=size(D);
    info.maxiter = 300;
    info.tol=1e-1;
    lambda = 0.91; %3e-4;%120,200% 0.005; %30,50% Regularization parameter limit
    % initialize alpha
    eigv=eig(D'*D);
    info.alpha=max(eigv(:))*1.02;%max(max(eigv(:)),0.5);
    % initialize H W t
    info.t=lambda/info.alpha;info.W=D'/info.alpha;
    info.H=eye(info.n)-D'*D/info.alpha;
    info.Zchange=zeros(1,info.maxiter);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Procedure of processing the test data %%%%%%%%%%%%%
load TestSignalForTwoTargets.mat % load the test data
CasesFor2tars = [25, 24; 26, 24; 27, 24; 28, 24; 29, 24; 22, 24]; % The six cases of the ground-truth locations of the two targets
index_of_cases = 1; % Here, we take one case as an example. The other cases are also optional.
fprintf('The ground-truth locations of 2 targets: RP-%d, RP-%d\n', ...
    CasesFor2tars(index_of_cases,1), CasesFor2tars(index_of_cases,2));
test_2tars = Matrix_2targets_tol (: , index_of_cases);

Data =   test_2tars;
% Normalization of Data_noise 
for h1=1:size(Data,2)
    Data(:,h1)=(Data(:,h1)-mean(Data(:,h1)))/std(Data(:,h1));
end
Data_norm=Data*diag(1./sqrt(sum(Data.*Data)));

%%%%%%%%%%%%%%%%%%%%%%%%  Sparse coding stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [x_res1, ~, info]=ISTAl_residChange(Dictionary_norm, Data_norm, lambda, info);
            res = abs(x_res1(:,:,end)); % Sparse solution

%%%%%%% The modified decision rule %%%%%%%
for i_no = 1:1:35
    pos_res (i_no, :) = sum (res((i_no*25-24):i_no*25,:));
end
[~, pos_sort1] = sort(pos_res,'descend');
[est_locations, ~] = sort(pos_sort1(1:2),'descend');

fprintf('Estimated locations of 2 targets: RP-%d, RP-%d\n', ...
    est_locations(1), est_locations(2));

figure;

h1=stem (pos_res);
h1=legend('Estamited result of the proposed SC-ISTA','location', 'northwest');
h1=xlabel('Index of reference-position (RP)');
h1=ylabel('Estimated strength of targets');
text(1,(0.9*max(pos_res)),'Ground-truth locations of 2 targets:RP-25, RP-24');
