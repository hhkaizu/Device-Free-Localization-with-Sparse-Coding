%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Comparison of efficiencies  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This demo shows the comparison of the proposed SC-ISTA Algorithm and the
% CVX tool proposed in the reference [19] with respect to the efficiency. 
% 
% If you would like to successfully run this demo, you have to install the
% CVX tool. The download link is as follows: http://cvxr.com/cvx/download/
% The following link shows a simple procedure of installing CVX tool: http://cvxr.com/cvx/doc/install.html
% Also, you can install it by the following two steps
% 1) Unpack the file 'cvx-w64.zip' anywhere you like; a directory called 'cvx' will be created.
% 2) Change directories to the top of the CVX distribution, and run the 'cvx_setup' command. 
%    For example, if you installed CVX into C\personal\cvx on Windows, type these commands:
%       cd C:\personal\cvx % Please change to your own installation path
%       cvx_setup
%%
%%%%%%%%%%%%%%% Install the CVX tool %%%%%%%%%%%%%%%%
% For your convenience, we auto-install the CVX tool here.
% Install the CVX tool %
cd cvx 
      cvx_setup
cd ..
%% Code begins here.
clear;
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
test_matrix = matrix_test(:,(5*index_of_RP-1):5*index_of_RP); % Use 5 samples of each RP for target localization.
                                                              % Also, you can just choose one of the 5 samples for target localization.
% add noise to the test data
Data_noise = awgn(test_matrix,20,'measured'); % SNR=20dB. The modified parameters are possibly required when SNR is low.                         
Data_noise = mean(Data_noise,2);

% normalize the noisy test data
for h1=1:size(Data_noise,2)
    Data_noise(:,h1)=(Data_noise(:,h1)-mean(Data_noise(:,h1)))/std(Data_noise(:,h1));
end
Data_noise_norm=Data_noise*diag(1./sqrt(sum(Data_noise.*Data_noise)));

%%%%%%%%%%%%%%%%%%%%%%%%  Efficiencies of two algorithms in sparse coding stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%  The CVX tool used in the reference [19] %%%%%%%
tic;
    cvx_begin
            variable x_bp(875*1)
            minimize (sum_square_abs(Data_noise_norm - Dictionary_noise_norm * x_bp) + 0.1 * norm(x_bp,1))
    cvx_end
Sparse_solution_CVX = abs(x_bp);
Timecost_CVX=toc;
fprintf('Sparse-coding time of the CVX tool is %d\n', Timecost_CVX); % 

%%%%%%%  The proposed SC-ISTA in our manuscript %%%%%%%
tic;
[x_res, ~, info]=ISTAl_residChange(Dictionary_noise_norm, Data_noise_norm, lambda, info);
Sparse_solution_SCISTA = abs(x_res(:,:,end)); % Sparse solution
Timecost_SCISTA=toc; % Sparse-coding time of the proposed SC-ISTA
fprintf('Sparse-coding time of the proposed SC-ISTA is %d\n', Timecost_SCISTA); % The timecost of SCISTA is much faster than that of CVX tool

timecost(1,1) = Timecost_CVX;
timecost(2,1) = Timecost_SCISTA;

% Figure for comparison
figure;clf; 

   b=stem(1,Timecost_CVX,'bo'); hold on
      b=stem(2,Timecost_SCISTA,'r*');
set(gca,'XTick',0:3);%设置要显示坐标刻度   grid on;
%    ch = get(b,'children');
%    set(gca,'XTickLabel',{'0'})

   legend('CVX tool in reference [19]','the proposed algorithm');

   xlabel('Two compared algorithms ');
   ylabel('Sparse-coding time');
