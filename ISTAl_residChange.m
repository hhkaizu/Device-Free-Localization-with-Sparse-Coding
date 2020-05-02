%ISTA - saving variables for LISTA
%Z 2017
function varargout = ISTAl_residChange(D, X, lambda, info)
% [info.m,info.n]=size(D);
[~,info.N]=size(X);
% initialize alpha
% eigv=eig(D'*D);
% info.alpha=max(eigv(:))*1.2;%max(max(eigv(:)),0.5);
% initialize H W t
b(:,:,1)=D'*X/info.alpha; 
% info.t=lambda/info.alpha;info.W=D'/info.alpha;
% info.H=eye(info.n)-D'*D/info.alpha;
Zout(:,:,1)=zeros(info.n,info.N);
% datafitting=zeros(1,info.maxiter);
% regularization=zeros(1,info.maxiter);
% avgdiversity_hoyer=zeros(1,info.maxiter);
% info.Zerr=zeros(1,info.maxiter);
% info.Zchange=zeros(1,info.maxiter);
for iter = 1:info.maxiter    
    %% -- Algorithm begins here --
    % FProp   
    % update Z proximal splitting
    [Zout(:,:,iter+1),~,~]=softthl1(b(:,:,iter),info.t);
    % update b
    b(:,:,iter+1)=b(:,:,iter)+info.H*(Zout(:,:,iter+1)-Zout(:,:,iter));
    % Calculate 
    info.Zchange(iter)=(norm(Zout(:,:,iter+1)-Zout(:,:,iter)))/norm(Zout(:,:,iter));
%     residual = D*Zout(:,:,iter+1) - X;
%     resid_change=sum((residual).^2);
%     if resid_change<info.residtol
%         break
%     end

%     info.resid = resid;
% %     info.residnorm = norm(resid)^2;
% %     residChange = D*Zout(:,:,iter+1) - X - (D*Zout(:,:,iter) - X);
%     datafitting(iter)=0.5*norm(resid,'fro')^2;%sum(sum(resid));
% %     regularization(iter)=sum(sum(abs(Zout(:,:,iter+1).^info.p)));
% 
% %     % Calculate numerosity/diversity
% %     diversity_hoyer = numerosity_hoyer(Zout(:,:,iter+1));
% %     diversity_hoyer(isnan(diversity_hoyer)) = 1; 
% %     avgdiversity_hoyer(iter) = mean(diversity_hoyer);
    if info.Zchange(iter)<  info.tol   %info.Zchangetol
        break
    end
%     if datafitting(iter)<info.tol
%         break
%     end
end
info.totaliter=iter;
% info.datafitting=datafitting(:,1:iter);
% regularization=regularization(:,1:iter);
% avgdiversity_hoyer=avgdiversity_hoyer(:,1:iter);
if nargout<=3
  varargout = {Zout, b, info};
else
  varargout = {Zout, b, info, datafitting, regularization, avgdiversity_hoyer};
end
% 