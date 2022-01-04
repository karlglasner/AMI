
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ODE parameter estimation using Approximate Model Inference (AMI) method
% (see K. Glasner, Learning differential equations in the presence of data and model uncertainty, 2022)
%
% Inputs:
% data is a N x K array where K are the number of system components, N is the number of
% points in time series, evenly spaced with timestep dt  
%
% mask is a N x K array of 1 or 0.   If the data is defined sparsely by mask, only
% data values where mask = 1 are used.  
%
% range [tmin tmax] is the range of times, i.e dt = (tmax-tmin)/(N-1)  
%
% fode is a ode45-style function defined externally; it must be written using array operations
%
% theta0 is a 1 x P array with initial guesses for parameters 
%
% lambdafixed: if nonzero, estimation is done just for inner optimization using a fixed
% value of lambda; if zero, ignored
%
% Outputs:
%
% theta is a 1 x P array containing parameter estimates
% lambda is an estimate for the variance ratio parameter
% u is a N x K array, providing state estimate

function [theta,lambda,u] = ami(data,mask,range,fode,theta0,lambdafixed) 

global u
u = []; %zero out initially
UT = (data.*mask)';  %target data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% setup
z = size(UT); N = z(2);
dt = (range(2)-range(1))/(N-1);
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton',...
'MaxFunctionEvaluations',1e10,'FiniteDifferenceStepSize', 1e-05);
  
if (lambdafixed==0)  
  %%%% first obtain a crude estimate of mu using fixed lambda
  f = @(mu) minfun(mu,fode,UT,range,dt,mask',10);
  [parameters,fval,exitflag,output] = fminunc(f,[theta0 0],options);
  %  %%%%%%%%%%%%%%%%%%%%% now minimize with variable lambda
  %  masked region for initial guess is filled in using filldata and previous param. estimate
  UT = filldata(data, mask', range, fode, parameters(1:end-1))';
  u=[];  %reset intial state guess  
  f = @(mu) minfun(mu,fode,UT,range,dt,mask',0);
  [parameters,fval,exitflag,output] = fminunc(f,[parameters(1:3) 0],options);
else
  f = @(mu) minfun(mu,fode,UT,range,dt,mask',lambdafixed);
  [parameters,fval,exitflag,output] = fminunc(f,mu0,options);
end;
theta = parameters(1:end-1);
lambda= exp(parameters(end)/10)+0.0001; % decode lambda

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Inner optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function F = minfun(p,fode,UT,range,dt,mask,lambdafixed)

% minimize ||N(u,theta)||_2^2 + lambda||u-u_T||_2^2 over u, where N(u,theta) = du/dt -f(u;theta)
% uses LM optimization with numerical deriatives and adaptive step size

global u   % this will keep in memory to use as initial guess next time

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% setup parameters
if (lambdafixed >0) lambda =lambdafixed; else lambda = min(exp(p(end)/10)+.0001, 10000); end;
mu = p(1:end-1);

%%%%%%%%%%%%%%%%%%%%%%%% setup other
z = size(UT); K = z(1);  %UT is k x nd array; K = # components in model
NP = length(mu);  %number of parameters 
ND = nnz(mask);  % number of data points
mag = max(max(abs(UT)));  %magnitude of target data
q = size(mu); if (q(2)>1) mu=mu'; end;  %force mu to be column vector

% grid etc
T1=range(1); T2 = range(2);  t=[T1:dt:T2]; N=length(t);
ds = .0001;  dsmin=1e-06;  dsmax=1e4;

%%%%%%%%%% even more  setup
FT = @(u,v,w) ( sum(((u-v)/dt -  feval(fode,0,0.5*(u+v),w)).^2,1)  ); %term in optimization expression
eps = 1e-4;  % small parameters for numerical differentiation
eps2 = 1e-4; 
E = eye(K); EP =eye(NP);  %handy 
J = zeros(K,N); D=zeros(K,K,N);  UD=zeros(K,K,N-1); %element storage 
IB = repmat([1:K]',1,K); JB =IB';  %indices for each block in Hessian
for i = 1:N,
    HDI(1+(i-1)*K^2 : i*K^2,1) =  (i-1)*K +IB(:);  % (i,j) indices for hessian 
    HDJ(1+(i-1)*K^2 : i*K^2,1) =  (i-1)*K +JB(:);
end;
HI = [HDI; HDI(1:(N-1)*K^2); HDJ(1:(N-1)*K^2)+K ];  %set of indices used to populate Hessian matrix
HJ = [HDJ; HDJ(1:(N-1)*K^2)+K;  HDI(1:(N-1)*K^2) ];
    
%prepare for loop
if (length(u)==0) u = UT; disp('resetting initial condition'); end; %use UT as initial condition only first time
step=0;  Fold = 1e100; df =1e100; duds = 0*u; dmuds=0*mu;  ng=1e100; lmsg = 0; du = 1e100;

while((ng>1e-6)&&(ds>dsmin))
    un = u;    b = un(:,2:end); a= un(:,1:end-1);
    for k=1:K,  %Compute first derivatives: four point stencil for accuracy
        e = eps*E(:,k)*ones(1,N-1);      
        J(k,:) = 2*lambda*mask(k,:).*(un(k,:)-UT(k,:)) + ...
         [0 (8*FT(b+e,a,mu)-8*FT(b-e,a,mu)-FT(b+2*e,a,mu) + FT(b-2*e,a,mu))]/(12*eps) +...
         [(8*FT(b,a+e,mu)-8*FT(b,a-e,mu)-FT(b,a+2*e,mu) + FT(b,a-2*e,mu)) 0]/(12*eps);
    end;
    Fu = reshape(J,N*K,1); 
      for k1=1:K, for k2=1:K,   % compute second derivatives
        e1 = eps2*E(:,k1)*ones(1,N-1); e2 = eps2*E(:,k2)*ones(1,N-1);
        if (k1==k2)
            D(k1,k2,:) = 2*lambda*ones(1,N).*mask(k1,:) + ...
                   + [0 (FT(b+e1,a,mu) + FT(b-e1,a,mu) - 2*FT(b,a,mu))/eps2^2]...
                   +[(FT(b,a+e1,mu) + FT(b,a-e1,mu) - 2*FT(b,a,mu))/eps2^2 0];
        else
           D(k1,k2,:)=[0 (FT(b+e1+e2,a,mu)+FT(b-e1-e2,a,mu)-FT(b+e1-e2,a,mu)-FT(b-e1+e2,a,mu))]/(4*eps2^2)...
           + [(FT(b,a+e1+e2,mu)+FT(b,a-e1-e2,mu)-FT(b,a+e1-e2,mu)-FT(b,a-e1+e2,mu)) 0]/(4*eps2^2);
        end;
        UD(k2,k1,:) = (FT(b+e1,a+e2,mu)+FT(b-e1,a-e2,mu)-FT(b+e1,a-e2,mu)-FT(b-e1,a+e2,mu))/(4*eps2^2);   
      end; end;
      Fuu = sparse(HI,HJ,[D(:); UD(:); UD(:)],N*K,N*K);  %assemble hessian
    
      M = speye(N*K,N*K) + ds*Fuu;
      R = -ds*Fu;
      du = M\R; [warnMsg, warnId] = lastwarn();
      if (length(warnId)==0)
        unew = u + reshape(du,K,N);   
        W = 0.5*(unew(:,1:N-1)+unew(:,2:N)); %cell centered values of U
        DUDT = (unew(:,2:N)-unew(:,1:N-1))/dt;
        for i=1:N-1,  FW(:,i) = feval(fode,0,W(:,i),mu); end;
        F = sum(sum((DUDT - FW).^2)) + lambda*sum(sum( mask.*(unew-UT).^2));
        rho = (Fold-F)/abs(sum(du.*(du/ds - Fu)));    %typical metric to accept step
      else
        rho=0; disp(['poor condition warning!']);  %reject; probably a bad condition warning from backslash 
      end;
      lastwarn('', '');  %reset warnings
        
      if (rho>.1)  
           u= unew; step=step+1; Fold=F;  ng = norm(Fu(:)); ds= min(ds*1.2,dsmax);
      else
          ds = ds/2; 
      end;

%optional plots during  LM iterations      
      if (mod(step+1,100)==0)
          plot(t,u);  
          title(['steps=' num2str(step)  ', grad=' num2str(ng) ...
    ',   theta= '  num2str(p(1:end-1)) ',   lambda= '  num2str(lambda)]); pause(.1)
      end; 
end;     
     
plot(t,u);  title(['steps=' num2str(step)  ', grad=' num2str(ng) ...
    ',   theta= '  num2str(p(1:end-1)) ',   lambda= '  num2str(lambda)]); pause(.1)

W = 0.5*(u(:,1:N-1)+u(:,2:N)); %cell centered values of u
DUDT = (u(:,2:N)-u(:,1:N-1))/dt;
for i=1:N-1,  FW(:,i) = feval(fode,0,W(:,i),mu); end;
[L,U] = lu(Fuu);   %better than cholesky;  Fuu might be numerically not PD
%diagonal blocks of grad N 
for k=1:K, 
   e = eps2*E(:,k);
   GD(k,:,:) = (8*feval(fode,0,W+e,mu)-8*feval(fode,0,W-e,mu)-feval(fode,0,W+2*e,mu)+feval(fode,0,W-2*e,mu))/(12*eps);
end;
for j=1:N-1,
  dd(j) = det( eye(K)/dt - .5*GD(:,:,j));
end;

W = 0.5*(u(:,1:N-1)+u(:,2:N)); %cell centered values of U
DUDT = (u(:,2:N)-u(:,1:N-1))/dt;
for i=1:N-1,  FW(:,i) = feval(fode,0,W(:,i),mu); end;

if (lambdafixed ==0)
  F = log( sum(sum((DUDT - FW).^2)) + lambda*sum(sum( mask.*(u-UT).^2)))...
   -log(lambda) + 2*(sum(log(abs(diag(U))))/2  - sum(log(abs(dd))))/ND;  %total -log likelihood
else
F = log( sum(sum((DUDT - FW).^2)) + lambda*sum(sum( mask.*(u-UT).^2)));
end;   

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% integrates fode forward and fills in data where mask=0

function datafilled = filldata(data, mask, range, fode, mu)

z = size(data); N = z(2);  dt = (range(2)-range(1))/(N-1);
datatimes = [range(1):(range(2)-range(1))/(N-1):range(2)];
datafilled = data;  % code will overwrite elements where mask=0
fp = @(t,x) feval(fode,0,x,mu);

% find first time where all components are present
for firsti = 1:N,
   if (nnz(1 - mask(:,firsti))== 0) break; end;
end;

%now fill in from here on
for i=firsti:N-1;
   [tt,dd] = ode45(fp,[datatimes(i) datatimes(i+1)],datafilled(i,:));
   datafilled(i+1,:) = mask(:,i+1)'.*data(i+1,:) + (1-mask(:,i+1)').*dd(end,:);
end;

end


