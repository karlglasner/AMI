
%Lorenz system
%note function must be written for array evaluation!


function f = fhn(t,q,mu);

sigma = mu(1); rho = mu(2); beta = mu(3);
x = q(1,:); y = q(2,:); z = q(3,:);

f(1,:) = sigma*(y-x);
f(2,:) = x.*(rho - z)-y;
f(3,:) = x.*y-beta*z;
