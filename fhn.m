
%FitzHugh–Nagumo Model with

function y = fhn(t,x,mu);

a = mu(1); b = mu(2); c = sqrt(mu(3));
V = x(1,:); R = x(2,:); 
y(1,:) = c^2*(V-V.^3/3+R); 
y(2,:) = -(V-a - b*R);

