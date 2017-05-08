

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jacobi method for computing the SVD of A, A*V = U*S 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create random data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 20; n=10;
rng('default');
A = randn(m,n);    % we want to orthogonalize columns of A
B = A;
V = eye(n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set termination criteria 
%     (a) smalness of off(A'*A), 
%     (b) max number of sweeps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sum = norm(A); i = 0;
maxsweep = 100; threshold = n*norm(B)*eps;

while ((i < maxsweep) && (sqrt(sum) > threshold)),
  sum = 0.0;

% cyclic by row annihilation ordering
  for j = 1:n-1,
    for k = j+1:n,

% get the working 2 by 2 submatrix of A'*A
      a = A(:,[j k])'*A(:,[j k]);
% accumulate on sum the current off(A)
      sum = sum + 2*a(1,2)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute rotation parameters c and s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      tau = (a(2,2)-a(1,1))/(2*a(1,2));
% 
% want abs(t) < 1 but to avoid numerical cancelations  we have
%     abs(t) = abs(-tau - sign(tau))*sqrt(1+tau^2))> 1
% old trick, take the other root, 
%     t = -tau + sign(tau)*sqrt(1+t^2)
% do not compute it but rather multiply and divide by
%     t = -tau -sign(tau)*sqrt(1+t^2)
% to safely obtain the smaller root 
      t = 1/(tau + sign(tau)*sqrt(1+tau^2));
      c = 1/sqrt(1+t^2); s = c*t;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rotate columns of A, and accumulate V from A*V = U*S
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      G = [c s; -s c];
      A(:, [j k]) = A(:, [j,k])*G;
      V(:, [j k]) = V(:, [j,k])*G;
    end
  end
  i = i+1;
end 
       
