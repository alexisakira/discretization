function [U,fval] = minVarTrace(A)
% find a unitary matrix U such that the diagonal components of U'*AU is as
% close to a multiple of identity matrix as possible
warningState = warning;
warningCleanup = onCleanup(@() warning(warningState));
warning('off','all');

[s1,s2] = size(A);
if s1 ~= s2
    error('input matrix must be square')
end

K = s1; % size of A
d = trace(A)/K; % diagonal of U'*A*U should be closest to d

if K == 1
    U = 1;
    fval = 0;
    return
end

if K == 2
    % A plane rotation equalizes the two diagonal elements exactly. This
    % is the closed-form solution reached numerically from the identity.
    offDiagonal = (A(1,2)+A(2,1))/2;
    theta = 0.5*atan2(A(2,2)-A(1,1),2*offDiagonal);
    U = [cos(theta) -sin(theta);sin(theta) cos(theta)];
    fval = norm(diag(U'*A*U)-d);
    return
end

obj =@(X)(norm(diag(X'*A*X)-d));
persistent options
if isempty(options)
    options = optimoptions(@fmincon,'Display','off');
end
[U,fval] = fmincon(obj,eye(K),[],[],[],[],[],[],@unitaryConstraint,options);

end

