function [x, history] = huber_fit(A, b, rho, alpha)
% huber_fit    Solve huber fitting problem via ADMM
%
% Solves the following problem via ADMM:
%
%   minimize 1 / 2 * sum(huber(Ax - b))  
%
% The solution is returned in the vector x

% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual 
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter 
%
% alpha is the over-relaxation parameter (typical valus for alpha are
% between 1.0 and 1.8).
t_start = tic;

%% Global constants and defaults
QUIET = 0;
MAX_ITER = 1000;
ABSTOL = 1e-4;
RELTOL = 1e-2;

%% Data preprocessing 
[m, n] = size(A);

% save a matrix-vector multiply
Atb = A' * b;

%% ADMM solver
x = zeros(n, 1);
z = zeros(m, 1);
u = zeros(m, 1);

% cache the factorization
[L, U] = factor(A);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1: MAX_ITER
    
    % x-update
    x = U \ (L \ (Atb + A' * (z - u)));
    
    % z-update 
    zold = z;
    Ax_hat = alpha * A * x + (1 - alpha) * (zold + b); % sort of momentum
    tmp = Ax_hat - b + u;
    z = rho / (1 + rho) * tmp + 1 / (1 + rho) * shrinkage(tmp, 1 + 1 / rho);
    
    % u-update
    u = u + (Ax_hat -z -b);
    
    % diagnostics, reporting, termination checks
    history.objval(k) = objective(A, b, x);
    
    history.r_norm(k) = norm(A * x - z - b);
    history.s_norm(k) = norm(-rho * A' * (z - zold));
    
    history.eps_pri(k) = sqrt(m) * ABSTOL + RELTOL * max(norm(A * x), max(norm(-z), norm(b)));
    history.eps_dual(k) = sqrt(n) * ABSTOL + RELTOL * norm(rho * A' * u);
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    
    if(history.r_norm(k) < history.eps_pri(k) && ...
            history.s_norm(k) < history.eps_dual(k))
        break;
    end
end

if ~QUIET
    toc(t_start);
end
end

%% Functions used above
function p = objective(A, b, x)
    sol = A * x - b;
    sol_abs = abs(sol);
    p1 = (sol .^ 2) / 2; % when |a| <= 1
    p2 = abs(sol) - 1 / 2; % when |a| > 1
    p = sum(p1 .* (sol_abs <= 1) + p2 .* (sol_abs > 1));
end

function z = shrinkage(x, kappa)
    z = max(0, x - kappa) - max(0, -x - kappa);
end

function [L, U] = factor(A)
    L = chol(A' * A, 'lower');
    U = L';
end