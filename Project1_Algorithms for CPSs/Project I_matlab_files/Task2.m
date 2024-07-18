clear all
close all
clc

%% data declaration

p = 10;
q = 20;
h =  2; % # of sensor attacks
C = randn(q, p);

eps = 10^-8;
tau = norm(C, 2).^-2 - eps;
tau_lambda = 2*10^-3;
LAMBDA = [zeros(p,1); (tau_lambda/tau) * ones(q, 1)];
om = 1e-2;
noise = om * randn(q,1);

b = 2; 
a = 1;

% exit condition
delta = 10^-12;

%% IST algoritm

n_iter = 20;
n_experiments = 20;
results = zeros(3, n_experiments);

% ---> this to identify average success - ends here (*)
for j = (1:n_experiments)

    success = 0;
    count = 0;
% ---|
    
    for iter = (1:n_iter)
    
        x = randn(p, 1);
        y = C * x + noise;

        % Unaware attack
        a_unaware = k_sparse(h, q, a, b);
    
        % Aware attack
        a_aware = k_sparse_support(h, q);
        a_aware = 0.5*y.*a_aware;
        support_aaw = find(a_aware);
    
        % choice of what attack
        a_curr = a_aware;
    
        y = y + a_curr;
    
        w_t = zeros(p+q, 1); % current estimation

        G = [C eye(q)];
        w = [x; a_curr];
    
        while true
            grad = (G')*(y-G*w_t);
    
            w_t_next = shrinkage(w_t + tau*grad, tau*LAMBDA);
        
            count = count + 1;
        
            if norm(w_t_next - w_t, 2) < delta % = Tmax
                break;
            end
        
            % update w_t
            w_t = w_t_next;
        
        end
    
        support_w_t_next = find(w_t_next(p+1:end));
        support_attack = find(a_curr);

        if isequal(support_attack,support_w_t_next) 
            success = success + 1;
        end
    
    end

    % collect stats on the experiment
    success = success/n_iter * 100;
    x_estimate = w_t_next(1:p);
    l2_norm = norm(x_estimate - x, 2);
    
    results(1, j) = success;
    results(2, j) = count;
    results(3, j) = l2_norm;

% ---> (*)
end
% ---|

avg_success = mean(results(1,:));
avg_norm = mean(results(3,:));
disp(['Success rate: ', num2str(avg_success)]);
disp(['Estimation of x accuracy: ',num2str(avg_norm)])