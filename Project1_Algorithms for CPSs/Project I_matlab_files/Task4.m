clear all
close all
clc

UNAWARE = 0;
AWARE = 1;

%% parameters
load('task4data.mat')

%% start up

j = 4; % # of targets
p = 100; % # of cells
q = 25; % # of sensors
h = 2; % # of attacks on sensors

sigma = 0.2;
noise = sigma * randn(q,1);

T = 50;

G = [D eye(q)];
G = normalize(G);

eps = 10^-9;
tau = norm(G,2)^-2 - eps;
lambda = [10*ones(p, 1); 20*ones(q, 1)];

% support of target vector
x = [ones(j, 1); zeros(p - j, 1)];
x = x(randperm(length(x)));

% support of attack vector 
a_support = [ones(h, 1); zeros(q - h, 1)];
a_support = a_support(randperm(length(a_support)));

% change here the type of attack to simulate: {AWARE, UNAWARE}
attack_type = UNAWARE; 

% unaware -->
if attack_type == UNAWARE
    ai = 30;
    a = ai * a_support;
end
% ---|

% z = [x a] with z(0) = 0
z = zeros(p + q, 1); 

%% algorithm

for k = (1:T)

    % Observation of a dynamic system, state and output changing over time
    x = A * x;
    y = D * x + noise;
    
    % ----> aware
    if attack_type == AWARE
        a = 0.5*y.*a_support;
    end
    % ---|

    y = y + a;

    % Algorithm of sparse observer
    grad = G'*(y - G*z);
    z_half_next = shrinkage(z + tau*grad, tau*lambda);
    x_next = A*z_half_next(1:p, 1);
    a_next = z_half_next(p+1:end, 1);
    z = [x_next; a_next];

    % Refinement of state estimation (taking j=4 largest components)
    [x_estimate, support] = maxk(z(1:p, 1), j);
    x_filtered=zeros(p,1);
    for tmp = 1:length(support)
        x_filtered(support(tmp)) = x_estimate(tmp);
    end    
    
    % Refinement of attacks
    if attack_type == UNAWARE
        a_filtered = arrayfun(@(x) filter_attacks(x, 2), z(p+1:end, 1));
        % a_filtered = z(p+1:end,1);
    elseif attack_type == AWARE
        % a_filtered = z(p+1:end,1);
        a_filtered = arrayfun(@(x) filter_attacks(x, 2), z(p+1:end, 1));
    end

    % Plots comparing true evolution and estimation
    subplot(2,1,1);
    curve_1 = plot(x, 'k');
    hold on
    curve_2 = plot(x_filtered, 'r');
    grid on
    legend('actual position of targets','estimated target position', 'Location', 'southoutside')
    
    subplot(2,1,2);
    curve_3 = plot(a, 'k');
    hold on
    curve_4 = plot(a_filtered, 'b');
    legend('actual attacks','estimated attacks', 'Location', 'southoutside')
    
    pause(0.2);
    if k~=T
        delete(curve_1);
        delete(curve_2);
        delete(curve_3);
        delete(curve_4);
    end

end




