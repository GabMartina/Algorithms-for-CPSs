clear
close all
clc

%% Initialization

load("stochasticmatrices.mat")


n=10;
q=20;

h=2; % # of sensor attacks

magnitude = 0.2; % Refinement of attacks

C = randn(q, n);
G = [C eye(q)];
x = randn(n, 1);

sigma = 1e-2;
noise = sigma * randn(q,1);

%% Analysis of eigenvalues and choice of schochastic matrix

% We check the two largest eigenvalues in magnitude.
% If eigPF=1 > abs(eig_2nd) > ... > abs(last eig) we meet the suff.
% condition for aver. consensus
eigs(Q1,2)
eigs(Q2,2)
eigs(Q3,2)
eigs(Q4,2)

% Choice of matrix
Q = Q4;

%% Generation of unaware attack [-2 -1] u [1 2]

b = 2; 
c = 1;

iter=20;
% iter_debug = 3;
success=0;

for count = (1:iter )
    a_unw = ones(1, h);
    for i = (1:h)
        a_unw(i) = (b-c)*rand(1)+c;
        prob_sign = rand;
        if prob_sign < 0.5 % with probability 50% the value is on the negative half of the domain
            a_unw(i) = a_unw(i) * -1;
        end
    end
    
    % add n - S values that are 0 and shuffle the vector
    a_unw = [a_unw zeros(1, q - h)];
    a_unw = a_unw(randperm(length(a_unw)));
    a_unw = a_unw';
    support_aun = find(a_unw);
    
    tau=0.03;
    lambda=2*(10^-4)/tau;
    LAMBDA = [zeros(n,1); (tau*lambda) * ones(q, 1)];
    
    y = C * x + noise + a_unw;
    
    %filtered_a = arrayfun(@(x) filter_attacks(x,magnitude), z(n+1:end, 1));
    
    %% algorithm DIST

    delta = 1e-7;
    T=1e5;
    %z = zeros(n+q, T+1);
    z_k = zeros(n+q, q); %i-th column is the current estimate of z for sensor i
    z_k_next = zeros(n+q, q);
    somma=0;
    SOMMA=0;
    
    for k = (1:T)
        SOMMA=0; %Used to compute sum of norms at time k for stop criterion

        for i = (1:q) %Computations for every node at "instant k"
            somma=0;
            
            % Calculating the argument of the shrinkage func
            for j = (1:q)
                somma = somma + (Q(i,j) * z_k(:,j)); %choose the right Q
            end
            somma = somma + tau*G(i,:)'*( y(i,1) - G(i,:)*z_k(:,i));
            
            % Calling shrinkage
            z_k_next(:,i) = shrinkage(somma, LAMBDA); %next estimation by i-th sensor
        end

        % Do our estimations reach the stop criterion?
        for t = (1:q)
            SOMMA = SOMMA + norm(z_k_next(:,i) -z_k(:,i) ,2);
        end
        
        if SOMMA < delta
            T=k;
            %SOMMA
            %T
            break;
        end

        z_k = z_k_next;

    end
    
    %update success
    
    % Refinement of attack
    filtered_a = arrayfun(@(x) filter_attacks(x,magnitude), z_k_next(n+1:end, 1));
    
    support_next = find(filtered_a);
    support_h = find(a_unw);
    if isequal(support_h,support_next) 
        success = success + 1;
    end

    % Estimation accuracy
    x_mean_estimates=mean(z_k(1:10,:),2);
    result = norm(x-x_mean_estimates,2)^2;
    X = ['x true - x mean estimate l2-norm square: ',num2str(result)];
    disp(X)
    X = ['# of iteration for try #',num2str(count),' :',num2str(k)];
    disp(X)
    
end

%Rate of correct attack detection
success = success/count * 100;

X = ['rate of correct attack detection', num2str(success)]
disp(X)



