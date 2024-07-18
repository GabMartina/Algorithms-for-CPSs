%% data declaration

clear all
clc
close all

q = 20;
p = 20;
k = 2;

C = randn(q, p);

eps = 10^-8;
tau = norm(C, 2).^-2 - eps;

lambda = 1/(10*tau);
LAMBDA = lambda * ones(p, 1);
sigma = 10^-2;
noise = sigma^2 * randn(q, 1);

delta = 10^-12;

% xi belongs to [-2, -1] U [1, 2] 
b = 2; 
a = 1;

%% IST algorithm

x_t = zeros(p, 1); % current estimation

success = 0;
count = 0;

iter = 20;

track_count = zeros(iter, 1);
x_estimated = zeros(p, iter);
x_real = zeros(p, iter);
successful_supports = zeros(iter, 1);

for exp = (1:iter)
    count = 0;

    % create a k sparse vector with non-zero values within [-b -a] U [a b]
    x = k_sparse(k, p, a, b);

    support_x = find(x);

    y = C*x + noise;

    x_t = zeros(p, 1); % current estimation

    while true
    
        grad = (C')*(y-C*x_t);
        grad = x_t + tau*grad;
        l = tau*LAMBDA;
        x_t_next = shrinkage(grad, l);
    
        count = count + 1;
    
        if norm(x_t_next - x_t, 2) < delta % Tmax
            break;
        end
    
        % update x_t
        x_t = x_t_next;
    
    
    end

    support_next = find(x_t_next);
    if size(support_x) == size(support_next) & isequal(support_x,support_next)
        success = success + 1;
        successful_supports(exp) = 1;
    end

    x_estimated(:,exp) = x_t_next;
    x_real(:, exp) = x;

    track_count(exp) = count;

end

% plot success rate
success = success/iter * 100; 
disp(['Success: ', num2str(success), '%']);

% plot stats over count
min_count = min(track_count);
disp('Convergence time (# of iterations needed)');
disp(['- min: ', num2str(min_count)]);
max_count = max(track_count);
disp(['- max: ', num2str(max_count)]);
avg_count = mean(track_count);
disp(['- avg: ', num2str(avg_count)]);


% plot experiment outcome
x_axis = 1:iter;
x_estimated_to_plot = cell(1, iter);
x_real_to_plot = cell(1, iter);

for i = 1:p 
    x_real_to_plot{i} = find(x_real(:, i));
    x_estimated_to_plot{i} = find(x_estimated(:, i));
end

figure
hold all
for i = 1:numel(x_axis)
    l1 = plot(ones(1,numel(x_estimated_to_plot{i}))*x_axis(i), x_estimated_to_plot{i}, 'r*');
    l2 = plot(ones(1,numel(x_real_to_plot{i}))*x_axis(i), x_real_to_plot{i}, 'ko');
end
hold off

x_green_stripes = zeros(1, nnz(successful_supports) * 5);
y_green_stripes = zeros(1, nnz(successful_supports) * 5);
i = 1; 
j = 1;
while i<=5*nnz(successful_supports)
    if successful_supports(j) == 1
        x_green_stripes(i) = max(j - 1/2, 0);
        y_green_stripes(i) = 0;

        x_green_stripes(i + 1) = min(j + 1/2, 20);
        y_green_stripes(i + 1) = 0;

        x_green_stripes(i + 2) = min(j + 1/2, 20);
        y_green_stripes(i + 2) = p;

        x_green_stripes(i + 3) = max(j - 1/2, 0);
        y_green_stripes(i + 3) = p;

        x_green_stripes(i + 4) = max(j - 1/2, 0);
        y_green_stripes(i + 4) = 0;

        i = i + 5;
    end
    j = j + 1;
end
p = patch(x_green_stripes, y_green_stripes, 'green', 'FaceAlpha', 0.1, 'EdgeColor','none');

set(gca,'XGrid','on');
set(gca,'YGrid','off');
set(gca, 'XTick',x_axis);  
h = [l1; l2; p];
legend(h, 'estimated support', 'actual support', 'successful estimation', 'Location', 'southoutside');
xlabel('Iteration #');
ylabel('Support (= p)');
daspect([1 2 1])
