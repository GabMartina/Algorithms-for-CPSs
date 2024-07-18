function x = k_sparse_support(k, len)

    tmp = ones(k, 1);
    % add len - k values that are 0 and shuffle the vector
    x = [tmp; zeros(len - k, 1)];
    x = x(randperm(length(x)));
    
end