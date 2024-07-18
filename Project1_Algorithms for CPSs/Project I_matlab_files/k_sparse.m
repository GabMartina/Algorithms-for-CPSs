function x = k_sparse(k, len, domain_start, domain_end) 

    x = k_sparse_support(k, len);

    for i = (1:k)
        if x(i) == 1
            x(i) = (domain_end - domain_start) * rand(1) + domain_start;
            prob_sign = rand;
            if prob_sign < 0.5 % with probability 50% the value is on the negative half of the domain
                x(i) = x(i) * -1; 
            end
        end
    end

end