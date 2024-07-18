function y = filter_attacks(x,magnitude)
    if abs(x) < magnitude 
        y = 0;
    else
        y = x;
    end
end