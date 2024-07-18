function s = shrinkage(grad, l)  %x è l'approssimazione del vettore da raggiungere
s = zeros(length(grad), 1);

for i = (1:length(grad))
    if grad(i) > l(i)
        s(i) = grad(i) - l(i);
    elseif grad(i) < -l(i)
        s(i) = grad(i) + l(i);
    elseif abs(grad(i)) <= l(i)
        s(i) = 0;
    end
end

end

