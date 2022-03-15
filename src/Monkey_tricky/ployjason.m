function [beta] = ployjason(x,y,q)
    n = length(y);
    b = zeros(n,1);
    A = zeros(n,q+1);
    for i=1:n
        b(i) = y(i);
        for j=1:q+1
            A(i,j) = x(i)^(q+1-j);
        end
    end
    C = A'*A;
    d = A'*b;
    beta = C\d;
end