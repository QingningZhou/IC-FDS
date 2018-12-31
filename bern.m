%%%%% Bernstein basis polynomials

function b = bern(j,p,l,u,t)
    b = mycombnk(p,j)*(((t-l)/(u-l)).^j).*((1-(t-l)/(u-l)).^(p-j));
end