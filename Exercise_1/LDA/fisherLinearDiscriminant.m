function v = fisherLinearDiscriminant(X1, X2)

    m1 = size(X1, 1);
    m2 = size(X2, 1);

    mu1 = mean(X1);
    mu2 =mean(X2);

    S1 = cov(X1);
    S2 = cov(X2);

    Sw = 0.5*S1+0.5*S2;
    
    inv_sw = inv(Sw);
    inv_sw_sb = inv_sw*(mu1-mu2)';
    

    v = inv_sw_sb./norm(inv_sw_sb);% return a vector of unit norm
