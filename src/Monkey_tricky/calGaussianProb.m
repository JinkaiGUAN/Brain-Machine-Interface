function prob = calGaussianProb(neuroFiringRate, meanVal, stdVal)
    if stdVal < 1e-5
        stdVal = 1e-9;
    end
    exponent = exp(-((neuroFiringRate - meanVal)^2 / (2 * stdVal^2 )));
   
    prob =  (1 / (sqrt(2 * pi) * stdVal)) * exponent;

end
