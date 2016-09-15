

function [A B] = plattScaling( scores, calibrationLabels )

%START THE CALIBRATION PROCESS
Npos = sum(calibrationLabels==+1);
Nneg = sum(calibrationLabels==-1);

yprime = zeros(Npos+Nneg,1);

yprime(calibrationLabels==+1) = (Npos+1) / (Npos+2);
yprime(calibrationLabels==-1) = 1 ./ (Nneg+2);

%Function to be fitted
sigfunc = @(params, x)(1 ./ (1 + exp(params(1)*x + params(2))));

%Hyperparameters (MaxIter: maximum no of iterations, TolFun: Residual error to be tolerated )
options = statset('MaxIter', 600, 'TolFun', 1e-10);

%Find optimal parameters for the sigmoid
params = nlinfit(scores, yprime, sigfunc, [1 1], options);

%return the fitted parameters
A = params(1);
B = params(2);

end
