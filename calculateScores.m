% CALCULATE SCORES
%
% This code calculates eq(24) in the main paper Nikolaou et al 2016.
%
function scores = calculateScores(alpha, votes)

denominator = sum(alpha);

votes(votes==-1) = 0;

%FIND ALL POSITIVE VOTE WEIGHTS
for t=1:length(alpha)
    tmp = votes(:,t);
    tmp(tmp==1) = alpha(t);
    votes(:,t) = tmp;
end
numerators = sum(votes,2);

%SCORES ARE THE SUM OF ALL POSITIVE VOTE WEIGHTS, NORMALISED.
scores = numerators./denominator;

end
