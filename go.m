
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This code provides a template implementation of the Calibrated-AdaMEC
% method proposed by Nikolaou et al 2016, "Cost Sensitive Boosting
% Algorithms: do we really need them?".
% 
% This follows the pseudocode laid out on p15 of the supplementary
% material.  If you make us of this code, please cite the main paper.
%
% Thanks! Happy Boosting!
% Nikolaou et al.
%
%%%%%%%%%

clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LOAD and SHUFFLE THE DATA
%
load data/spliceM
labels(labels~=1) = -1;
numExamples = length(labels);
%
% FIXED SEED FOR REPRODUCIBILITY DURING TESTING
rng(123);
%
randomOrder = randperm(numExamples);
data = data(randomOrder,:);
labels = labels(randomOrder);
%
% SPLIT IT UP INTO TRAIN/CALIBRATE/TEST SETS
datasets = splitData([0.25 0.25 0.5], data, labels);
Dtrain = datasets{1};
Dcalib = datasets{2};
Dtest = datasets{3};



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TRAIN A NORMAL ADABOOST ENSEMBLE
%
T=5; %ENSEMBLE SIZE
model = adaboost(logreg, T); %USE LOGISTIC REGRESSIONS
model = model.train(Dtrain.data, Dtrain.labels);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%START THE CALIBRATION PROCESS
%
%CALCULATE VOTES ON CALIBRATION DATA
[H votes] = model.test(Dcalib.data);

%CALCULATE SCORES
scores = calculateScores(model.alpha, votes);

%USE PLATT SCALING
[A B] = plattScaling(scores, Dcalib.labels);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NOW TEST
%
%CALCULATE VOTES ON TEST DATA
[H testvotes] = model.test(Dtest.data);

%CALCULATE SCORES
scores = calculateScores(model.alpha, testvotes);

%APPLY PLATT SCALING PARAMETERS DERIVED FROM TRAINING DATA
probs = 1 ./ (1+exp(A*scores + B) );




%COST OF FP/FN (NOTE: THIS CAN BE SET AT TEST TIME ONLY FOR AdaMEC)
cFP = 5;
cFN = 1;
threshold = cFP/(cFP+cFN);
%APPLY MINIMUM EXPECTED COST (MEC) THRESHOLD
calibH(probs>threshold,1)   = +1;
calibH(probs<=threshold,1)  = -1;




%OUTPUT RESULTS!
accuracy = mean(H==Dtest.labels);
calibratedAccuracy = mean(calibH==Dtest.labels);

%CALCULATE SKEW, z, EQ(27) IN THE PAPER.
probYpos = sum(Dtest.labels==1)/length(Dtest.labels);
probYneg = 1-probYpos;
z = (probYneg*cFP) / ((probYneg*cFP)+(probYpos*cFN)); 

%PRINT THE COSTS FOR INFORMATION
disp(['Cost of False Positive: ' num2str(cFP)]);
disp(['Cost of False Negative: ' num2str(cFN)]);

%PRINT THE CONFUSION MATRIX
UncalibratedConfusionMatrix = confusionmat(H,Dtest.labels);
TP = sum(H==+1 & Dtest.labels==+1);
TN = sum(H==-1 & Dtest.labels==-1);
FP = sum(H==+1 & Dtest.labels==-1);
FN = sum(H==-1 & Dtest.labels==+1);

%CALCULATE THE [0,1] NORMALIZED COST, EQ (26)
fpr = FP/(FP+TN); fnr = FN/(FN+TP);
UncalibratedCost = fpr*z + fnr*(1-z); %eq(26) from the paper

%PRINT A MESSAGE ABOUT THE UNCALIBRATED VERSION
disp(sprintf('\nUncalibrated cost = %f',UncalibratedCost));
disp('Matrix  = ');
disp(UncalibratedConfusionMatrix);

%PRINT THE CONFUSION MATRIX AGAIN
CalibratedConfusionMatrix = confusionmat(calibH,Dtest.labels);
TP = sum(calibH==+1 & Dtest.labels==+1);
TN = sum(calibH==-1 & Dtest.labels==-1);
FP = sum(calibH==+1 & Dtest.labels==-1);
FN = sum(calibH==-1 & Dtest.labels==+1);

%CALCULATE THE [0,1] NORMALIZED COST WHEN CALIBATED
fpr = FP/(FP+TN); fnr = FN/(FN+TP);
CalibratedCost = fpr*z + fnr*(1-z); %eq(26) from the paper

%PRINT A MESSAGE ABOUT THE CALIBRATED RESULTS!
disp(sprintf('Calibrated cost  = %f',CalibratedCost));
disp('Matrix = ');
disp(CalibratedConfusionMatrix);






