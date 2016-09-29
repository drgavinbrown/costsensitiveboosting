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
load data/krvskp
labels(labels~=1) = -1;
numExamples = length(labels);
%
% FIXED SEED FOR REPRODUCIBILITY DURING TESTING
r = randi(99999);
rng(r);
disp(['Random Seed: ' num2str(r)]);
%
randomOrder = randperm(numExamples);
data = data(randomOrder,:);
labels = labels(randomOrder);
%
% SPLIT IT UP INTO TRAIN/TEST SETS
datasets = splitData([0.5 0.5], data, labels);
Dtrain = datasets{1};
Dtest = datasets{2};



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SET SIZE FOR BOTH ENSEMBLES
%
T=10;
%SET COST OF FP/FN (NOTE: WITH AdaMEC, THIS CAN BE SET AT *TEST* TIME)
cFP = 5;
cFN = 1;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TRAIN A NORMAL ADABOOST ENSEMBLE
%
model = adaboost(logreg, T); %USE LOGISTIC REGRESSIONS
model = model.train(Dtrain.data, Dtrain.labels);
%
%CALCULATE VOTES ON TEST DATA
uncalibPredictions = model.test(Dtest.data);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NOW TRAIN A CALIBRATED/SHIFTED ADABOOST ENSEMBLE
%
%FIRST SPLIT THE TRAIN DATA INTO MODEL FITTING/CALIBRATION SETS
datasets = splitData([0.5 0.5], Dtrain.data, Dtrain.labels);
Dfitting = datasets{1};
Dcalib = datasets{2};
%
model = adaboost(logreg, T);
model = model.train(Dfitting.data, Dfitting.labels);

%START THE CALIBRATION PROCESS
%

%CALCULATE VOTES ON CALIBRATION DATA
[H calibvotes] = model.test(Dcalib.data);
%CALCULATE SCORES
scores = calculateScores(model.alpha, calibvotes);
%CALCULATE PLATT SCALING PARAMETERS FROM CALIBRATION DATA
[A B] = plattScaling(scores, Dcalib.labels);


%FINALLY, CALCULATE VOTES ON TEST DATA
[H testvotes] = model.test(Dtest.data);
%CALCULATE SCORES
scores = calculateScores(model.alpha, testvotes);
%APPLY PLATT SCALING PARAMETERS DERIVED FROM *CALIBRATION* DATA
probs = 1 ./ ( 1+exp(A*scores + B) );
%SET VOTE THRESHOLD
threshold = cFP/(cFP+cFN);
%APPLY MINIMUM EXPECTED COST (MEC) THRESHOLD
calibPredictions(probs>threshold,1)   = +1;
calibPredictions(probs<=threshold,1)  = -1;






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUT RESULTS!
accuracy = mean(uncalibPredictions==Dtest.labels);
calibratedAccuracy = mean(calibPredictions==Dtest.labels);

disp(sprintf('\nUncalibrated error = %f',1-accuracy));
disp(sprintf('Calibrated error = %f\n',1-calibratedAccuracy));



%CALCULATE SKEW, z, EQ(27) IN THE PAPER.
%NOTE: z IS CALCULATED FROM TESTING DATA
%
probYpos = sum(Dtest.labels==1)/length(Dtest.labels);
probYneg = 1-probYpos;
z = (probYneg*cFP) / ((probYneg*cFP)+(probYpos*cFN)); 

%PRINT THE COSTS FOR INFORMATION
disp(['Train p(y=1) = ' num2str(sum(Dtrain.labels==1)/length(Dtrain.labels))]);
disp(['Test  p(y=1) = ' num2str(probYpos)]);
disp(sprintf(' '));
disp(['Cost of False Positive: ' num2str(cFP)]);
disp(['Cost of False Negative: ' num2str(cFN)]);



%PRINT THE UNCALIBRATED CONFUSION MATRIX
TP = sum(uncalibPredictions==+1 & Dtest.labels==+1);
TN = sum(uncalibPredictions==-1 & Dtest.labels==-1);
FP = sum(uncalibPredictions==+1 & Dtest.labels==-1);
FN = sum(uncalibPredictions==-1 & Dtest.labels==+1);
UncalibratedConfusionMatrix = [ TN FN; FP TP ];

%CALCULATE THE [0,1] NORMALIZED COST, EQ (26)
fpr = FP/(FP+TN); fnr = FN/(FN+TP);
UncalibratedCost = fpr*z + fnr*(1-z); %eq(26) from the paper

%PRINT A MESSAGE ABOUT THE UNCALIBRATED VERSION
disp(sprintf('\nUncalibrated cost = %f',UncalibratedCost));
disp('Matrix  = ');
disp(UncalibratedConfusionMatrix);




%PRINT THE CALIBRATED CONFUSION MATRIX
TP = sum(calibPredictions==+1 & Dtest.labels==+1);
TN = sum(calibPredictions==-1 & Dtest.labels==-1);
FP = sum(calibPredictions==+1 & Dtest.labels==-1);
FN = sum(calibPredictions==-1 & Dtest.labels==+1);
CalibratedConfusionMatrix  = [ TN FN; FP TP ];

%CALCULATE THE [0,1] NORMALIZED COST WHEN CALIBRATED
fpr = FP/(FP+TN); fnr = FN/(FN+TP);
CalibratedCost = fpr*z + fnr*(1-z); %eq(26) from the paper

%PRINT A MESSAGE ABOUT THE CALIBRATED RESULTS!
disp(sprintf('Calibrated cost  = %f',CalibratedCost));
disp('Matrix = ');
disp(CalibratedConfusionMatrix);






