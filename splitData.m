
%GENERIC METHOD FOR DIVIDING A DATASET INTO CHUNKS
%Arguments:
% splitProportion - a vector of M values, must sum to 1.0
% data, labels - the dataset to split
%
%Returns:
% datasets - a cell array containing M structs
%            each with two fields called 'data' and 'labels'
%
%Example usage:
%
% sets = splitData([0.4 0.4 0.2], data,labels)
%
% trainData = sets{1}.data
% trainLabels = sets{1}.labels
%
function datasets = splitData(splitProportion, data, labels)

if sum(splitProportion) ~= 1.0
    error('Argument splitProportion must sum to 1.0');
end

numExamples = length(labels);
numSplits = length(splitProportion);

datasets = cell(numSplits,1);
firstItem = 1;
for d=1:numSplits
    
    %size of this split
    examplesInThisSplit = round(splitProportion(d)*numExamples);
    
    %identify the final item in this split
    lastItem = firstItem+(examplesInThisSplit-1);
    if d==numSplits
        lastItem = numExamples;
    end
      
    %get the indices to go in this split
    indices = firstItem:lastItem;
    
    %for debugging
    %[firstItem lastItem]
    
    %extract the data
    datasets{d} = struct('data',data(indices,:),'labels',labels(indices));
    
    firstItem = lastItem+1;
end

