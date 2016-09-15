% Adaboost
% Implements the Adaboost algorithm for an arbitrary base classifier.
% Code (c) Gavin Brown, Nikos Nikolaou 2016
%
% Assumes labels are in {-1,+1}
%
% model = adaboost(nbayes,5);
% model = model.train(data,labels);
% predictions = model.test(testdata,testlabels).labels;

classdef adaboost <  handle
    
   properties

        members;        %cell array of classifier objects
        maxSize;        %how many members should there be?
        currentSize;    %how many members currently?
        basemodeltype;  %command to eval when making a new one
 
        %fields specific to Adaboost
        D;
        epsilon;
        alpha;
        
    end
    
    methods

               
        function newmodel = clone(model)
            
            newmodel = adaboost( model.basemodeltype, model.maxSize );
            for i=1:model.maxSize
                newmodel.members{i} = model.members{i}.clone();
            end
            
            newmodel.maxSize = model.maxSize;
            newmodel.currentSize = model.currentSize;
            newmodel.basemodeltype = model.basemodeltype;
            newmodel.D = model.D;
            newmodel.epsilon = model.epsilon;
            newmodel.alpha = model.alpha;
        end%clone
        
        
        function model = adaboost( basemodeltype, maxSize )
            
            model.maxSize = maxSize;
            model.currentSize = 0;
            model.alpha = 0;
            model.basemodeltype = basemodeltype;
            
            for i=1:maxSize
                model.members{i} = basemodeltype().clone();
            end
            
        end%constructor
        

        
        
        function model = train(model, data, labels)
           
            if length(unique(labels))~=2
                error('Sorry, Adaboost is a 2-class classifier.');
            end
            if sum( unique(labels)==[-1 1]' )~=2
                error('Sorry, Adaboost requires labels to be in [-1,+1]');
            end
            
            
            numtr = size(data,1);
            
            %INITIAL UNIFORM DISTRIBUTION
            model.D = zeros(numtr,model.maxSize);
            model.D(:,1) = ones(numtr,1) / numtr;
            
            %LOOP T TIMES
            while model.currentSize<model.maxSize
                
                model.currentSize = model.currentSize + 1;
                t = model.currentSize;
                
                %RESAMPLE DATA ACCORDING TO DISTRIBUTION D_t
                indices = randsample( numtr, numtr, true, model.D(:,t) );
                datasamp = data(indices,:);
                labelsamp = labels(indices);
                
                %BUILD A WEAK HYPOTHESIS
                model.members{t} = model.basemodeltype().clone();
                model.members{t}.train( datasamp, labelsamp );
                
                
                %TEST IT
                guess = model.members{t}.test(data);
                
                
                %CALCULATE ITS ERROR
                correct   = ( guess == labels );
                
                %CALCULATE ALPHA WEIGHTS
                model.epsilon(t) = sum( model.D(~correct,t) );
                if model.epsilon(t) > 0.5
                    model.currentSize = model.currentSize - 1;
                    model.D(:,t-1) = ones(numtr,1) / numtr;
                    %disp('Err > 0.5' );
                    continue;
                end
                
                %UPDATE DISTRIBUTION
                model.D( correct,   t+1 ) = model.D(correct,t)   .* 1/(2*(1-model.epsilon(t)));
                model.D( ~correct,  t+1 ) = model.D( ~correct,t) .* 1/(2*model.epsilon(t));
                
                model.alpha(t) = 0.5 * log( (1-model.epsilon(t))/model.epsilon(t) );
                
            end
            
            
            
            
        end%train
        
        
        function [predictedlabels testingvotes] = test(model, data)
            
            testingvotes = zeros(size(data,1),model.maxSize);
            weightedtestingvotes = zeros(size(data,1),model.maxSize);

            for t=1:model.maxSize
                
                %RECORD TEST DATA VOTES
                tmp = model.members{t}.test(data);
                uniq = unique(tmp); %unique always returns a sorted list
                v(tmp==uniq(1)) = -1; %convert whatever the learner returns
                if length(uniq)>1
                    v(tmp==uniq(2)) = +1; %into a -1/+1 representation
                end
                testingvotes(:,t) = v;
                clear v;
                
                %WEIGHT BY ALPHA VALUE
                weightedtestingvotes(:,t) = testingvotes(:,t) * model.alpha(t);
                
            end
            
            predictedlabels = adaboost.ssign( sum(weightedtestingvotes,2) );
            
        end%test
        
    end%methods
    
    
    methods (Static)
        
        % Strict Sign Function
        % Returns -1 if x<=0
        % Returns +1 if x>0
        %
        % Behaves the same as SIGN(INPUT), except that SSIGN(0) = -1. This enables
        % conversion from 0/1 to -1/1, as well as arbitrary tie-breaking in
        % a vote.
        function result = ssign(input)
            
            result = (input > 0) * 2 - 1;
            
        end
        
    end
    
    
end%classdef

