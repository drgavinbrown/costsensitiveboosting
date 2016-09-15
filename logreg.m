%
% LogReg
%
% Implements a 2-class Logistic Regression classifier.
% Trains by simple gradient descent in batch updates.
%
% Train/Test with default iterations/learning rate:
%  model = logreg().train(data,labels);
%  predictedlabels = model.test(data,labels);
%
% Or specify yourself:
%  model = logreg(100, 0.01).train(data,labels);
%
% Default learning rate = 0.1, iterations = (10 x numFeatures);
%
classdef logreg < handle
    
    properties
        
        w;
        learningrate = 0.1;
        iter;
        labelalphabet;
        
        trained = false;
        
    end
    
    methods
        
        function newmodel = clone(model)
            
            p = properties(model);
            newmodel = logreg();
            for i=1:length(properties(model))
                newmodel.(p{i}) = model.(p{i});
            end
            
        end
        
        function model = logreg(iter, alpha)
                        
            if ~exist('iter','var')
                model.iter = -1;
            else
                if ~isnumeric(iter) || iter<0
                    error('Number of iterations must be a positive integer');
                end
                model.iter = iter;
            end
            
            if ~exist('alpha','var')
                model.learningrate = 0.1;
            else
                model.learningrate = alpha;
            end
            
        end
        
        function model = train(model, data, labels)

            if model.iter == -1
                model.iter = 10*size(data,2);
            end
               
            if length(unique(labels))==1
                error('All labels in your training data are the same!  There is nothing to learn!');
            end
            
            N = size(data,1);
            J = size(data,2);
            numclasses = length(unique(labels));
            
            if numclasses>2
                error('This logistic regression model only handles 2 classes.');
            end
            
            L=1; % one output, for the two-class problem
            
            model.labelalphabet = unique(labels);
            
            if ~model.trained
                %error('Model already trained.');
                model.w = 2*rand(J+1,L)-1; % (J+1)xL (bottom row is bias weight)
            end
            x = data';  % x is now a JxN matrix

            y = labels';
            y(y~=1)=0; % y is now a 1xN matrix from {0,1}
            
              
            for t=1:model.iter
                
                %FORWARD PASS (BATCH, ALL EXAMPLES)
                a   = model.w'*[x;-ones(1,N)];         % Lx(J+1) * (J+1)xN = LxN
                
                out = 1./(1+exp(-a));                  % logistic outputs (1xN matrix)
                                
                %BACKWARD PASS
                update = - model.learningrate*( [x;-ones(1,N)] * (out-y)' );  % (K+1)xL matrix
                model.w = model.w + update;
                
            end
            
            model.trained = true;
            
        end%train
        
        
        function predictedlabels = test(model, data, labels)

            numfeatures = size(data,2);
            
            if size(model.w,1)-1 ~= numfeatures
                error(['Model has ' num2str(model.numinputs) ' inputs, while task has ' num2str(numfeatures) '.']);
            end

            N = size(data,1);    %numexamples

            x = data';       % x is now a JxN matrix

            %FORWARD PASS (BATCH, ALL EXAMPLES)
            a   = model.w'*[x;-ones(1,N)];  % Lx(J+1) * (J+1)xN = LxN
            
            out = 1./(1+exp(-a));           % simple logreg (LxN)
            
            guess = 1+double(out>0.5);      % guess in {1,2}
            
            %Take {1,2,..,C} output from this LogReg and convert
            % to whatever class labels the task is using.
            uniqguess = unique(guess);
            if length(uniqguess)==1
                uniqguess=1:length(model.labelalphabet);
            end
            
            uniqtruelabels = model.labelalphabet;
            predictedlabels = guess';
            for c=1:length(uniqguess)
                predictedlabels(guess==uniqguess(c)) = uniqtruelabels(c);
            end
            
            
  
        end%test
        
        
    end%methods
    
end%classdef
