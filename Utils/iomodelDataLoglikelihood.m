function loglikelihood = iomodelDataLoglikelihood(model, evidence, inputObs)
    alph = cell(1,size(evidence,2));
    loglikelihood = cell(1,size(evidence,2));    
    for i = 1:size(evidence,2)
        levidence = evidence{i};
        linputObs = inputObs{i};
        alpha_sample = zeros(model.nstates,size(levidence,2));
        starting_prob = multiClassProbability(model.piW,linputObs(:,1));
        [alpha_sample(:,1), Z] = normalize_local(levidence(:,1).*starting_prob);
        Zlog = log(Z);
        for j = 2:size(levidence,2)
            A = transitionMatrix(model.W,linputObs(:,j),model); 
            [alpha_sample(:,j), Z] = normalize_local(levidence(:,j).*(A'*alpha_sample(:,j-1)));
            Zlog = Zlog + log(Z);
        end;
        alph{i} = alpha_sample;
        loglikelihood{i} = Zlog;
    end;
end

function [v,Z] = normalize_local(v)
    Z = sum(v);
    v = v/Z;
end

function probability = multiClassProbability(W,U)
    %% Returns probability of each class Softmax regression
    potentials = [exp(W*U);1.0];
    Z = sum(potentials);
    probability = potentials./Z;
end

function A = transitionMatrix(W,U,model)
    %% Returns the state transition matrix for the given input U
    A = zeros(model.nstates,model.nstates);
    for i = 1:model.nstates
        probability = multiClassProbability(reshape(W(i,:,:),model.nstates-1,model.inputDimension),U);
        A(i,:) = probability';
    end;
end