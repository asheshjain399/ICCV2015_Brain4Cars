%% Implementation of Input-Output HMM
% Author: Ashesh Jain
% Email: ashesh@cs.cornell.edu

function [model,ll] = IOhmmFit( truemodel, data, inputObs, model)
    numiter = 30;
    ll = zeros(numiter,1);
    for i = 1:numiter
        evidence = iocalculateEvidence(model, data, inputObs);
        [alph,bet,gam,xi,loglikelihood] = FBWpass(model, evidence, inputObs);
        ll(i) = sum(cell2mat(loglikelihood));
        sufficient_statistics = estep(gam,model,data,inputObs);
        model = mstep(gam,xi,sufficient_statistics,model,inputObs,data);
    end;
    figure, plot(ll);
    disp(sum(cell2mat(loglikelihood)));
    %[~,loglikelihood,~] = ForwardPass(truemodel, calculateEvidence(truemodel, data));
    %disp(sum(cell2mat(loglikelihood)));
end

function model = mstep(gam,xi,expected,model,inputObs,data_original)
    addpath(genpath('../minFunc_2012/'));
    N = size(gam,2);
  
    data.inputObs = inputObs;
    data.nstates = model.nstates;
    data.inputDimension = model.inputDimension;
    data.xi = xi;
    data.gam = gam;
    
    funcObj = @(parameters)completeObj0(parameters, data);
    x0 = reshape(model.piW',model.inputDimension*(model.nstates-1),1);
    
    %funcObj = @objective0;
    %funcGrad = @gradient0;
    
    options.Method = 'lbfgs';
    options.Display = 'none';
    options.maxFunEvals = 500;
    parameters = minFunc(funcObj, x0, options);
    
    %[~,parameters] = minimizeFunc(funcObj,funcGrad,reshape(model.piW',model.inputDimension*(model.nstates-1),1),data);
    model.piW = reshape(parameters,model.inputDimension,model.nstates-1)';
    
    %funcObj = @objective;
    %funcGrad = @gradient;
    
    nstates = model.nstates;
    inputDimension = model.inputDimension;
    Wt = cell(1,nstates);
    data_complete = cell(1,nstates);
    for i = 1:model.nstates
        Wt{i} = model.W(i,:,:);
        data.state = i;
        data_complete{i} = data;
    end;
    
    parfor i = 1:model.nstates
        funcObj = @(parameters)completeObj(parameters, data_complete{i});
        x0 = reshape(reshape(Wt{i},nstates-1,inputDimension)',inputDimension*(nstates-1),1);
        parameters = minFunc(funcObj, x0, options);
        %[~,parameters] = minimizeFunc(funcObj,funcGrad,reshape(reshape(Wt{i},nstates-1,inputDimension)',inputDimension*(nstates-1),1),data_complete{i});
        Wt{i} = reshape(parameters,inputDimension,nstates-1)';
    end;

    for i = 1:model.nstates
        model.W(i,:,:) = Wt{i};
    end;

    if strcmp(model.type ,'discrete')
        model.B = expected.observation_count./repmat(expected.state_frequency,1,model.ostates);
    elseif strcmp(model.type ,'gauss')
        for i = 1:model.nstates
            mu = expected.mean_vector(:,i)/expected.state_frequency_const(i);
            
            xmu = zeros(model.observationDimension,model.observationDimension);
            for x = 1:N
                if model.learn_aparam
                    model.a = model.aparam{i};
                else
                    model.a = [0.5;-0.5;1.0;0];
                end;
                lgamma = gam{x};
                mult_const = (1.0 + sum(repmat(model.a,1,size(lgamma,2)).*inputObs{x},1))';
                
                gamma_times_data = repmat(lgamma(i,:),model.observationDimension,1).*data_original{x};
                xmu = xmu + gamma_times_data*(repmat(mult_const,1,model.observationDimension).*repmat(mu',size(lgamma,2),1));
            end;
            
            %sigma = (expected.cov_matrix(:,:,i) - expected.state_frequency(i)*mu*mu')/expected.state_frequency(i);
            sigma = (expected.cov_matrix(:,:,i) + expected.state_frequency_const(i)*mu*mu' - xmu - xmu')/expected.state_frequency(i);

            if model.prior.use == 1
                c1 = model.prior.k0*expected.state_frequency(i)/(model.prior.k0 + expected.state_frequency(i));
                sigma = model.prior.Psi + sigma + c1*(model.prior.mu0-mu)*(model.prior.mu0-mu)';
                mu = (model.prior.k0*model.prior.mu0 + expected.state_frequency(i)*mu)/(model.prior.k0+expected.state_frequency(i));
            end;
            model.mu{i} = mu;
            model.sigma{i} = 0.5*(sigma + sigma');
            sig_inv_mu = (model.sigma{i})\model.mu{i};
            mu_sig_inv_mu = mu'*sig_inv_mu;
            if model.learn_aparam
                model.aparam{i} = (expected.cov_matrix_input(:,:,i))\(expected.cov_matrix_cross(:,:,i)*sig_inv_mu*(1.0/mu_sig_inv_mu) - expected.mean_vector_input(:,i));
            end;
        end;
    end;
end

function expected = estep(gam,model,data,inputObs)
    N = size(gam,2);
    expected.state_frequency = zeros(model.nstates,1);
    expected.state_frequency_const = zeros(model.nstates,1);
    for i = 1:N
        lgamma = gam{i};
        T = size(lgamma,2);
        
        mult_const = zeros(model.nstates,T);
        for j = 1:model.nstates
                if model.learn_aparam
                    model.a = model.aparam{j};
                else
                    model.a = [0.5;-0.5;1.0;0];
                end;
            mult_const(j,:) = (1.0 + sum(repmat(model.a,1,T).*inputObs{i},1));
        end;    
        %mult_const = (1.0 + sum(repmat(model.a,1,T).*inputObs{i},1));
        mult_const = mult_const.^2;
        %expected.state_frequency_const = expected.state_frequency_const + sum(lgamma.*repmat(mult_const,model.nstates,1),2);
        expected.state_frequency_const = expected.state_frequency_const + sum(lgamma.*mult_const,2);
        expected.state_frequency = expected.state_frequency + sum(lgamma,2);
    end;
    
    if strcmp(model.type,'gauss')
        expected.mean_vector = zeros(model.observationDimension,model.nstates);
        expected.mean_vector_input = zeros(model.inputDimension,model.nstates);
        expected.cov_matrix = zeros(model.observationDimension,model.observationDimension,model.nstates);
        expected.cov_matrix_input = zeros(model.inputDimension,model.inputDimension,model.nstates);
        expected.cov_matrix_cross = zeros(model.inputDimension,model.observationDimension,model.nstates);
        for i = 1:N
            lgamma = gam{i};
            T = size(lgamma,2);
            
            for j = 1:model.nstates
                if model.learn_aparam
                    model.a = model.aparam{j};
                else
                    model.a = [0.5;-0.5;1.0;0];
                end;
                mult_const = (1.0 + sum(repmat(model.a,1,T).*inputObs{i},1));
                gamma_times_data_const = repmat(mult_const,model.observationDimension,1).*repmat(lgamma(j,:),model.observationDimension,1).*data{i};
                gamma_times_data = repmat(lgamma(j,:),model.observationDimension,1).*data{i};
                gamma_times_input_data = repmat(lgamma(j,:),model.inputDimension,1).*inputObs{i};
                
                expected.mean_vector(:,j) = expected.mean_vector(:,j) + sum(gamma_times_data_const,2);
                expected.mean_vector_input(:,j) = expected.mean_vector_input(:,j) + sum(gamma_times_input_data,2);
                expected.cov_matrix(:,:,j) = expected.cov_matrix(:,:,j) + gamma_times_data*data{i}';
                expected.cov_matrix_input(:,:,j) = expected.cov_matrix_input(:,:,j) + gamma_times_input_data*inputObs{i}';
                expected.cov_matrix_cross(:,:,j) = expected.cov_matrix_cross(:,:,j) + gamma_times_input_data*data{i}';
            end;
        end;
    elseif strcmp(model.type,'discrete')
        expected.observation_count = zeros(model.nstates,model.ostates);
        for i = 1:N
            lgamma = gam{i};
            for j = 1:model.ostates
                expected.observation_count(:,j) = expected.observation_count(:,j) + sum(lgamma(:,find(data{i}==j)),2);
            end;
        end;
    end;
end

function [alph,bet,gam,xi,loglikelihood] = FBWpass(model, evidence, inputObs)
    %% Here we do the forward-backward pass
    [alph,loglikelihood,mult_const] = ForwardPass(model, evidence, inputObs);
    bet = BackwardPass(model, evidence,mult_const,inputObs);
    gam = cell(1,size(evidence,2));
    xi = cell(1,size(evidence,2));
    for i = 1:size(evidence,2)
        lgam = alph{i}.*bet{i};
        lgam = lgam./repmat(mult_const{i},model.nstates,1);
        gam{i} = lgam;
    end;
    
    for i = 1:size(evidence,2)
        levidence = evidence{i};
        linputObs = inputObs{i};
        xi_sample = zeros(model.nstates,model.nstates,size(levidence,2)-1);
        for j = 1:size(levidence,2)-1
            A = transitionMatrix(model.W,linputObs(:,j+1),model); 
            xi_sample(:,:,j) = A.*(alph{i}(:,j)*(levidence(:,j+1).*bet{i}(:,j+1))');
            val = sum(sum(xi_sample(:,:,j)));
            
            %assert(abs(val-1.0)<1e-6);
        end;
        xi{i} = xi_sample;
    end;
end

function [alph,loglikelihood,mult_const] = ForwardPass(model, evidence, inputObs)
    %% Here we do the forward pass
    alph = cell(1,size(evidence,2));
    loglikelihood = cell(1,size(evidence,2));    
    mult_const = cell(1,size(evidence,2));    
    for i = 1:size(evidence,2)
        levidence = evidence{i};
        linputObs = inputObs{i};
        alpha_sample = zeros(model.nstates,size(levidence,2));
        const = zeros(1,size(levidence,2));
        starting_prob = multiClassProbability(model.piW,linputObs(:,1));
        [alpha_sample(:,1), Z] = normalize_local(levidence(:,1).*starting_prob);
        Zlog = log(Z);
        const(1) = 1.0/Z;
        for j = 2:size(levidence,2)
            A = transitionMatrix(model.W,linputObs(:,j),model); 
            [alpha_sample(:,j), Z] = normalize_local(levidence(:,j).*(A'*alpha_sample(:,j-1)));
            Zlog = Zlog + log(Z);
            const(j) = 1.0/Z;
        end;
        alph{i} = alpha_sample;
        loglikelihood{i} = Zlog;
        mult_const{i} = const;
    end;
end

function bet = BackwardPass(model, evidence, mult_const, inputObs)
    %% Here we do the backward pass to calculate beta
    bet = cell(1,size(evidence,2));
    for i = 1:size(evidence,2)
        levidence = evidence{i};
        linputObs = inputObs{i};
        const = mult_const{i};
        beta_sample = zeros(model.nstates,size(levidence,2));
        beta_sample(:,end) = 1;
        beta_sample(:,end) = const(end)*beta_sample(:,end);
        for j = (size(levidence,2)-1):-1:1
            A = transitionMatrix(model.W,linputObs(:,j+1),model); 
            beta_sample(:,j) = A*(levidence(:,j+1).*beta_sample(:,j+1));
            beta_sample(:,j) = const(j)*beta_sample(:,j);
        end;
        bet{i} = beta_sample;
    end;
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

function [v,Z] = normalize_local(v)
    %% Normalize the vector v
    Z = sum(v);
    v = v/Z;
end

function [obj, grad] = completeObj(w, data)
    grad = gradient(w,data);
    obj = objective(w,data);
end

function [obj, grad] = completeObj0(w, data)
    grad = gradient0(w,data);
    obj = objective0(w,data);
end

function grad = gradient(w,data)
    %% Find gradient of j->i transition
    grad = zeros(size(w,1),1);
    inputObs = data.inputObs;
    xi = data.xi;
    gam = data.gam;
    state = data.state;
    nstates = data.nstates;
    inputDimension = data.inputDimension;
    for i = 1:size(inputObs,2)
        linputObs = inputObs{i};
        lxi = xi{i};
        lgamma = gam{i};
        for j = 1:size(linputObs,2)-1
            probability = multiClassProbability(reshape(w,inputDimension,nstates-1)',linputObs(:,j+1));
            lxi_ = lxi(:,:,j);
            assert(sum(lxi_(state,1:end)) - lgamma(state,j) < 1e-8);
            grad = grad + kron(lxi_(state,1:end-1)' - lgamma(state,j)*probability(1:end-1),linputObs(:,j+1));
        end;
    end;
    grad = -1.0*grad;
end

function obj = objective(w,data)
    %% Find objective function value of j->i transition
    obj = 0.0;
    inputObs = data.inputObs;
    xi = data.xi;
    state = data.state;
    nstates = data.nstates;
    inputDimension = data.inputDimension;
    for i = 1:size(inputObs,2)
        linputObs = inputObs{i};
        lxi = xi{i};
        for j = 1:size(linputObs,2)-1
            probability = multiClassProbability(reshape(w,inputDimension,nstates-1)',linputObs(:,j+1));
            lxi_ = lxi(:,:,j);
            obj = obj + sum(lxi_(state,:)'.*log(probability));
        end;
    end;
    obj = -1.0*obj;
end

function grad = gradient0(w,data)
    %% Find gradient of start state of IOHMM
    grad = zeros(size(w,1),1);
    inputObs = data.inputObs;
    gam = data.gam;
    nstates = data.nstates;
    inputDimension = data.inputDimension;
    for i = 1:size(inputObs,2)
        linputObs = inputObs{i};
        lgamma = gam{i};
        probability = multiClassProbability(reshape(w,inputDimension,nstates-1)',linputObs(:,1));
        grad = grad + kron(lgamma(1:end-1,1) - probability(1:end-1),linputObs(:,1));
        
    end;
    grad = -1.0*grad;
end

function obj = objective0(w,data)
    %% Find objective function value of start state of IOHMM
    obj = 0.0;
    inputObs = data.inputObs;
    gam = data.gam;
    nstates = data.nstates;
    inputDimension = data.inputDimension;
    for i = 1:size(inputObs,2)
        linputObs = inputObs{i};
        lgamma = gam{i};
        probability = multiClassProbability(reshape(w,inputDimension,nstates-1)',linputObs(:,1));
        obj = obj + sum(lgamma(:,1).*log(probability));
    end;
    obj = -1.0*obj;
end