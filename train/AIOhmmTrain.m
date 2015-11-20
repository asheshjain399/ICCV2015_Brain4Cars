function AIOhmmTrain()
    %% Entry point for training HMM
    clc;
    
    addpath(genpath('../Utils'));
    
    train_model_for = 'turns';
    use_fixed_split = false;
    
    learnedModels = containers.Map;
    testData = containers.Map;
    trainData = containers.Map;
    testObs = containers.Map;
    trainObs = containers.Map;

    separate_actions = true;
    retrain_models = true;
    delta_frames = 20;

    if separate_actions
%        actions = {'lchange' ,'rchange','end_action'};
%        showName = {'L lane','R Lane','Straight'};
        if strcmp(train_model_for,'turns')
            actions = {'lturn' ,'rturn','end_action'};
            showName = {'L Turn','R Turn','Straight'};
            fixed_split_name = 'fold_turns_';
            thresh = 60;
        elseif strcmp(train_model_for,'lane')
            actions = {'lchange','rchange','end_action'};
            showName = {'L Lane','R Lane','Straight'};
            fixed_split_name = 'fold_io2_';
            thresh = 130;
        elseif strcmp(train_model_for,'all')
            actions = {'lchange' ,'rchange','lturn' ,'rturn','end_action'};
            showName = {'L Lane','R Lane','L Turn','R Turn','Straight'};
            thresh = 130;
            fixed_split_name = '';
        else
            actions = {'lchange' ,'rchange','end_action'};
            showName = {'L Lane','R Lane','Straight'};
            train_model_for = 'turns';
            fixed_split_name = 'fold_io2_';
            thresh = 130;
        end;
    else
        train_model_for = 'combined';
        actions = {'combined','end_action'};
    end;

    fname_ext = '_f_13_ww_20_df_20.mat';
    disp(fname_ext);
    folds_action = containers.Map;
    for i = 1:size(actions,2)
        action = actions{i};
        load(['features/' action fname_ext]);      
        size_fold = round(min(size(data,2),thresh)/5.0);
        rand_list = randperm(min(size(data,2),thresh));
        folds={};
        folds{1} = rand_list(1:1:size_fold);
        folds{2} = rand_list((size_fold+1):1:(2*size_fold));
        folds{3} = rand_list((2*size_fold+1):1:(3*size_fold));
        folds{4} = rand_list((3*size_fold+1):1:(4*size_fold));
        folds{5} = rand_list((4*size_fold+1):1:(min(thresh,size(data,2))));
        folds_action(action) = folds;
    end;
    
    if retrain_models
        for numiter = 1:5
            if use_fixed_split
                if exist(['features/' fixed_split_name num2str(numiter) '.mat'],'file') == 2
                    load(['features/' fixed_split_name num2str(numiter) '.mat']);
                else
                    use_fixed_split = false;
                end;
            end;
            learnedModels_ = {};
            testData_ = {};
            trainData_ = {};
            testObs_ = {};
            trainObs_ = {};
            parfor i = 1:size(actions,2)
                action = actions{i};
                
                if use_fixed_split
                    test_data = testData(action);
                    train_data = trainData(action);
                    test_obs = testObs(action);
                    train_obs = trainObs(action);
                else
                    folds = folds_action(action);
                    load_data = load(['features/' action fname_ext]);
                    data = load_data.data;
                    inputObs = load_data.inputObs;
                    test_data = data(folds{i});
                    test_obs = inputObs(folds{i});
                    train_idx = 1:min(thresh,size(data,2));
                    train_idx(folds{i}) = [];
                    train_data = data(train_idx);
                    train_obs = inputObs(train_idx);
                end;
                
                model.type = 'gauss';
                model.iotype = true;
                model.observationDimension = size(train_data{1},1);
                model.inputDimension = size(train_obs{1},1);
                if strcmp(action,'end_action')
                    model.nstates=3;
                elseif strcmp(action,'rchange')
                    model.nstates=6;
                elseif strcmp(action,'lchange')
                    model.nstates=5;
                elseif strcmp(action,'rturn')
                    model.nstates=3;
                elseif strcmp(action,'lturn')
                    model.nstates=3;
                elseif strcmp(action,'combined')
                    model.nstates=6;
                end;

                model = initializeHMMmodel(model.type,model.nstates,model.observationDimension,model.iotype,model.inputDimension);
                for k = 1:model.nstates
                    model.bparam{k} = (1.0/model.observationDimension)*zeros(model.observationDimension,1);
                end;
                model.action = action;
                model.learn_aparam = true;
                
                %model.a = [0.5;-0.5;1.0;0];


                model.prior.use = 1; % 1 use the prior, 0 not prior on parameters
                %% Prior values
                model.prior.k0 = 1;
                model.prior.mu0 = (1.0/model.observationDimension)*ones(model.observationDimension,1);
                model.prior.Psi = (1.0/model.observationDimension)*eye(model.observationDimension,model.observationDimension);


                [model,ll] = AIOhmmFit(model,train_data,train_obs,model);
                learnedModels_{i} = model;
                testData_{i} = test_data;
                trainData_{i} = train_data;
                testObs_{i} = test_obs;
                trainObs_{i} = train_obs;
            end;
            for i = 1:size(actions,2)
                action = actions{i};
                learnedModels(action) = learnedModels_{i};
                testData(action) = testData_{i};
                trainData(action) = trainData_{i};                
                testObs(action) = testObs_{i};
                trainObs(action) = trainObs_{i};                
            end;
            save(['features/learnedModel_' train_model_for '_aiohmm_' num2str(numiter) fname_ext],'learnedModels','testData','trainData','testObs',...
                'trainObs','folds_action');
            %{
            if separate_actions
                save(['observations/learnedModel_aio2_' num2str(numiter) fname_ext '.mat'],'learnedModels','testData','trainData','testObs','trainObs');
            else
                save(['observations/learnedModel_aio2_combined_' num2str(numiter) fname_ext '.mat'],'learnedModels','testData','trainData','testObs','trainObs');
            end;
            %}
        end;
    end;
    p=[];
    r=[];
    for THRESH = 0:0.1:1.0
        time_global = [];
       true_label = [];
    predicted_label = [];
    accuracy = []; 
        for numiter = 1:5
            load(['features/learnedModel_' train_model_for '_aiohmm_' num2str(numiter) fname_ext]);
            %{
            if separate_actions
                load(['observations/learnedModel_aio2_' num2str(numiter) fname_ext '.mat']);
            else
                load(['observations/learnedModel_aio2_combined_' num2str(numiter) fname_ext '.mat']);
            end;
            %}
            prediction_local = [];
            true_label_local = [];

            for i = 1:size(actions,2)
                action = actions{i};
                test_data = testData(action);
                test_obs = testObs(action);
                for j = 1:size(test_data,2)
                    true_label = [true_label,i];
                    true_label_local = [true_label_local,i];
                    prediction_probability = timeSeriesPrediction(learnedModels,test_data{j},test_obs{j},actions);      
                    [action_predict,time_before] = predictAction(prediction_probability,actions,delta_frames,THRESH);
                    predicted_label = [predicted_label action_predict];
                    prediction_local = [prediction_local action_predict];
                    time_global = [time_global time_before];
                end;
            end;
            disp([num2str(numel(find(prediction_local == true_label_local))) '/' num2str(size(true_label_local,2))]);
            accuracy = [accuracy numel(find(prediction_local == true_label_local))*100.0/size(true_label_local,2)];
        end;

        matching = find(predicted_label == true_label);
        time_global = time_global(matching);
        time_global(find(time_global == -1)) = [];
        disp(['Mean time = ' num2str(mean(time_global))]);
        confMat = confusionMatrix(predicted_label,true_label);
        %p = [p (confMat(1,1) + confMat(2,2))/sum(sum(confMat(:,1:2)))];
        %r = [r (confMat(1,1) + confMat(2,2))/sum(sum(confMat(1:2,:)))];
        pr = (confMat(1,1)*1.0/sum(confMat(:,1)));% + confMat(2,2)*1.0/sum(confMat(:,2)));
        re = (confMat(1,1)*1.0/sum(confMat(1,:))); % + confMat(2,2)*1.0/sum(confMat(2,:)));
        disp(['accuracy = ' num2str(mean(accuracy)) '(' num2str(std(accuracy)) ')']);
        
        p_lchange = confMat(1,1)*100.0/sum(confMat(:,1));
        p_rchange = confMat(2,2)*100.0/sum(confMat(:,2));
        p_end_act = confMat(3,3)*100.0/sum(confMat(:,3));
        disp(['precision lchange = ' num2str(p_lchange)]);
        disp(['precision rchange = ' num2str(p_rchange)]);
        disp(['precision end_act = ' num2str(p_end_act)]);
        
        r_lchange = confMat(1,1)*100.0/sum(confMat(1,:));
        r_rchange = confMat(2,2)*100.0/sum(confMat(2,:));
        r_end_act = confMat(3,3)*100.0/sum(confMat(3,:));
        disp(['recall lchange = ' num2str(r_lchange)]);
        disp(['recall rchange = ' num2str(r_rchange)]);
        disp(['recall end_act = ' num2str(r_end_act)]);
        
        disp(confMat);
        p = [p mean([p_lchange p_rchange])];
        r = [r mean([r_lchange r_rchange])];
    end;
    p = 0.01*p;
    r = 0.01*r;
    disp(p);
    disp(r);
    plot(p,r);
    xlim([0 1]);
    ylim([0 1]);
    
    
end

function [action_taken,time_before] = predictAction(probability_table,actions,delta_frames,THRESH)
    %THRESH = 0.7;
    time_before = -1;
    T = size(probability_table,1);
    [~,action_taken] = ismember('end_action',actions);
    for i = 1:T
        [val,idx] =  max(probability_table(i,:));
        if val > THRESH && idx ~= action_taken && probability_table(i,action_taken) < THRESH
            action_taken = idx;
            time_before = (T-idx)*delta_frames*1.0/25.0;
            return
        end;
    end;
end

function prediction_probability = timeSeriesPrediction(learnedModels,test_data,test_obs,actions)
    T = size(test_data,2);
    prediction_probability = zeros(T,size(actions,2));
    for j = 1:T 
        test = {};
        test{1} = test_data(:,1:j);
        obs{1} = test_obs(:,1:j);
        loglikelihood = zeros(1,size(actions,2));
        for i = 1:size(actions,2)
            action = actions{i};
            model = learnedModels(action);  
            ll = iomodelDataLoglikelihood(model,aiocalculateEvidence(model,test,obs),obs);
            loglikelihood(i) = ll{1}; 
        end;
        log_a_b = logTologOfSum(loglikelihood);
        prediction_probability(j,:) = exp(loglikelihood - log_a_b);
    end;
end

function log_a_b = logTologOfSum(loglikelihood)
    if size(loglikelihood,2) == 2
        v1 = loglikelihood(1);
        v2 = loglikelihood(2);
        log_a_b = logOfsum(v1,v2);
    else
        v1 = loglikelihood(1);
        v2 = loglikelihood(1,2:end);
        log_a_b = logOfsum(v1,logTologOfSum(v2));
    end;
end

function confMat = confusionMatrix(predict,actual)
    lchange = find(actual==1);
    rchange = find(actual==2);
    nchange = find(actual==3);
    confMat = [numel(find(predict(lchange)==1)),numel(find(predict(lchange)==2)),numel(find(predict(lchange)==3));...
        numel(find(predict(rchange)==1)),numel(find(predict(rchange)==2)),numel(find(predict(rchange)==3));...
        numel(find(predict(nchange)==1)),numel(find(predict(nchange)==2)),numel(find(predict(nchange)==3))];
end
