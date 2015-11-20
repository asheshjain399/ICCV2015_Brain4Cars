function results = predictActions(algorithm,matfiles,THRESH,actions,saveName)
    time_plot = {};
    time_global = [];
    true_label = [];
    predicted_label = [];
    accuracy = []; 
    delta_frames = 20;
    for t = 1:20
        time_plot{t}.predicted_label = [];
        time_plot{t}.true_label = [];
        time_plot{t}.time_global = [];
    end;

    for numiter = 1:5
        load([matfiles '_' num2str(numiter) '.mat']);
        prediction_local = [];
        true_label_local = [];

        for i = 1:size(actions,2)
            action = actions{i};
            test_data = testData(action);
            if exist('testObs','var') && isKey(testObs,action)
                test_obs = testObs(action);
            else
                test_obs = test_data;
            end;
            for j = 1:size(test_data,2)
                true_label = [true_label,i];
                true_label_local = [true_label_local,i];
                prediction_probability = timeSeriesPrediction(learnedModels,test_data{j},test_obs{j},actions,algorithm);      
                [action_predict,time_before] = predictAction(prediction_probability,actions,delta_frames,THRESH);
                predicted_label = [predicted_label action_predict];
                prediction_local = [prediction_local action_predict];
                time_global = [time_global time_before];

                [action_taken,time_before] = predictTimeSeriesAction(prediction_probability,actions,delta_frames,THRESH);
                action_taken = fliplr(action_taken);
                time_before = fliplr(time_before);
                for k = 1:size(action_taken,2)
                    time_plot{k}.predicted_label = [time_plot{k}.predicted_label action_taken(k)];
                    time_plot{k}.true_label = [time_plot{k}.true_label i];
                    time_plot{k}.time_global = [time_plot{k}.time_global time_before(k)];
                end;

            end;
        end;
        confMat_local = confusionMatrix(prediction_local,true_label_local);
        for k = 1:size(confMat_local,1)-1
            p_local(k) = confMat_local(k,k)*100.0/sum(confMat_local(:,k));
            r_local(k) = confMat_local(k,k)*100.0/sum(confMat_local(k,:));
        end;
        p_iter(numiter) = mean(p_local);        
        r_iter(numiter) = mean(r_local);
        f_iter(numiter) = sum(confMat_local(size(confMat_local,1),1:size(confMat_local,1)-1))*100.0/sum(confMat_local(size(confMat_local,1),:)); 

        %disp([num2str(numel(find(prediction_local == true_label_local))) '/' num2str(size(true_label_local,2))]);
        accuracy = [accuracy numel(find(prediction_local == true_label_local))*100.0/size(true_label_local,2)];
    end;

    matching = find(predicted_label == true_label);
    time_global = time_global(matching);
    time_global(find(time_global == -1)) = [];
    %disp(['THRESH = ' num2str(THRESH)]);
    %disp(['Mean time = ' num2str(mean(time_global)) '(' num2str(std(time_global)) ')' ]);
    %disp(['accuracy = ' num2str(mean(accuracy)) '(' num2str(std(accuracy)) ')']);
    
    confMat = confusionMatrix(predicted_label,true_label);
    for k = 1:size(confMat,1)
        p_actions(k) = confMat(k,k)*100.0/sum(confMat(:,k));
        r_actions(k) = confMat(k,k)*100.0/sum(confMat(k,:));
    end;
    %disp(confMat);    
    results.precision = mean(p_iter);
    results.recall = mean(r_iter);
    results.precision_std = std(p_iter);
    results.recall_std = std(r_iter);
    results.precision_std_err = std(p_iter)/sqrt(numel(p_iter));
    results.recall_std_err = std(r_iter)/sqrt(numel(r_iter));
    
    results.time = mean(time_global);
    results.time_std = std(time_global);
    results.time_std_err = std(time_global)/sqrt(numel(time_global));
    results.fpp = mean(f_iter);
    results.fpp_std = std(f_iter);
    results.fpp_std_err = std(f_iter)/sqrt(numel(f_iter));
    results.confMat = confMat;
    results.confMat_normalized = confMat./repmat(sum(confMat,2),1,size(actions,2)); 
    results.confMat_normalized_precision = confMat./repmat(sum(confMat,1),size(actions,2),1); 
    p = [];
    r = [];
    for t = 1:20
        matching = find(time_plot{t}.predicted_label == time_plot{t}.true_label);
        if numel(time_plot{t}.predicted_label) > 0
            confMat = confusionMatrix(time_plot{t}.predicted_label,time_plot{t}.true_label);
            for k = 1:size(confMat,1)-1
                p_actions(k) = confMat(k,k)*100.0/sum(confMat(:,k));
                r_actions(k) = confMat(k,k)*100.0/sum(confMat(k,:));
            end;
            p = [p mean(p_actions)];
            r = [r mean(r_actions)];
        end;
    end;
    results.p_timeseries = p;
    results.r_timeseries = r;
    %disp(results);
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
            time_before = (T-i)*delta_frames*1.0/25.0;
            return
        end;
    end;
end


function [action_taken,time_before] = predictTimeSeriesAction(probability_table,actions,delta_frames,THRESH)
    %THRESH = 0.7;
    time_before = -1;
    T = size(probability_table,1);   
    [~,end_action] = ismember('end_action',actions);
    for i = 1:T
        [val,idx] =  max(probability_table(i,:));
        if val > THRESH && idx ~= end_action && probability_table(i,end_action) < THRESH
            action_taken(i) = idx;
            time_before(i) = (T-i)*delta_frames*1.0/25.0;
        else
            action_taken(i) = end_action;
            time_before(i) = (T-i)*delta_frames*1.0/25.0;
        end;
        
    end;
end

function prediction_probability = timeSeriesPrediction(learnedModels,test_data,test_obs,actions,algorithm)
    
    
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
            model.save_and_print = false;
            
            if strcmp(algorithm,'IOHMM_I_O') || strcmp(algorithm,'AIOHMM_I_O') 
                tstart = tic;
                ll = iomodelDataLoglikelihood(model,iocalculateEvidence(model,test,obs),obs);
                telapsed = toc(tstart);
                %disp(telapsed);
            else
                ll = modelDataLoglikelihood(model,calculateEvidence(model,test));
            end;
            
            loglikelihood(i) = ll{1}; 
        end;
        usePrior = true;
        if usePrior
            if strcmp(algorithm,'IOHMM_I_O') || strcmp(algorithm,'AIOHMM_I_O') || strcmp(algorithm,'HMM_I_O')
                intersection = test_obs(3,1);
                [l,lidx] = ismember('lturn',actions);
                [r,ridx] = ismember('rturn',actions);
                if intersection == 0 && l == 1 && r == 1
                    loglikelihood(lidx) = loglikelihood(lidx) - 10.0*abs(loglikelihood(lidx));
                    loglikelihood(ridx) = loglikelihood(ridx) - 10.0*abs(loglikelihood(ridx));
                end;
                left_action = test_obs(1,1);
                [l,lidx] = ismember('lchange',actions);
                if left_action == 0 && l == 1
                    loglikelihood(lidx) = loglikelihood(lidx) - 10.0*abs(loglikelihood(lidx));
                end;
                right_action = test_obs(2,1);
                [r,ridx] = ismember('rchange',actions);
                if right_action == 0 && r == 1
                    loglikelihood(ridx) = loglikelihood(ridx) - 10.0*abs(loglikelihood(ridx));
                end;
            end;
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
    labels = unique(actual);
    numlabels = size(labels,2);
    idx = {};
    for i = 1:numlabels
        idx{i} = find(actual==labels(i)); 
    end;
    confMat = zeros(numlabels,numlabels);
    for i = 1:numlabels
        for j = 1:numlabels
            confMat(i,j) = numel(find(predict(idx{i})==labels(j)));
        end;
    end;
end