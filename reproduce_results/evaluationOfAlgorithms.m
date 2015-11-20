%run_experiment_for = {'lane','turns','all'};
%clear all;clc
%run_experiment_for = {'all'};
%algorithms = {'HMM_I','HMM_I_O','IOHMM_I_O','AIOHMM_I_O'};
%algorithms = {'AIOHMM_I_O'};

function evaluationOfAlgorithms(algo,test_setting)

% 'algo' can take one of the following values {'HMM_I','HMM_I_O','IOHMM_I_O','AIOHMM_I_O'}
% 'test' can take one of the following values {'lane','turns','all'}

addpath ../Utils;
run_experiment_for = {test_setting};
algorithms = {algo};

experimentActions = containers.Map;
algorithmSettings = containers.Map;


actions = {'lchange','rchange','end_action'};
saveName = {'L Lane','R Lane','Straight'};
maneuvers.actions = actions;
maneuvers.saveName = saveName;
experimentActions('lane') = maneuvers;

actions = {'lturn','rturn','end_action'};
saveName = {'L Turn','R Turn','Straight'};
maneuvers.actions = actions;
maneuvers.saveName = saveName;
experimentActions('turns') = maneuvers;

actions = {'lchange','rchange','lturn','rturn','end_action'};
saveName = {'L Lane','R Lane','L Turn','R Turn','Straight'};
maneuvers.actions = actions;
maneuvers.saveName = saveName;
experimentActions('all') = maneuvers;

%% All
setting.THRESH = 0.6;
algorithmSettings('all_HMM_O') = setting;

setting.THRESH = 0.6;
algorithmSettings('all_HMM_I') = setting;

setting.THRESH = 0.9;
algorithmSettings('all_HMM_I_O') = setting;

setting.THRESH = 0.8;
algorithmSettings('all_IOHMM_I_O') = setting;

setting.THRESH = 0.9;
algorithmSettings('all_AIOHMM_I_O') = setting;


%% Lane Change
setting.THRESH = 0.6;
algorithmSettings('lane_HMM_O') = setting;

setting.THRESH = 0.5;
algorithmSettings('lane_HMM_I') = setting;

setting.THRESH = [0.93];
algorithmSettings('lane_HMM_I_O') = setting;

setting.THRESH = 0.8;
algorithmSettings('lane_IOHMM_I_O') = setting;

setting.THRESH = 0.89;
algorithmSettings('lane_AIOHMM_I_O') = setting;

%% Turns
setting.THRESH = 0.3;
algorithmSettings('turns_HMM_O') = setting;

setting.THRESH = 0.6;   %0.44;
algorithmSettings('turns_HMM_I') = setting;

setting.THRESH = 0.3;
algorithmSettings('turns_HMM_I_O') = setting;

setting.THRESH = 0.6;
algorithmSettings('turns_IOHMM_I_O') = setting;

setting.THRESH = 0.87;
algorithmSettings('turns_AIOHMM_I_O') = setting;


for i = 1:size(run_experiment_for,2)
    disp(['Running experiment for ' run_experiment_for{i}]);
    for j = 1:size(algorithms,2)
        experiment = run_experiment_for{i};
        algorithm = algorithms{j};
        matfiles_prefix = ['trainedModels/' experiment '/' algorithm];
        maneuvers = experimentActions(experiment);
        actions = maneuvers.actions;
        saveName = maneuvers.saveName;
        setting = algorithmSettings([experiment '_' algorithm]);
        THRESHOLD = setting.THRESH;
        p = [];
        r = [];
        t = [];
        f = [];
        p_std = [];
        r_std = [];
        t_std = [];
        t_std_err = [];
        p_std_err = [];
        r_std_err = [];
        f_std_err = [];
        for k = 1:size(THRESHOLD,2)
            THRESH = THRESHOLD(k);
            results = predictActions(algorithm,matfiles_prefix,THRESH,actions,saveName);
            %visualizeConfMat( results.confMat_normalized_precision,saveName,[matfiles_prefix '_precision.jpg'] );
            %saveConfMatFile( results.confMat_normalized_precision,saveName,[matfiles_prefix '_precision.csv'] );
            p = [p results.precision];
            r = [r results.recall];
            t = [t results.time];
            f = [f results.fpp];
            p_std = [p_std results.precision_std];
            r_std = [r_std results.recall_std];
            t_std = [t_std results.time_std];
            p_std_err = [p_std_err results.precision_std_err];
            r_std_err = [r_std_err results.recall_std_err];
            t_std_err = [t_std_err results.time_std_err];
            f_std_err = [f_std_err results.fpp_std_err];
        end;
        %disp([algorithm ': ' num2str(f(1)) ' (' num2str(f_std_err(1)) ')']);
        %{
        disp([algorithm ': ' num2str(f(1))]);
        %}
        
        %disp(results.p_timeseries);
        %disp(results.r_timeseries);
        disp('F1-score for Figure 8 in the paper');
        disp(2*results.p_timeseries.*results.r_timeseries./(results.p_timeseries+results.r_timeseries));
        
        disp('Precision in Table 1');
        disp(p);
        disp(p_std_err);
        disp('Recall in Table 1');
        disp(r);
        disp(r_std_err);
        disp('Time  in Table 1');
        disp(t);

        %disp('F1 Score')
        %disp(2*(p.*r)./(p+r));
        %f1 = 2*(p.*r)./(p+r);
        %save(['f1_scores/' algorithm '.mat'],'f1','p','r');
        %}
        disp('FPP in Table 2');
        disp(f);
        disp(f_std_err);
        %disp(t_std_err);
        
        %[precision, recall, fpp, confusionMat, timeSeriesPrediction]
    end;
end;