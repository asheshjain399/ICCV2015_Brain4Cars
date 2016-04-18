function generateObservationsCLM()
    %% Generate the features for HMM = Observations and Input
    
    clc;
   
    %actions = {'lchange','rchange','end_action'};
    actions = {'lturn','rturn','lchange','rchange','end_action'};
    observationDir = '../train/features';
    feature_type = 16;
    delta_frames = 20;
    window_width = 20;
    main_dir = '../new_params';
    %failedCount = 1;
    %%
    for i = 1:size(actions,2)
        action = actions{i};
        data = {};
        inputObs = {};
        number = 1;
        curr_dir = [main_dir actions{i} '/'];
        param_list = dir(curr_dir);
        for j=3:length(param_list)
            params_file = [curr_dir param_list(j).name '/new_param_' param_list(j).name '.mat'];
            if exist(params_file,'file') == 2
                    load(params_file);
                    fs = param_new.frame_start;
                    fe = param_new.frame_end;
                     if ((fe - fs) < 20)
                         %disp(actions{i});
                         %param_list(j).name
                         %failedCount = failedCount + 1;
                         %disp('no tracking:');
                         %disp(count);
                         continue;
                     end
            
                    count = 1;
                    for k = fe:-delta_frames:fs
                        param_new.frame_end = k;                    
                        if k-window_width > fs
                            param_new.frame_start = k-window_width;
                        else
                            continue;
                        end;
                        features = extractFeatures(param_new);
                        try
                            data_sample(:,count) = returnFeatures(features,feature_type)';
                            input_sample(:,count) = [features.lane_features features.speed_features(3)]';
                        catch
                        end;
                        count = count + 1;
                    end;
                    %number
                    data{number} = fliplr(data_sample);
                    inputObs{number} = fliplr(input_sample);
                    number = number + 1;
             end;
        end
        disp(['saving ovservations:' action]);
        save([observationDir 'clm_' action '_f_' num2str(feature_type) '_ww_' num2str(window_width) '_df_' num2str(delta_frames) '.mat'],'data','inputObs');

    end
end

function feature_out = returnFeatures(features,feature_type)
     if feature_type == 1
        feature_out = [features.hist_angle_subframe features.hist_move_in_x_subframe features.hist_distance_subframe];
    elseif feature_type == 2
        feature_out = [features.hist_angle_subframe];
    elseif feature_type == 3
        feature_out = [features.hist_angle features.hist_move_in_x features.hist_distance];
    elseif feature_type == 4
        feature_out = [features.hist_angle];
    elseif feature_type == 5
        feature_out = [features.bbox_center.movement_x_positive features.bbox_center.movement_x_negative features.bbox_center.motion_angle features.bbox_center.net_displacement]; 
    elseif feature_type == 6
        feature_out = [features.hog];
    elseif feature_type == 7
        feature_out = [features.bbox_center.xcenter];
    elseif feature_type == 8
        feature_out = [features.unified_subframe_hist mean(features.mean_movement_x)];
    elseif feature_type == 9
        feature_out = [features.hist_mean_movement_x./norm(features.hist_mean_movement_x) mean(features.mean_movement_x)];
    elseif feature_type == 10
        feature_out = mean(features.mean_movement_x);
    elseif feature_type == 11
        feature_out = [features.unified_fullframe_hist mean(features.mean_movement_x)];
    elseif feature_type == 12
        feature_out = [features.hist_mean_movement_x./norm(features.hist_mean_movement_x) features.unified_fullframe_hist mean(features.mean_movement_x)];
    elseif feature_type == 13
        feature_out = [features.hist_mean_movement_x./norm(features.hist_mean_movement_x) features.unified_fullframe_hist mean(features.mean_movement_x) features.lane_features features.speed_features(3)];
    elseif feature_type == 14
        feature_out = [features.lane_features features.speed_features(3)];
    elseif feature_type == 15
        feature_out = [features.euler features.hist_mean_movement_x./norm(features.hist_mean_movement_x) features.unified_fullframe_hist mean(features.mean_movement_x) features.lane_features features.speed_features(3)];
    elseif feature_type == 16
        feature_out = [features.lane_features features.speed_features(3)];
    elseif feature_type == 17
        feature_out = [features.euler features.hist_mean_movement_x./norm(features.hist_mean_movement_x) features.lane_features features.speed_features(3)];
    elseif feature_type == 18
        feature_out = [features.euler features.hist_mean_movement_x./norm(features.hist_mean_movement_x) features.unified_fullframe_hist mean(features.mean_movement_x)];
    end;
end