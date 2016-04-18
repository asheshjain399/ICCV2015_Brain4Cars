clear; clc;

%addpath '/home/ashesh/project/Brain4Cars/Software/data/labels_face_cam';

%load('path_to_videos.mat');

%path_face_cam_multiple_labels = {'/home/ashesh/project/Brain4Cars/Software/data/labels_face_cam',...
%    '/media/ssd2/assistive_driving/shane/DCIMA/NORMAL/train_labels' };

% path_face_cam_multiple_labels = {'/media/ssd2/assistive_driving/hema/DCIMA/NORMAL/train_labels',...
%     '/media/ssd2/assistive_driving/nadia/DCIMA/NORMAL/train_labels',...
%     '/media/ssd2/assistive_driving/driver_7/DCIMA/NORMAL/train_labels',...
%     '/home/ashesh/project/Brain4Cars/Software/data/labels_face_cam',...
%     '/media/ssd2/assistive_driving/shane/DCIMA/NORMAL/train_labels'
%     };

path_to_features = '../bulkFeatures/';
%actions = {'lturn'};

actions = {'lturn','rturn','lchange','rchange','end_action'};

offset_frames = 50;
delta_frames = 50;

main_dir = '../new_params/';


for i = 1:size(actions,2)
    action = actions{i};
    curr_dir = [main_dir actions{i} '/'];
    param_list = dir(curr_dir);
    for j=3:length(param_list)
            params_file = [curr_dir param_list(j).name '/new_param_' param_list(j).name '.mat'];
            matfile_save_name = [path_to_features '/' action '/' param_list(j).name '_frames_' num2str(delta_frames) '_offset_' num2str(offset_frames) '.mat'];

            if exist(params_file,'file') == 2
                    load(params_file);
                    fe = param_new.frame_end - offset_frames;
                    fs = fe - delta_frames;
                    if fs >= param_new.frame_start  && fe <= param_new.frame_end
                        %disp('hey, ssup?');
                        param_new.frame_start = fs;
                        param_new.frame_end = fe;
                        features = extractFeatures( param_new );
                        save(matfile_save_name,'features');
                    end;
            else
                disp('param file missing');
            end;
            clear param_new;
    end;
end;
