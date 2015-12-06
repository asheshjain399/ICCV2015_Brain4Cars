clear all; clc;

addpath ../facetrack;
addpath ../sift;
addpath '/home/ashesh/project/Brain4Cars/Software/data/labels_face_cam';

load('path_to_videos.mat');

path_face_cam_multiple_labels = {'/home/ashesh/project/Brain4Cars/Software/data/labels_face_cam',...
    '/media/ssd2/assistive_driving/shane/DCIMA/NORMAL/train_labels' };

path_face_cam_multiple_labels = {'/media/ssd2/assistive_driving/hema/DCIMA/NORMAL/train_labels',...
    '/media/ssd2/assistive_driving/nadia/DCIMA/NORMAL/train_labels',...
    '/media/ssd2/assistive_driving/driver_7/DCIMA/NORMAL/train_labels',...
    '/home/ashesh/project/Brain4Cars/Software/data/labels_face_cam',...
    '/media/ssd2/assistive_driving/shane/DCIMA/NORMAL/train_labels'
    };

path_to_features = '/home/ashesh/project/Brain4Cars/Software/data/features';
%actions = {'lturn'};
actions = {'lturn','rturn','lchange','rchange','end_action'};

offset_frames = 0;
delta_frames = 150;

create_params_file = false;
path_to_params = '/media/ssd2/assistive_driving/params';
if create_params_file
    for j = 1:size(path_face_cam_multiple_labels,2)
        path_face_cam_labels = path_face_cam_multiple_labels{j};
        for i = 1:size(actions,2)
            action = actions{i};
            unix(['mkdir -p ' path_to_params '/' action]);
            fname = [path_face_cam_labels '/' 'all_train_' action '.txt'];
            fptr = fopen(fname);
            fgets(fptr);
            tline = fgets(fptr);
            while ischar(tline)
                tline = strtrim(tline);
                C = strsplit(tline,',');
                name = char(C(1)); frame = char(C(2)); time = char(C(3)); action_ = char(C(4)); frame_start = char(C(5)); frame_end = char(C(6));
                videoPath = time_to_dir(name(1:8));
                videoFilename = name;
                fe = str2num(frame_end);
                fs = str2num(frame_start);
                [checkPass,params] = createParams_for_extractFeatures(videoPath,videoFilename,fs,fe,action);
                if checkPass
                    save([path_to_params '/' action '/params_' name '_' frame_start '_' frame_end '.mat'],'params');
                    disp(['Saving ' videoFilename]);
                else
                    disp(['Skipping ' videoFilename]);
                end;
                tline = fgets(fptr);
            end;

        end;
    end;
end;

for j = 1:size(path_face_cam_multiple_labels,2)
    path_face_cam_labels = path_face_cam_multiple_labels{j};
    for i = 1:size(actions,2)
        action = actions{i};
        unix(['mkdir -p ' path_to_features '/' action]);
        fname = [path_face_cam_labels '/' 'all_train_' action '.txt'];
        fptr = fopen(fname);
        fgets(fptr);
        tline = fgets(fptr);
        while ischar(tline)
            disp(tline);
            tline = strtrim(tline);
            C = strsplit(tline,',');
            name = char(C(1)); frame = char(C(2)); time = char(C(3)); action_ = char(C(4)); frame_start = char(C(5)); frame_end = char(C(6));
            videoPath = time_to_dir(name(1:8));
            videoFilename = name;

            matfile_save_name = [path_to_features '/' action '/' videoFilename '_fs_' frame_start '_fe_' frame_end '_frames_' num2str(delta_frames) '_offset_' num2str(offset_frames) '.mat'];
            
            params_file = [path_to_params '/' action '/params_' name '_' frame_start '_' frame_end '.mat'];
            if exist(params_file,'file') == 2 && strcmp(name,'20141019_154641') 
                load(params_file);
                fe = str2num(frame_end) - offset_frames;
                fs = fe - delta_frames;
                
                if fs >= params.frame_start  && fe <= params.frame_end
                    params.frame_start = fs;
                    params.frame_end = fe;
                    features = extractFeatures( params );
                    save(matfile_save_name,'features');
                end;
                clear params;
            end;
            tline = fgets(fptr);
        end;
        fclose(fptr);
    end;
end;
