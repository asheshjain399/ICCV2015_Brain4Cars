function features = extractFeatures( params )
    
    %{
        params:
            frame_start: 
            frame_end
            frame_data: Cells of frames
                box: Bounding box eg. [a, b, c, d] (1x4 array)
                match_next: Face points matching the next frame (2xN array)
                match_prev: Face points match the previous frame (2xN array)
        
        In order to access 1st frame_data do the following params.frame_data{1}
    %}


    %%% Histogram bins
    hist_angle_values = [-0.75*pi,-0.25*pi,0.25*pi,0.75*pi];
    hist_distance_values = [1.5,2.5,7.5,8.5,11.5];
    hist_distance_x = [-3.0,-1.0,1.0,3.0];

    %params.frame_rate=30;
    %frame_start = min(max_frames,round(params.frame_rate*T));
    %frame_end = min(max_frames,round(params.frame_rate*(T+deltaT)));
    
    frame_start = params.frame_start;
    frame_end = params.frame_end;

    if frame_end <= frame_start
        features = [];
        return;
    end;
    
    bbox_center = [];
    Time = [];
    %features = containers.Map;
    
    features_hist_angle = [];
    features_hist_distance = [];
    features_hist_move_in_x = [];
    features_mean_movement = [];
    features_mean_movement_x = [];
    
    features_hist_angle_subframe = [];
    features_hist_distance_subframe = [];
    features_hist_move_in_x_subframe = [];
    
    mean_frame_movement = [];
    mean_subframe_movement = {};
    iter = 1;
    transform = {};
    unified_subframe_hist = zeros(4,size(hist_angle_values,2));
    unified_fullframe_hist = zeros(1,size(hist_angle_values,2));
    
    if strcmp(params.laneInfo,'') || strcmp(params.laneInfo,'-1')
        lane_features = [1,1,0];
    else
        lane = strsplit(params.laneInfo,',');
        lane_no = str2num(lane{1});
        total_lanes = str2num(lane{2});
        intersection = str2num(lane{3});
        if total_lanes > lane_no
            left_action = 1;
        else
            left_action = 0;
        end;
        if lane_no > 1
            right_action = 1;
        else
            right_action = 0;
        end;
        lane_features = [left_action, right_action, intersection];
    end;
    overall_points = [];
    motion_vectors = [];
    time_instants = [];
    euler = zeros(1, 3);
    for i = frame_start:1:frame_end-1

        frame_data_cur = params.frame_data{i};    
        frame_data_next = params.frame_data{i+1};
        %disp(size(euler));
        %disp(size(frame_data_cur.Euler));
        if (strcmp(params.tracker,'CLM'))
            euler = euler + frame_data_cur.Euler';
        end
        %Time(i) = frame_data_cur.time;
        bbox_cur = frame_data_cur.box;
        bbox_next = frame_data_next.box;
        bbox_center(:,iter) = [bbox_cur(1)+0.5*bbox_cur(3); bbox_cur(2)+0.5*bbox_cur(4)];
        
        match_cur = frame_data_cur.match_next;
        match_next = frame_data_next.match_prev;
        
        speedInfo(iter) = frame_data_cur.speed;
        
        if size(match_next,2) == 0
            match_cur = bbox_center(:,iter);
            match_next = bbox_center(:,iter);
        end;

        if (strcmp(params.tracker,'KLT'))
            transform{iter} = frame_data_cur.T;
            if iter > 1
                transform{iter} = transform{iter-1}*frame_data_cur.T;
                if size(match_cur,2) < 20
                    transform{iter} = transform{iter-1};
                end;
            end;
        end;

        points_move_vec = match_next - match_cur;
        points_distance = sqrt(sum(points_move_vec.^2,1));
        points_angle = atan2(points_move_vec(2,:),points_move_vec(1,:));
        points_move_in_x = points_move_vec(1,:);
        overall_points = [overall_points points_angle];
        motion_vectors = [motion_vectors ; points_move_vec'];
        time_instants = [time_instants; repmat((i-frame_start)*1.0/25,size(points_move_vec,2),1)];
        
        sub_bbox = subBBoxes(bbox_cur,bbox_next);
        
        % Full frame features
        features_hist_angle_fullframe = hist(points_angle,hist_angle_values);
        features_hist_distance_fullframe = hist(points_distance,hist_distance_values);
        features_hist_move_in_x_fullframe = hist(points_move_in_x,hist_distance_x);
        
        unified_fullframe_hist = unified_fullframe_hist + features_hist_angle_fullframe;
        features_hist_angle = [features_hist_angle features_hist_angle_fullframe/max(1.0,norm(features_hist_angle_fullframe))];
        features_hist_distance = [features_hist_distance features_hist_distance_fullframe/max(1.0,norm(features_hist_distance_fullframe))];
        features_hist_move_in_x = [features_hist_move_in_x features_hist_move_in_x_fullframe/max(1.0,norm(features_hist_move_in_x_fullframe))];
        features_mean_movement = [features_mean_movement mean(points_distance)];
        features_mean_movement_x = [features_mean_movement_x mean(points_move_in_x)];
        
        % Sub-frame features
        for j = 1:size(sub_bbox,2)
            sub_bbox_cur = sub_bbox{j}.cur;
            sub_bbox_next = sub_bbox{j}.next;
            
            [Xv,Yv] = BBoxToXvYv(sub_bbox_cur);
            IN_cur = double(inpolygon(match_cur(1,:),match_cur(2,:),Xv,Yv));
            [Xv,Yv] = BBoxToXvYv(sub_bbox_next);
            IN_next = double(inpolygon(match_next(1,:),match_next(2,:),Xv,Yv));
            IN = IN_cur.*IN_next;
            
            features_hist_angle_ = hist(points_angle(find(IN==1)),hist_angle_values);
            features_hist_distance_ = hist(points_distance(find(IN==1)),hist_distance_values);
            features_hist_move_in_x_ = hist(points_move_in_x(find(IN==1)),hist_distance_x);
            
            unified_subframe_hist(j,:) = unified_subframe_hist(j,:) + features_hist_angle_;
            
            features_hist_angle_ = features_hist_angle_/max(1.0,norm(features_hist_angle_));
            features_hist_distance_ = features_hist_distance_/max(1.0,norm(features_hist_distance_));
            features_hist_move_in_x_ = features_hist_move_in_x_/max(1.0,norm(features_hist_move_in_x_));
            
            features_hist_angle_subframe = [features_hist_angle_subframe features_hist_angle_];
            features_hist_distance_subframe = [features_hist_distance_subframe features_hist_distance_];
            features_hist_move_in_x_subframe = [features_hist_move_in_x_subframe features_hist_move_in_x_];
            
        end;
        mean_frame_movement = [mean_frame_movement mean(points_move_vec,2)];        
        iter = iter + 1;
    end;
    if (strcmp(params.tracker,'CLM'))
        euler = euler/(frame_end - frame_start + 1);
        features.euler = euler;
    end
    unified_fullframe_hist = unified_fullframe_hist/max(1.0,norm(unified_fullframe_hist));
    unified_subframe_hist = unified_subframe_hist./repmat(max(ones(4,1),sum(unified_subframe_hist,2)),1,size(hist_angle_values,2));
    features.unified_subframe_hist = reshape(unified_subframe_hist,1,4*size(hist_angle_values,2));
    features.unified_fullframe_hist = unified_fullframe_hist;
    features.hist_angle = features_hist_angle;
    features.hist_distance = features_hist_distance;
    features.hist_move_in_x = features_hist_move_in_x;
    features.mean_movement = features_mean_movement;
    features.mean_movement_x = features_mean_movement_x;
    features.hist_mean_movement_x = hist(features_mean_movement_x,hist_distance_x);
    features.hist_angle_subframe = features_hist_angle_subframe;
    features.hist_distance_subframe = features_hist_distance_subframe;
    features.hist_move_in_x_subframe = features_hist_move_in_x_subframe;
    if (strcmp(params.tracker,'KLT'))
        features.face_transform = transform;
    end
    motion_vector = bbox_center(:,2:end) - bbox_center(:,1:end-1);
    features.bbox_center = BBoxCenterTrajectoryFeature(motion_vector, bbox_center(1,:));
    img = zeros(960,1);
    if strcmp(params.tracker,'KLT')
        face_mean_x_pos = mean(bbox_center,2);
        face_mean_x_pos = face_mean_x_pos(1);
        for i = frame_start:frame_end
            frame_data = params.frame_data{i}; 
            klt_points = frame_data.klt_points(1,:);
            klt_points = klt_points';
            img(round(klt_points - face_mean_x_pos + 300),i-frame_start+1) = 1;
        end;
        img = flipud(img);
        features.hog = double(extractHOGFeatures(img));
        features.hog_image = img;
    end;
    speedInfo(find(speedInfo == -1)) = [];
    if numel(speedInfo) == 0
        speed_features = [30.0/160 30.0/160 30.0/160];
    else
        speed_features = (1.0/160) * [max(speedInfo) min(speedInfo) mean(speedInfo)];
    end;
    features.speed_features = speed_features;
    features.lane_features = lane_features;
    Xmark = cos(overall_points);
    Ymark = sin(overall_points);
    
     %{
    unit_vector = motion_vectors./repmat(sqrt(sum(motion_vectors.^2,2)),1,2);
    time_to_plot = intersect(find(time_instants >=2.0),find(time_instants < 2.1));
    h=figure;compass(motion_vectors(time_to_plot,1),motion_vectors(time_to_plot,2));
    t = findall(gcf,'type','text');
    delete(t);
    print(h,'-dpng','sys.png');
    close all;
   
    time_to_plot = intersect(find(time_instants >=3.2),find(time_instants < 3.3));
    h=figure;compass(motion_vectors(time_to_plot,1),motion_vectors(time_to_plot,2));
    t = findall(gcf,'type','text');
    delete(t);
    print(h,'-dpng','d_t_3_2.png');
    close all;
    
    time_to_plot = intersect(find(time_instants >=4.2),find(time_instants < 4.3));
    h=figure;compass(motion_vectors(time_to_plot,1),motion_vectors(time_to_plot,2));
    t = findall(gcf,'type','text');
    delete(t);
    print(h,'-dpng','d_t_4_3.png');
    close all
    %}
end

function bbox_center = BBoxCenterTrajectoryFeature(motion_vector, bbox_xcenter)
    
    movement = sum(motion_vector,2);
    movement_x_positive = sum(motion_vector(1,find(motion_vector(1,:)>0)));
    movement_x_negative = sum(motion_vector(1,find(motion_vector(1,:)<0)));
    motion_angle = atan2(movement(2),movement(1));
    net_displacement = norm(movement);
    bbox_center.movement_x_positive = movement_x_positive;
    bbox_center.movement_x_negative = movement_x_negative;
    bbox_center.motion_angle = motion_angle;
    bbox_center.net_displacement = net_displacement;
    
    delta = max(bbox_xcenter) - min(bbox_xcenter);
    overlap = (delta*1.0/3.0)*(1.0/10.0);
    l1 = min(bbox_xcenter);
    l2 = l1 + delta*1.0/3.0;
    l3 = l1 + delta*2.0/3.0;
    
    hist_bbox_xcenter = [];
    width = [];
    
    idx = find(bbox_xcenter<(l2+overlap));
    width = [width (max(idx)-min(idx))];
    hist_bbox_xcenter = [hist_bbox_xcenter numel(idx)];
    
    idx = find(bbox_xcenter>(l2-overlap) & bbox_xcenter<(l3+overlap));
    width = [width (max(idx)-min(idx))];
    hist_bbox_xcenter = [hist_bbox_xcenter numel(idx)];
    
    idx = find(bbox_xcenter>(l3-overlap));
    width = [width (max(idx)-min(idx))];
    hist_bbox_xcenter = [hist_bbox_xcenter numel(idx)];
    
    width = width./max(width);
    hist_bbox_xcenter = hist_bbox_xcenter./sum(hist_bbox_xcenter);
    bbox_center.xcenter = [hist_bbox_xcenter width];
    
end

function [Xv,Yv] = BBoxToXvYv(bbox)
        Xv = [bbox(1), bbox(1)+bbox(3), bbox(1)+bbox(3), bbox(1)];
        Yv = [bbox(2), bbox(2), bbox(2)+bbox(4), bbox(2)+bbox(4)];
end

function sub_bbox = subBBoxes(bbox_cur,bbox_next)
    sub_bbox = {};
    w_cur = round(bbox_cur(3)*0.5);
    h_cur = round(bbox_cur(4)*0.5);
    w_next = round(bbox_next(3)*0.5);
    h_next = round(bbox_next(4)*0.5);
    
    sub_bbox{1}.cur = [bbox_cur(1) bbox_cur(2) w_cur h_cur];
    sub_bbox{1}.next = [bbox_next(1) bbox_next(2) w_next h_next];
    
    sub_bbox{2}.cur = [bbox_cur(1)+w_cur bbox_cur(2) w_cur h_cur];
    sub_bbox{2}.next = [bbox_next(1)+w_next bbox_next(2) w_next h_next];
    
    sub_bbox{3}.cur = [bbox_cur(1) bbox_cur(2)+h_cur w_cur h_cur];
    sub_bbox{3}.next = [bbox_next(1) bbox_next(2)+h_next w_next h_next];
    
    sub_bbox{4}.cur = [bbox_cur(1)+w_cur bbox_cur(2)+h_cur w_cur h_cur];
    sub_bbox{4}.next = [bbox_next(1)+w_next bbox_next(2)+h_next w_next h_next];
    
end

function visualizeFrame(frame,scale,bbox,points)
    frame = imresize(frame,scale);
    frame = insertObjectAnnotation(frame,'rectangle',bbox,'Face');
    points(3,:) = 0.02;
    frame = insertShape(frame,'circle',points','Color','green'); 
    figure, imshow(frame);
end