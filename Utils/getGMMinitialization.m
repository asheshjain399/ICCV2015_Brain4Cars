function [mu, sigma] = getGMMinitialization(train_data, nstates)

data_agg = [];
count = 1;

for i=1:length(train_data)
    tmp_data = train_data{i};
    for j = 1:size(tmp_data, 2)
        data_agg(:, count) = tmp_data(:,j);
        count = count + 1;
    end
end

data_agg = data_agg';

ftr_dim = size(data_agg, 2);
const_ftr = [];
j=1;
for i=1:ftr_dim
    std_dev = std(data_agg(:, i));
    if (std_dev<0.01)
        const_ftr(j) = i;
        j = j+1;
    end
end

for i=1:length(const_ftr)
    ftr_ind = const_ftr(i);
    data_agg(:, ftr_ind) = data_agg(:, ftr_ind) +  0.001*randn(size(data_agg, 1),1);
end

%data_agg(:, ftr_dim-1) = data_agg(:, ftr_dim-1) + 0.001*randn(size(data_agg, 1),1);
%data_agg(:, ftr_dim-2) = data_agg(:, ftr_dim-2) + 0.001*randn(size(data_agg, 1),1);
%data_agg(:, ftr_dim-3) = data_agg(:, ftr_dim-3) + 0.001*randn(size(data_agg, 1),1);

options = statset('Display','final', 'MaxIter', 1000);
gm = fitgmdist(data_agg,nstates,'Options',options, 'RegularizationValue',0.1);

mu = (gm.mu)';
sigma_ = gm.Sigma;

mu = num2cell(mu, 1);
sigma = cell(1, size(sigma_, 3));
for i=1:size(sigma_, 3)
     sigma{i} = sigma_(:, :, i);
end

end