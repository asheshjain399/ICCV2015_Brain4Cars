function log_a_b = logOfsum( log_a, log_b )
%% Calculates Log of sum from sum of logs

% Input
% log(a) and log(b)
% Return
% log(a+b)

log_a_b = log(exp(log_a - log_b) + 1.0) + log_b;

end

