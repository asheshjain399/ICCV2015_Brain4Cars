function evidence = iocalculateEvidence(model, data, inputObs)

    if strcmp(model.type,'gauss')
        for i = 1:size(data,2)
            data_sample = data{i};
            T = size(data_sample,2);
            evidence_sample = zeros(model.nstates,T);
            for j = 1:model.nstates
                if isfield(model,'aparam')
                    model.a = model.aparam{j};
                    model.b = model.bparam{j};
                elseif ~isfield(model,'a')
                    model.a = [0.5;-0.5;1.0;0];
                end;
                mult_const = (1.0 + sum(repmat(model.a,1,T).*inputObs{i},1) + [0 , sum(repmat(model.b,1,T-1).*data_sample(:,1:T-1),1)])';
                mu = model.mu{j};
                MU = repmat(mult_const,1,model.observationDimension).*repmat(mu',T,1);
                sigma = model.sigma{j};
                probability = mvnpdf(data{i}',MU,sigma);
                evidence_sample(j,:) = probability';
            end;
            evidence{i} = evidence_sample;
        end;
    elseif strcmp(model.type,'discrete')
        for i = 1:size(data,2)
            evidence{i} = model.B(:,data{i});
        end;
    end;
end
