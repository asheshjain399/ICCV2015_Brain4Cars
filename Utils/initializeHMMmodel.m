function model = initializeHMMmodel(type,nstates,ostates,iotype,istates)
    model.nstates = nstates;
    model.type = type;
    model.iotype = iotype;
    if strcmp(type,'discrete')
        model.ostates = ostates;
        model.pi = (1.0/nstates)*ones(nstates,1);
        model.A = rand(nstates,nstates);
        model.A = model.A./repmat(sum(model.A,2),1,nstates);
        model.B = (1.0/ostates)*ones(nstates,ostates);
    elseif strcmp(type,'gauss') && ~iotype
        model.observationDimension = ostates;
        model.pi = (1.0/nstates)*ones(nstates,1);
        model.A = rand(nstates,nstates);
        model.A = model.A./repmat(sum(model.A,2),1,nstates);
        for i = 1:model.nstates
            model.mu{i} = rand(model.observationDimension,1);
            model.sigma{i} = (1.0/model.observationDimension)*eye(model.observationDimension,model.observationDimension); % rbfK(rand(),model.observationDimension);
        end;
    elseif strcmp(type,'gauss') && iotype
        model.observationDimension = ostates;
        model.inputDimension = istates;
        model.piW = rand(nstates-1,istates);
        model.W = rand(nstates,nstates-1,istates);
        for i = 1:model.nstates
            model.aparam{i} = [0.5 ; -0.5 ; 1.0 ; 0];
            model.mu{i} = rand(model.observationDimension,1);
            model.sigma{i} = (1.0/model.observationDimension)*eye(model.observationDimension,model.observationDimension); % rbfK(rand(),model.observationDimension);
        end;
    end;
end

function K = rbfK(gam,s)
    K = zeros(s,s);
    for i = 1:s
        for j = 1:s
            K(i,j) = normpdf(i-j,0,gam);
        end;
    end;
    K = K + (1.0/s)*eye(s,s);
    
end

