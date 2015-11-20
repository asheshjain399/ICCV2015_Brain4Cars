function [ minObj,parameters ] = minimizeFunc (funcObj,funcGrad,parameters,data )

    %parameters = [0;0;-1];
    %funcGrad = @gradient;
    %funcObj = @objective;

    THRESH = 0.4;
    grad = funcGrad(parameters,data);
    startObj = funcObj(parameters,data);
    minObj = startObj;
    numiter = 1;
    
    while norm(grad) > THRESH && numiter < 200
        step_size = 1.0;
        direction_decent = -grad;

        step_size = Armijo(funcObj,grad,direction_decent,data,parameters,step_size);
        parameters = parameters + step_size*direction_decent;

        grad = funcGrad(parameters,data);
        newObj = funcObj(parameters,data);
        assert(newObj - minObj < 1e-2);
        minObj = newObj;
        numiter = numiter + 1;
        
        %disp(['Norm of gradient = ' num2str(norm(grad))]);
    end;
    assert(minObj <= startObj);
    
end

function val = objective(X)
    A = [1 0 0;
        0 1.4 0;
        0 0 3.3];

    B = [1 2 3]';
    C = 10;
    val = 0.5*X'*A*X - B'*X + C;
end

function grad = gradient(X)
    A = [1 0 0;
        0 1.4 0;
        0 0 3.3];

    B = [1 2 3]';
    grad = A*X - B;
end