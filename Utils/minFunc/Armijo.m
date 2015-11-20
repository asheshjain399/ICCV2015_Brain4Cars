function step_size = Armijo(funcObj,grad,direction_decent,data,parameters,step_size)
    c1 = 0.9;

    while funcObj(parameters + step_size*direction_decent,data) > funcObj(parameters,data) + c1*step_size*grad'*direction_decent
        step_size = 0.5*step_size;
    end;

end

