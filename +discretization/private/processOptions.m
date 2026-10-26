function [values,floorValue] = processOptions(args,defaults)
values = defaults;
position = 1;
while position <= numel(args) && position <= numel(defaults)
    value = args{position};
    if (ischar(value) || (isstring(value) && isscalar(value))) && strcmpi(value,'ProbabilityFloor')
        break
    end
    if ~isempty(value)
        values{position} = value;
    end
    position = position+1;
end
floorValue = probabilityFloorOption(args(position:end));
end
