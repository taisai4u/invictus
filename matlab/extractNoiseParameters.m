function [Q,R] = extractNoiseParameters(vector)
Q = vector(1).^2;
R = vector(2).^2;
end