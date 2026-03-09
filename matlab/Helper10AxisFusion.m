classdef Helper10AxisFusion < handle
%   This class is for internal use only. It may be removed in the future. 
%HELPER10AXISFUSION Helper methods for 10-axis fusion example 

%   Copyright 2018 The MathWorks, Inc.    

    properties (Access = protected)
        viewer = [];
        firstView = true;
    end

    methods
        function view(obj, p1,q1,p2,q2)
            %PoseViewer with viewing angle adjustment
            if isempty(obj.viewer)
                obj.viewer = HelperPoseViewer(...
                    'XPositionLimits', [-23 23], ...
                    'YPositionLimits', [-23 23], ...
                    'ZPositionLimits', [-5 5]);
                obj.firstView = true;
            end 
            
            if obj.firstView
                ap = obj.viewer.AppWindow;
                c = get(ap, 'Children');
                for ii=1:numel(c)
                    if isa(c(ii), 'matlab.graphics.axis.Axes') && ...
                            contains(c(ii).Title.String, "Position")
                        view(c(ii), [0 1 0]);
                    end
                end
                obj.firstView = false;
            end
            
            obj.viewer([ 0 0 p1],q1,[0 0 p2],q2);     
        end
    end

    methods (Static)
        function y = growAmplitude(x)
        % Increases sine wave amplitude at zero crossings 
            % find zero crossings
            sg = sign(x);
            d = diff(sg);
            zc = find(d);
            
            % increase amplitude at zero crossings
            a = 1;
            amp = zeros(size(x));
            for ii=1:numel(zc)-1
                amp(zc(ii):zc(ii+1)) = a;
                a = a*1.5;
            end
            amp(zc(ii):end) = a;
            y = amp.*x;
        end

        function plotErrs(actP, actQ, expP, expQ)
            % Plot 10-axis fusion estimation errors
            figure('Name', 'Estimation Errors', 'NumberTitle', 'off');
            subplot(2,1,1);
            plot(rad2deg(dist(expQ, actQ)));
            title("Orientation Error - Quaternion Distance");
            ylabel("degrees")
            subplot(2,1,2);
            plot(actP - expP);
            title("Z Position Error");
            ylabel("meters");
        end        
    end
    
end
