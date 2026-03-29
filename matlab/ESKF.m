classdef ESKF
    properties
        x_nom
        P
        sig_an
        sig_wn
        sig_aw
        sig_ww
        g
    end

    methods
        function obj = ESKF(x_nom, P, sig_an, sig_wn, sig_aw, sig_ww, g)
            arguments
                x_nom (16,1) double
                P (15,15) double
                sig_an double
                sig_wn double
                sig_aw double
                sig_ww double
                g (3,1) double
            end
            obj.x_nom = x_nom;
            obj.P = P;
            obj.sig_an = sig_an;
            obj.sig_wn = sig_wn;
            obj.sig_aw = sig_aw;
            obj.sig_ww = sig_ww;
            obj.g = g;
        end

        function q = get_quat(obj)
            arguments (Output)
                q (1,1) quaternion
            end
            q = quaternion(obj.x_nom(7), obj.x_nom(8), obj.x_nom(9), obj.x_nom(10));
        end

        function R = get_rotation_matrix(obj)
            R = obj.get_quat().rotmat("point");
        end

        function F_x = get_F_x(obj, u, dt)
            arguments
                obj
                u (6,1) double
                dt double
            end
            a_m = u(1:3);
            w_m = u(4:6);
            a_b = obj.x_nom(11:13);
            w_b = obj.x_nom(14:16);
            R = obj.get_rotation_matrix();

            F_x = eye(15);
            F_x(1:3,4:6) = eye(3) .* dt;

            F_x(4:6,7:9) = -R * ESKF.skew(a_m - a_b) .* dt;
            F_x(4:6,10:12) = -R .* dt;

            F_x(7:9, 7:9) = quaternion((w_m-w_b).' .* dt, 'rotvec').rotmat("point").';
            F_x(7:9, 13:15) = -eye(3) .* dt;
        end

        function obj = predict(obj, u, dt)
            arguments
                obj
                u (6,1) double
                dt double
            end
            % construct process noise covariance matrix
            V_i = eye(3) .* obj.sig_an.^2 .* dt.^2;
            T_i = eye(3) .* obj.sig_wn.^2 .* dt.^2;
            A_i = eye(3) .* obj.sig_aw.^2 .* dt;
            O_i = eye(3) .* obj.sig_ww.^2 .* dt;
            Q_i = blkdiag(V_i, T_i, A_i, O_i);

            % update error-state covariance
            F_x = obj.get_F_x(u, dt);
            F_i = [zeros(3,12); eye(12)];
            obj.P = F_x * obj.P * F_x.' + F_i * Q_i * F_i.';

            % update nominal state
            R = obj.get_rotation_matrix();

            a_m = u(1:3);
            w_m = u(4:6);
            a_b = obj.x_nom(11:13);
            w_b = obj.x_nom(14:16);

            [p, v, q] = deal(obj.x_nom(1:3), obj.x_nom(4:6), obj.get_quat());
            obj.x_nom(1:3) = p + v .* dt + 0.5 * (R*(a_m-a_b) + obj.g) .* dt.^2;
            obj.x_nom(4:6) = v + (R*(a_m-a_b) + obj.g) .* dt;
            obj.x_nom(7:10) = compact(q * quaternion((w_m-w_b).' .* dt, 'rotvec')).';
        end

        function X_dx = get_X_dx(obj)
            [qw, qx, qy, qz] = deal(obj.x_nom(7), obj.x_nom(8), obj.x_nom(9), obj.x_nom(10));
            Q_dtheta = 0.5 * [-qx, -qy, -qz;
                               qw, -qz,  qy;
                               qz,  qw, -qx;
                              -qy,  qx,  qw];
            X_dx = zeros(16, 15);
            X_dx(1:6, 1:6) = eye(6);
            X_dx(7:10, 7:9) = Q_dtheta;
            X_dx(11:16, 10:15) = eye(6);
        end

        function H = get_inverse_rotation_H_x(obj, vec)
            arguments
                obj
                vec (3,1) double
            end
            [q0, p] = deal(obj.x_nom(7), obj.x_nom(8:10));
            H = zeros(3, 4);
            H(1:3,1) = 2 .* (q0 .* vec + cross(vec, p));
            H(1:3,2:4) = 2 .* (p.'*vec .* eye(3) + p*vec.' - vec*p.' + q0 .* ESKF.skew(vec));
        end

        function obj = update(obj, pred, z, R, H_x)
            arguments
                obj
                pred (:,1) double
                z (:,1) double
                R double
                H_x double
            end
            H = H_x * obj.get_X_dx();
            y = z - pred;
            S = H * obj.P * H.' + R;
            K = obj.P * H.' / S;

            % update error state
            x = K * y;
            A = eye(15) - K * H;
            obj.P = A*obj.P*A.' + K*R*K.';

            % inject error state into nominal state
            obj.x_nom(1:6) = obj.x_nom(1:6) + x(1:6);
            obj.x_nom(7:10) = compact(obj.get_quat() * quaternion(x(7:9).', 'rotvec')).';
            obj.x_nom(11:16) = obj.x_nom(11:16) + x(10:15);

            % reset error state to 0, adjust P to account for injection
            G = eye(15);
            G(7:9,7:9) = eye(3) - ESKF.skew(0.5 .* x(7:9));
            obj.P = G*obj.P*G.';
        end

        function [pos, quat] = pose(obj)
            pos = obj.x_nom(1:3);
            quat = obj.get_quat();
        end
    end

    methods (Static)
        function s = skew(v)
            s = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        end
    end
end
