function res = Power_Allocation_RS(N, BLER_TH, R_Meta, R_Payload, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error)
%%%%%%%%%%%%%% Power allocation for Rate Splittiing %%%%%%%%%%%%%%%%%%%%%%

% Note that since we use the network availbility as the performance
% indicator. We directly use the target BLER as the bisection results
% because if the target BLER is achievable, then we mark this iteration as
% feasible. This process can avoid the bisection and improve the running
% speed. For who wanna see the point-to-point source code of Algorithm 3,
% please refer to Figure 5. Note that the results obtained from both
% version are same.

t = BLER_TH/2;

% Find the minimum value of SINR for metadata with a given target t (Bisection method)
Gamma_min_Meta = Bisection(N, R_Meta, t);

% Find the minimum value of SINR for payload with a given target t(Bisection method)
Gamma_min_Payload = Bisection(N, R_Payload, t);

% Initialized power coefficients
Pk_Meta = 0.5*Pt.*ones(L, tau_p);
Pk_Payload = 0.5*Pt.*ones(L, tau_p);

% Maximum iteration numbers
max_iter = 10;

% Prestore Z
Z_iter = zeros(max_iter, 1);
BLER_iter = zeros(max_iter, 1);

for iter = 1:max_iter

    % Update detectors
    [akl_Meta, ckl_Meta] = Detector_Generation_RS(K, L, KL, Mr, Pk_Payload, Pk_Meta, h_est, labels, Rcorr_error, Noise_power, tau_p, "Meta");
    [akl_Payload, ckl_Payload] = Detector_Generation_RS(K, L, KL, Mr, Pk_Payload, Pk_Meta, h_est, labels, Rcorr_error, Noise_power, tau_p, "Payload");

    %---------------------------------------------------------------------%
    % Accelarating CVX
    Mask = false(L, tau_p); % Create logical Mask
    for l = 1:L
        Mask(l, 1:KL(l)) = true;
    end

    G_Meta = zeros(L, tau_p); % Compute the signal gain for metadata
    G_Payload = zeros(L, tau_p); % Compute the signal gain for metadata
    for l = 1:L
        for k = 1:KL(l)
            G_Meta(l, k) = akl_Meta(l, k, l, k);
            G_Payload(l, k) = akl_Payload(l, k, l, k);
        end
    end

    total_dims = L * tau_p; % total number of variables
    A_mat_Meta    = reshape(akl_Meta, [total_dims, total_dims]);
    C_mat_Meta    = reshape(ckl_Meta, [total_dims, total_dims]);
    A_mat_Payload = reshape(akl_Payload, [total_dims, total_dims]);
    C_mat_Payload = reshape(ckl_Payload, [total_dims, total_dims]);

    %---------------------------------------------------------------------%
    % CVX SOLVER
    cvx_begin quiet
    cvx_solver mosek
        variable A_Meta(L, tau_p) nonnegative
        variable A_Payload(L, tau_p) nonnegative
        variable Pk_Meta(L, tau_p) nonnegative
        variable Pk_Payload(L, tau_p) nonnegative

        Pk_Total = Pk_Meta + Pk_Payload; % Compute total power (L x tau_p)
        P_vec = reshape(Pk_Total, [total_dims, 1]); % P_vec( (kl_pr-1)*L + l_pr ) corresponds to Pk_Total(l_pr, kl_pr)

        % Compute total interference
        Total_I_A_Meta_vec    = (P_vec' * A_mat_Meta)'; % total interference
        Total_I_C_Meta_vec    = (P_vec' * C_mat_Meta)';
        Total_I_A_Payload_vec = (P_vec' * A_mat_Payload)';
        Total_I_C_Payload_vec = (P_vec' * C_mat_Payload)';

        Total_I_A_Meta    = reshape(Total_I_A_Meta_vec, [L, tau_p]); % Reshape total interference matrices (L x tau_p)
        Total_I_C_Meta    = reshape(Total_I_C_Meta_vec, [L, tau_p]);
        Total_I_A_Payload = reshape(Total_I_A_Payload_vec, [L, tau_p]);
        Total_I_C_Payload = reshape(Total_I_C_Payload_vec, [L, tau_p]);
        
        % Compute true interference plus noise for metadata
        I11_I12 = Total_I_A_Meta - Pk_Total .* G_Meta;         % I11 + I12 = total interference - self-interference
        I13 = Total_I_C_Meta;
        B_Meta = I11_I12 + I13 + Pk_Payload .* G_Payload + Noise_power;

        % Compute true interference plus noise for payload
        I21_I22 = Total_I_A_Payload - Pk_Total .* G_Payload;
        I23 = Total_I_C_Payload;
        B_Payload = I21_I22 + I23 + Noise_power;
        
        % Objective function
        Z = (sum(A_Meta(Mask)) + sum(A_Payload(Mask)));

        minimize Z
        subject to
            % SINR constraints
            % Note that we multiply by a very large number for numerical precision 
            % and to ensure a stable solution; this will not affect the result.
            1e12 .* (Pk_Meta(Mask) .* G_Meta(Mask) - Gamma_min_Meta .* B_Meta(Mask)) + A_Meta(Mask) == 0; % 52a
            1e12 .* (Pk_Payload(Mask) .* G_Payload(Mask) - Gamma_min_Payload .* B_Payload(Mask)) + A_Payload(Mask) == 0; % 52b

            % Power constraint
            Pk_Total(Mask) <= Pt; % 32e

            % Set all non-active users' power to 0
            Pk_Meta(~Mask) == 0;
            Pk_Payload(~Mask) == 0;
            A_Meta(~Mask) == 0;
            A_Payload(~Mask) == 0;
    cvx_end
    %---------------------------------------------------------------------%
    Pk_Meta(Pk_Meta < 0) = 0;
    Pk_Payload(Pk_Payload < 0) = 0;
    Z_iter(iter) = Z;
    if iter >= 2 && abs(Z_iter(iter) - Z_iter(iter - 1))/Z_iter(iter) <= 0.1
        break
    end
end

% Compute SINRs as per the obtained power coefficients to double check the results
[Gamma_Meta, ~] = Compute_Y(L, tau_p, Pk_Meta, Pk_Payload, akl_Meta, ckl_Meta, Noise_power, KL, R_Meta, N, "Meta");
[Gamma_Payload, ~] = Compute_Y(L, tau_p, Pk_Meta, Pk_Payload, akl_Meta, ckl_Meta, Noise_power, KL, R_Payload, N, "Payload");

if min(Gamma_Meta(:)) >= Gamma_min_Meta - 1e-3 && min(Gamma_Payload(:)) >= Gamma_min_Payload - 1e-3 % feasible
    res = 1;
else % infeasible
    res = 0;
end

end

%% Local Function for Computing SINR
function [Gamma, BLER] = Compute_Y(L, tau_p, Pk_Meta, Pk_Payload, akl, ckl, Noise_power, KL, R, N, flag)
V = @(gamma) 2.*gamma./(1 + gamma);
f = @(gamma) sqrt(log(2)*N./V(gamma)).*(log2(1 + gamma) - R);

Gamma = nan(L, tau_p);
BLER = nan(L, tau_p);
B = zeros(L, tau_p);

I1 = zeros(L, tau_p); I2 = zeros(L, tau_p); I3 = zeros(L, tau_p);

for l = 1:L
    for kl = 1:KL(l)
        for l_pr = 1:L
            for kl_pr = 1:KL(l_pr)
                if kl_pr ~= kl
                    I1(l, kl) = I1(l, kl) + (Pk_Meta(l_pr, kl_pr) + Pk_Payload(l_pr, kl_pr))*akl(l_pr, kl_pr, l, kl);
                end
                if kl_pr == kl && l_pr ~= l
                    I2(l, kl) = I2(l, kl) + (Pk_Meta(l_pr, kl_pr) + Pk_Payload(l_pr, kl_pr))*akl(l_pr, kl_pr, l, kl);
                end
                I3(l, kl) = I3(l, kl) + (Pk_Meta(l_pr, kl_pr) + Pk_Payload(l_pr, kl_pr))*ckl(l_pr, kl_pr, l, kl);
            end
        end

        if flag == "Meta"
            B(l, kl) = I1(l, kl) + I2(l, kl) + I3(l, kl) + Pk_Payload(l, kl)*akl(l, kl, l, kl) + Noise_power;
        else
            B(l, kl) = I1(l, kl) + I2(l, kl) + I3(l, kl) + Noise_power;
        end

        Gamma(l, kl) = Pk_Meta(l, kl)*akl(l, kl, l, kl) / B(l, kl);
        if Gamma(l, kl) == 0
            BLER(l, kl) = 1;
        else
            BLER(l, kl) = qfunc(f(Gamma(l, kl)));
        end
    end
end

end

%% Local Function for Bisection
function Gamma_min = Bisection(N, R, t)
V = @(gamma) 2.*gamma./(1 + gamma);
f = @(gamma) sqrt(log(2)*N./V(gamma)).*(log2(1 + gamma) - R);

gamma_max = 1000;
gamma_min = 1e-6;

while gamma_max - gamma_min >= 1e-4
    gamma_mid = (gamma_max + gamma_min)/2;
    if qfunc(f(gamma_mid)) < t
        gamma_max = gamma_mid;
    else
        gamma_min = gamma_mid;
    end
end

Gamma_min = gamma_mid;
end