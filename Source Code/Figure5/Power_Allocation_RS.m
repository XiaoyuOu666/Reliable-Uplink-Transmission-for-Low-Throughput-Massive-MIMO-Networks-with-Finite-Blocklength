function [Epsi_max, Epsi_min, Z] = Power_Allocation_RS(N, R_Meta, R_Payload, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error)
% Power control for the RS scheme
% Note that this function is only for the convergence evaluation

%% Bisection
BLER_max = 1; % We set the maximum value of the BLER to 1
BLER_min = 1e-30; % We set the minimum value of the BLER to 10^(-30) (enough small)

% Prerecord max and min values of the BLER in each iteration
Epsi_max = {1};
Epsi_min = {1e-30};

% Begin bisection
while (BLER_max - BLER_min)/BLER_max >= 1e-2
    BLER_obj = 10^((log10(BLER_max) + log10(BLER_min))/2);
    t = BLER_obj/2;
    Gamma_min_Meta = Bisection(N, R_Meta, t); % Bisection to find minimum SINR required for metadata
    Gamma_min_Payload = Bisection(N, R_Payload, t); % Bisection to find minimum SINR required for Payload

    Pk_Meta = 0.5*Pt.*ones(L, tau_p); % Initialize power coefficient for Metadata
    Pk_Payload = 0.5*Pt.*ones(L, tau_p); % Initialize power coefficient for Payload

    max_iter = 10;
    Z = zeros(max_iter, 1);
    for iter = 1:max_iter
        % Update auxiliary variables (after updating detectors)
        [akl_Meta, ckl_Meta] = Detector_Generation_RS(K, L, KL, Mr, Pk_Payload, Pk_Meta, h_est, labels, Rcorr_error, Noise_power, tau_p, "Meta");
        [akl_Payload, ckl_Payload] = Detector_Generation_RS(K, L, KL, Mr, Pk_Payload, Pk_Meta, h_est, labels, Rcorr_error, Noise_power, tau_p, "Payload");
        
        % Update power coefficients
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
        
        % We set the power coefficients that are less than 0 to 0 to ensure
        % robust solving
        Pk_Meta(Pk_Meta < 0) = 0;
        Pk_Payload(Pk_Payload < 0) = 0;

        Z(iter) = Z;
    end

    if Z(max_iter) <= 1e-4 % feasible
        BLER_max = BLER_obj;
        Epsi_min{end+1} = BLER_min;
        Epsi_max{end+1} = BLER_max;
    else
        BLER_min = BLER_obj;
        Epsi_min{end+1} = BLER_min;
        Epsi_max{end+1} = BLER_max;
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