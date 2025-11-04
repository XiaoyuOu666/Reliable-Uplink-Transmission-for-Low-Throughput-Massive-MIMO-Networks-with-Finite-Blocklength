function [Gamma_Meta, Gamma_Payload, N_Meta_c, N_Meta_d, N_Meta_a, N_Meta_b] = ...
    Power_Allocation_SE(Nmax, K, L, KL, Mr, tau_p, Package_Meta, Package_Payload, Pt, Noise_power, labels, h_est, Rcorr_error)
% Power control for the SE scheme
% Note that this function is only for convergence evaluation

%% Begin Golden Search
N_Payload_max = Nmax;
N_Payload_min = 1;
gold = (sqrt(5) - 1)/2;
N_Payload_left = floor(N_Payload_min + (1 - gold)*(N_Payload_max - N_Payload_min));
N_Payload_right = floor(N_Payload_min + gold*(N_Payload_max - N_Payload_min));

N_Meta_left = Nmax - N_Payload_left;
N_Meta_right = Nmax - N_Payload_right;
R_Meta_left = Package_Meta/N_Meta_left;
R_Meta_right = Package_Meta/N_Meta_right;
R_Payload_left = Package_Payload/N_Payload_left;
R_Payload_right = Package_Payload/N_Payload_right;

% Record iteration values
N_Meta_c = {N_Payload_left};
N_Meta_d = {N_Payload_right};
N_Meta_a = {1};
N_Meta_b = {Nmax};

% Only for recoding SINR values
[Gamma_Meta, Gamma_Payload, ~] = Compute_Epsi(100, 100, Package_Meta/100, Package_Payload/100, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);

% Initializition for Gloden search
[~, ~, Epsi_left] = Compute_Epsi(N_Meta_left, N_Payload_left, R_Meta_left, R_Payload_left, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);
[~, ~, Epsi_right] = Compute_Epsi(N_Meta_right, N_Payload_right, R_Meta_right, R_Payload_right, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);

while N_Payload_max - N_Payload_min > 1 % Golden search for blocklength
    if Epsi_left >= Epsi_right
        N_Payload_min = N_Payload_left; % step 14
        N_Meta_a{end+1} = {N_Payload_min};
        N_Meta_b{end+1} = {N_Payload_max};
        N_Payload_left = N_Payload_right; % step 15
        N_Meta_c{end+1} = N_Payload_left;
        N_Payload_right = floor(N_Payload_min + gold*(N_Payload_max - N_Payload_min)); % step 16
        N_Meta_d{end+1} = N_Payload_right;
        N_Meta_right = Nmax - N_Payload_right; % step 17
        R_Meta_right = Package_Meta/N_Meta_right; % Update coding rate for metadata
        R_Payload_right = Package_Payload/N_Payload_right; % Update coding rate for payload
        Epsi_left = Epsi_right; % step 15
        [~, ~, Epsi_right] = Compute_Epsi(N_Meta_right, N_Payload_right, R_Meta_right, R_Payload_right, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);

    else
        N_Payload_max = N_Payload_right; % step 7
        N_Meta_a{end+1} = {N_Payload_min};
        N_Meta_b{end+1} = {N_Payload_max};
        N_Payload_right = N_Payload_left; % step 8
        N_Meta_d{end+1} = N_Payload_right;
        N_Payload_left = floor(N_Payload_min + (1 - gold)*(N_Payload_max - N_Payload_min)); % step 9
        N_Meta_c{end+1} = N_Payload_left;
        N_Meta_left = Nmax - N_Payload_left; % step 10
        R_Meta_left = Package_Meta/N_Meta_left;
        R_Payload_left = Package_Payload/N_Payload_left;
        Epsi_right = Epsi_left; % step 8
        [~, ~, Epsi_left] = Compute_Epsi(N_Meta_left, N_Payload_left, R_Meta_left, R_Payload_left, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);

    end
end

end


%% Local Function for Obtaining MMF SINRS
function [Gamma_Meta_iter, Gamma_Payload_iter, res] = Compute_Epsi(N_Meta, N_Payload, R_Meta, R_Payload, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error)

Max_iter = 10;
Pm = Pt.*ones(L, tau_p); % Initial metadata power
Pp = Pt.*ones(L, tau_p); % Initial payload power
Gamma_Meta_iter = zeros(Max_iter, 1); % Record convergence value
Gamma_Payload_iter = zeros(Max_iter, 1); % Record convergence value

for iter = 1:Max_iter
    % Generate auxiliary variables for metadata (after detector generation)
    [akl_Meta, ckl_Meta] = Detector_Generation(K, L, KL, Mr, Pm, h_est, labels, Rcorr_error, Noise_power, tau_p);
    %---------------------------------------------------------------------%
    % Accelarating CVX speed
    M = false(L, tau_p);
    for l = 1:L
        M(l, 1:KL(l)) = true;
    end

    % Signal gain
    G_akl = zeros(L, tau_p);
    for l = 1:L
        for k = 1:KL(l) % Only for active UE
            G_akl(l, k) = akl_Meta(l, k, l, k);
        end
    end

    total_dims = L * tau_p;
    A_mat = reshape(akl_Meta, [total_dims, total_dims]);
    C_mat = reshape(ckl_Meta, [total_dims, total_dims]);

    A_mat_interf = A_mat;
    A_mat_interf(1:total_dims+1:end) = 0;

    %--------------------------------------------------------------------%
    % CVX SOLVER
    % Finding Maximum SINR for metadata
    cvx_begin gp quiet
    cvx_solver mosek
        variable Pm(L, tau_p) nonnegative % optimization variable
        variable gamma_obj nonnegative % optimization variable
            % Vectorize power matrix 
            P_vec = reshape(Pm, [total_dims, 1]);

            % Compute total interference
            Interf_I_A_vec = (P_vec' * A_mat_interf)';
            Total_I_C_vec = (P_vec' * C_mat)';

            % Reshape interference matrices
            C1_C2 = reshape(Interf_I_A_vec, [L, tau_p]);
            C3    = reshape(Total_I_C_vec, [L, tau_p]);
            
            % Signal gain
            Signal = Pm .* G_akl;
            
            % True interference
            Interference = C1_C2 + C3 + Noise_power;

        maximize gamma_obj
        subject to
            % only constrains when ActiveMaskVec is true
            1e20 .* gamma_obj .* Interference(M) <= 1e20 .* Signal(M);
            Pm(M) <= Pt;
    cvx_end
    
    Gamma_Meta_iter(iter) = gamma_obj;
end

for iter = 1:Max_iter
    % Generate auxiliary variables for payload (after detector generation)
    [akl_Payload, ckl_Payload] = Detector_Generation(K, L, KL, Mr, Pp, h_est, labels, Rcorr_error, Noise_power, tau_p);
    %---------------------------------------------------------------------%
    % Accelarating CVX speed
    M = false(L, tau_p);
    for l = 1:L
        M(l, 1:KL(l)) = true;
    end

    % Signal gain
    G_akl = zeros(L, tau_p);
    for l = 1:L
        for k = 1:KL(l) % Only for active UE
            G_akl(l, k) = akl_Payload(l, k, l, k);
        end
    end

    total_dims = L * tau_p;
    A_mat = reshape(akl_Payload, [total_dims, total_dims]);
    C_mat = reshape(ckl_Payload, [total_dims, total_dims]);

    A_mat_interf = A_mat;
    A_mat_interf(1:total_dims+1:end) = 0;

    %--------------------------------------------------------------------%
    % CVX SOLVER
    % Finding Maximum SINR for metadata
    cvx_begin gp quiet
    cvx_solver mosek
        variable Pp(L, tau_p) nonnegative % optimization variable
        variable gamma_obj nonnegative % optimization variable
            % Vectorize power matrix 
            P_vec = reshape(Pp, [total_dims, 1]);

            % Compute total interference
            Interf_I_A_vec = (P_vec' * A_mat_interf)';
            Total_I_C_vec = (P_vec' * C_mat)';

            % Reshape interference matrices
            C1_C2 = reshape(Interf_I_A_vec, [L, tau_p]);
            C3    = reshape(Total_I_C_vec, [L, tau_p]);
            
            % Signal gain
            Signal = Pp .* G_akl;
            
            % True interference
            Interference = C1_C2 + C3 + Noise_power;

        maximize gamma_obj
        subject to
            % only constrains when ActiveMaskVec is true
            1e20 .* gamma_obj .* Interference(M) <= 1e20 .* Signal(M);
            Pp(M) <= Pt;
    cvx_end
    
    Gamma_Payload_iter(iter) = gamma_obj;
end


% Compute SINR for metadata and payload
V = @(gamma) 2.*gamma./(1 + gamma);
f_Meta = @(gamma) log(2)*sqrt(N_Meta./V(gamma)).*(log2(1 + gamma) - R_Meta);
f_Payload = @(gamma) log(2)*sqrt(N_Payload./V(gamma)).*(log2(1 + gamma) - R_Payload);

BLER_Payload = qfunc(f_Payload(Gamma_Payload_iter));
BLER_Meta = qfunc(f_Meta(Gamma_Meta_iter));

BLER = min(BLER_Payload + BLER_Meta);

res = min(BLER);

end