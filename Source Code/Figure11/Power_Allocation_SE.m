function res = Power_Allocation_SE(BLER_TH, Nmax, K, L, KL, Mr, tau_p, Package_Meta, Package_Payload, Pt, Noise_power, labels, h_est, Rcorr_error)
%%%%%%%%%%%%% POWER ALLOCATION Separate Encoding Scheme %%%%%%%%%%%%%%%%
% This function directly return the feasibility when given the target BLER
% threshold. For BLER < BLER_TH, return 1, otherwise return 0;

%% Begin Golden Search
% Initialze Golden search parameters
% Corresponds to the steps 1-3 in Algorithm 2
N_Payload_max = Nmax;
N_Payload_min = 1;
gold = (sqrt(5) - 1)/2;
N_Payload_left = floor(N_Payload_min + (1 - gold)*(N_Payload_max - N_Payload_min));
N_Payload_right = floor(N_Payload_min + gold*(N_Payload_max - N_Payload_min));
R_Payload_left = Package_Payload/N_Payload_left;
R_Payload_right = Package_Payload/N_Payload_right;

N_Meta_left = Nmax - N_Payload_left;
N_Meta_right = Nmax - N_Payload_right;
R_Meta_left = Package_Meta/N_Meta_left;
R_Meta_right = Package_Meta/N_Meta_right;


% Conpute initial BLER (step 4 in Algorithm 2)
Epsi_left = Compute_Epsi(N_Meta_left, N_Payload_left, R_Meta_left, R_Payload_left, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);
if Epsi_left <= BLER_TH
    res = 1;
    return
end

Epsi_right = Compute_Epsi(N_Meta_right, N_Payload_right, R_Meta_right, R_Payload_right, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);
if Epsi_right <= BLER_TH
    res = 1;
    return
end

% Golden search for blocklength
% (Note that we use blocklength for the payload as the main variable for golden search, but it doesn't affect results)
while N_Payload_right - N_Payload_left > 0
    if Epsi_left >= Epsi_right

        % Update blocklength and coding rate
        N_Payload_min = N_Payload_left; % step 14
        N_Payload_left = N_Payload_right; % step 15
        Epsi_left = Epsi_right; % step 15
        N_Payload_right = floor(N_Payload_min + gold*(N_Payload_max - N_Payload_min)); % step 16
        N_Meta_right = Nmax - N_Payload_right; % step 17
        R_Meta_right = Package_Meta/N_Meta_right; % Update coding rate for metadata
        R_Payload_right = Package_Payload/N_Payload_right; % Update coding rate for payload
        % step 18
        Epsi_right = Compute_Epsi(N_Meta_right, N_Payload_right, R_Meta_right, R_Payload_right, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);

        % if the BLER value in the iteration is less than the target BLER
        % threshold, we can break the process to save the running time since it is already feasible
        if Epsi_right <= BLER_TH
            res = 1;
            return
        end

    else
        N_Payload_max = N_Payload_right; % step 7
        N_Payload_right = N_Payload_left; % step 8
        Epsi_right = Epsi_left; % step 8
        N_Payload_left = floor(N_Payload_min + (1 - gold)*(N_Payload_max - N_Payload_min)); % step 9
        N_Meta_left = Nmax - N_Payload_left; % step 10
        R_Meta_left = Package_Meta/N_Meta_left; % update coding rate for metadata
        R_Payload_left = Package_Payload/N_Payload_left; % update coding rate for payload
        % step 11
        Epsi_left = Compute_Epsi(N_Meta_left, N_Payload_left, R_Meta_left, R_Payload_left, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);
        if Epsi_left <= BLER_TH
            res = 1;
            return
        end
    end
end

res = ((Epsi_left + Epsi_right)/2 <= BLER_TH)*1;
end


%% Local Function for Obtaining MMF SINR
function res = Compute_Epsi(N_Meta, N_Payload, R_Meta, R_Payload, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error)
Max_iter = 10; % Maximum iteration number
Gamma_Meta_iter = zeros(Max_iter, 1); % Record SINR values in each iteration
Gamma_Payload_iter = zeros(Max_iter, 1);

% Initialize power coefficients
Pm = Pt.*ones(L, tau_p); % Metadata power
Pp = Pt.*ones(L, tau_p); % Payload power

% Finding Maximum SINR for metadata
for iter = 1:Max_iter
    % Update auxiliary variables after detertor generation
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

    if cvx_status == 'Solved'
        Gamma_Meta_iter(iter) = gamma_obj;
    else
        break
    end

    if iter >= 2 && Gamma_Meta_iter(iter) - Gamma_Meta_iter(iter - 1) <= 1e-3
        break
    end
end

% Finding Maximum SINR for Payload
for iter = 1:Max_iter
    % Update auxiliary variables after detertor generation
    [akl_Payload, ckl_Payload] = Detector_Generation(K, L, KL, Mr, Pp, h_est, labels, Rcorr_error, Noise_power, tau_p);
    %--------------------------------------------------------------------%
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

    if cvx_status == 'Solved'
        Gamma_Payload_iter(iter) = gamma_obj;
    else
        break
    end

    if iter >= 2 && Gamma_Payload_iter(iter) - Gamma_Payload_iter(iter - 1) <= 1e-3
        break
    end
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
