function res = Power_Allocation_SE_ZF_MRC(BLER_TH, Nmax, L, KL, tau_p, Package_Meta, Package_Payload, Pt, Noise_power, akl, ckl)
%%%%%%%%%%%%% POWER ALLOCATION Separate Encoding Scheme %%%%%%%%%%%%%%%%
% This function is used when the detector is selected as ZF or MRC %

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
Epsi_left = Compute_Epsi(N_Meta_left, N_Payload_left, R_Meta_left, R_Payload_left, L, KL, tau_p, Pt, Noise_power, akl, ckl, akl, ckl);
if Epsi_left <= BLER_TH
    res = 1;
    return
end

Epsi_right = Compute_Epsi(N_Meta_right, N_Payload_right, R_Meta_right, R_Payload_right, L, KL, tau_p, Pt, Noise_power, akl, ckl, akl, ckl);
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
        Epsi_right = Compute_Epsi(N_Meta_right, N_Payload_right, R_Meta_right, R_Payload_right, L, KL, tau_p, Pt, Noise_power, akl, ckl, akl, ckl);

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
        Epsi_left = Compute_Epsi(N_Meta_left, N_Payload_left, R_Meta_left, R_Payload_left, L, KL, tau_p, Pt, Noise_power, akl, ckl, akl, ckl);
        if Epsi_left <= BLER_TH
            res = 1;
            return
        end
    end
end

res = ((Epsi_left + Epsi_right)/2 <= BLER_TH)*1;
end


%% Local Function for Computing SINR
function [Gamma, BLER] = Compute_SINR(L, tau_p, Pk, akl, ckl, Noise_power, KL, N, R)
V = @(gamma) 2.*gamma./(1 + gamma);
f = @(gamma) log(2)*sqrt(N/V(gamma)).*(log2(1 + gamma) - R);

Gamma = nan(L, tau_p);
BLER = nan(L, tau_p);
I11 = zeros(L, tau_p); I12 = zeros(L, tau_p); I13 = zeros(L, tau_p);

for l = 1:L
    for kl = 1:KL(l)
        for l_pr = 1:L
            for kl_pr = 1:KL(l_pr)
                if kl_pr ~= kl
                    I11(l, kl) = I11(l, kl) + Pk(l_pr, kl_pr)*akl(l_pr, kl_pr, l, kl);
                end
                if kl_pr == kl && l_pr ~= l
                    I12(l, kl) = I12(l, kl) + Pk(l_pr, kl_pr)*akl(l_pr, kl_pr, l, kl);
                end
                I13(l, kl) = I13(l, kl) + Pk(l_pr, kl_pr)*ckl(l_pr, kl_pr, l, kl);
            end
        end
        Gamma(l, kl) = Pk(l, kl)*akl(l, kl, l, kl) / (I11(l, kl) + I12(l, kl) + I13(l, kl) + Noise_power);
        BLER(l, kl) = qfunc(f(Gamma(l, kl)));
    end
end

end


%% Local Function for Obtaining MMF SINR
function res = Compute_Epsi(N_Meta, N_Payload, R_Meta, R_Payload, L, KL, tau_p, Pt, Noise_power, akl_Meta, ckl_Meta, akl_Payload, ckl_Payload)

% Finding Maximum SINR for metadata
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

% Finding Maximum SINR for Payload
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
% Compute SINR for metadata and payload
[~, BLER_Meta] = Compute_SINR(L, tau_p, Pm, akl_Meta, ckl_Meta, Noise_power, KL, N_Meta, R_Meta);
[~, BLER_Payload] = Compute_SINR(L, tau_p, Pp, akl_Payload, ckl_Payload, Noise_power, KL, N_Payload, R_Payload);

BLER = min(BLER_Payload(:) + BLER_Meta(:));

res = min(BLER);

end