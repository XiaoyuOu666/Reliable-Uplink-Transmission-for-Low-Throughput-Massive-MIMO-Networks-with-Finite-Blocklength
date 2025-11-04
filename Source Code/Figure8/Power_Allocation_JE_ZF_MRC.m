function res = Power_Allocation_JE_ZF_MRC(BLER_TH, Nmax, R_JE, L, KL, tau_p, Pt, Noise_power, akl, ckl)
%%%%%%%%%%%%%%%% POWER ALLOCATION Joint Encoding Scheme %%%%%%%%%%%%%%%%%%
% This function is used when the detector is selected as ZF or MRC %

% Perform power control
%---------------------------------------------------------------------%
% Accelarating CVX speed
M = false(L, tau_p); % Mask matrix
for l = 1:L
    M(l, 1:KL(l)) = true;
end

% Signal gain
G_akl = zeros(L, tau_p);
for l = 1:L
    for k = 1:KL(l) % Only for active UE
        G_akl(l, k) = akl(l, k, l, k);
    end
end

total_dims = L * tau_p;
A_mat = reshape(akl, [total_dims, total_dims]);
C_mat = reshape(ckl, [total_dims, total_dims]);

A_mat_interf = A_mat;
A_mat_interf(1:total_dims+1:end) = 0;
%--------------------------------------------------------------------%
% CVX SOLVER
cvx_begin gp quiet
cvx_solver mosek

    % define P as a vector to simplify computing
    variable Pd(L, tau_p) nonnegative
    variable gamma_obj nonnegative
        
        % Vectorize power matrix 
        P_vec = reshape(Pd, [total_dims, 1]);

        % Compute total interference
        Interf_I_A_vec = (P_vec' * A_mat_interf)';
        Total_I_C_vec = (P_vec' * C_mat)';

        % Reshape interference matrices
        C1_C2 = reshape(Interf_I_A_vec, [L, tau_p]);
        C3    = reshape(Total_I_C_vec, [L, tau_p]);
        
        % Signal gain
        Signal = Pd .* G_akl;
        
        % True interference
        Interference = C1_C2 + C3 + Noise_power;

    
    maximize gamma_obj
    subject to
        % only constrains when ActiveMaskVec is true
        1e20 .* gamma_obj .* Interference(M) <= 1e20 .* Signal(M);
        Pd(M) <= Pt;
cvx_end

if cvx_status == 'Solved' % determine if the problem is solved
    gamma = gamma_obj; % record the SINR value for the each iteration
else
    gamma = 0; % if fails to solve, set the SINR to 0
end

% Compute BLER
V = @(gamma) 2.*gamma./(1 + gamma);
f = @(gamma) log(2)*sqrt(Nmax./V(gamma)).*(log2(1 + gamma) - R_JE);

BLER = qfunc(f(gamma));

BLER_JE_min = min(BLER); % Find the minimum error

res = (BLER_JE_min <= BLER_TH)*1; % if the minimum error is less than the target threshold, then res = 1, otherwise res = 0

end
