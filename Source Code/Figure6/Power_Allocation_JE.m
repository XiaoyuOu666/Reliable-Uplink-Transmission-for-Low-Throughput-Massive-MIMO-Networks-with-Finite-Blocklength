function res = Power_Allocation_JE(BLER_TH, Nmax, R_JE, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error)
%%%%%%%%%%%%%%%% POWER ALLOCATION Joint Encoding Scheme %%%%%%%%%%%%%%%%%%
% This function directly return the feasibility when given the target BLER
% threshold. For BLER < BLER_TH, return 1, otherwise return 0;

% initialize power coefficients
Pd = Pt.*ones(L, tau_p); 
Max_iter = 10;

% prepare to store the SINR in each iteration
gamma_iter = zeros(Max_iter, 1); 

for iter = 1:Max_iter
    % Update detector
    [akl, ckl] = Detector_Generation(K, L, KL, Mr, Pd, h_est, labels, Rcorr_error, Noise_power, tau_p); % Update MMSE detector

    % Update power coefficients for the next iteration
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
        gamma_iter(iter) = gamma_obj; % record the SINR value for the each iteration
    else
        gamma_iter(iter) = 0; % if fails to solve, set the SINR to 0
        break
    end

    if  iter >=2 && gamma_iter(iter) - gamma_iter(iter-1) <= 1e-2
        break
    end

end

%% Compute BLER
V = @(gamma) 2.*gamma./(1 + gamma);
f = @(gamma) log(2)*sqrt(Nmax./V(gamma)).*(log2(1 + gamma) - R_JE);

BLER = qfunc(f(gamma_iter));

BLER_JE_min = min(BLER); % Find the minimum error

% If the minimum error is less than the target threshold, then res = 1, otherwise res = 0
res = (BLER_JE_min <= BLER_TH)*1; 

end
