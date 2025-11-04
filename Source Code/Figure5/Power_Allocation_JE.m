function res = Power_Allocation_JE(K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error)
% POWER ALLOCATION Joint Encoding Scheme
% Note that this function is only for the convergence evaluation

Pd = Pt.*ones(L, tau_p); % initialize power coefficients for MMSE detector
Max_iter = 10;
Gamma = zeros(Max_iter, 1);

for iter = 1:Max_iter
    % Update detector
    [akl, ckl] = Detector_Generation(K, L, KL, Mr, Pd, h_est, labels, Rcorr_error, Noise_power, tau_p); % Update MMSE detector

    % Update power coefficients for the next iteration
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
    %---------------------------------------------------------------------%
    Gamma(iter) = gamma_obj;
    % Update power coefficients
    Pd = reshape(P_vec, L, tau_p);

    % Double check the CVX results
    % I1 = zeros(L, tau_p); % Auxiliary variables
    % I2 = zeros(L, tau_p);
    % I3 = zeros(L, tau_p);
    % GammaUE = zeros(L, tau_p); % SINRs of all UE
    % for l = 1:L
    %     for kl = 1:KL(l)
    %         for l_pr = 1:L
    %             for kl_pr = 1:KL(l_pr)
    %                 if kl_pr ~= kl
    %                     I1(l, kl) = I1(l, kl) + Pd(l_pr, kl_pr)*akl(l_pr, kl_pr, l, kl);
    %                 end
    %                 if kl_pr == kl && l_pr ~= l
    %                     I3(l, kl) = I3(l, kl) + Pd(l_pr, kl_pr)*akl(l_pr, kl_pr, l, kl);
    %                 end
    %                 I2(l, kl) = I2(l, kl) + Pd(l_pr, kl_pr)*ckl(l_pr, kl_pr, l, kl);
    %             end
    %         end
    %         GammaUE(l, kl) = Pd(l, kl)*akl(l, kl, l, kl) / ( I1(l, kl) + I2(l, kl) + I3(l, kl) + Noise_power);
    %     end
    % end
end

res = Gamma;

