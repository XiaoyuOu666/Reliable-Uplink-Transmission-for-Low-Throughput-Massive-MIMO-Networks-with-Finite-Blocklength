function [akl, ckl] = Detector_Generation(K, L, KL, Mr, Pd, h_est, labels, Rcorr_error, Noise_power, tau_p)
% Generate MMSE detector and a series of auxiliary variables

H_est = zeros(Mr, K);
Pr = ones(K, 1);
group_counters = zeros(L, 1);
Rcorr_error_r = zeros(Mr, Mr, K);
for k = 1:K % reconstruct estimation error, power matrix, and channel matrix
    current_group = labels(k);
    group_counters(current_group) = group_counters(current_group) + 1;
    H_est(:, k) = h_est(:, current_group, group_counters(current_group));
    Pr(k) = Pd(current_group, group_counters(current_group));
    Rcorr_error_r(:, :, k) = Rcorr_error(:, :, current_group, group_counters(current_group));
end

P = diag(Pr);
Cr = sum(Rcorr_error_r .* reshape(Pr, 1, 1, []), 3);

V_MMSE = (H_est*P*H_est' + Cr + Noise_power.*eye(Mr)) \ H_est*P;
for kl = 1:K
    if norm(V_MMSE(:, kl)) == 0
        V_MMSE(:, kl) = zeros(Mr, 1);
    else
        V_MMSE(:, kl) = V_MMSE(:, kl)./norm(V_MMSE(:, kl));
    end
end

V_MMSE_r = zeros(Mr, L, tau_p); % MMSE detector after pilot reuse
for l = 1:L
    group_users = find(labels == l);
    if ~isempty(group_users)
        V_MMSE_r(:, l, 1:KL(l)) = V_MMSE(:, group_users);
    end
end

% Compute a series of auxiliary variables
akl = zeros(L, tau_p, L, tau_p);
ckl = zeros(L, tau_p, L, tau_p);

for l = 1:L % compute
    for kl = 1:KL(l)
        for l_pr = 1:L
            for kl_pr = 1:KL(l_pr)
                akl(l_pr, kl_pr, l, kl) = abs(V_MMSE_r(:, l, kl)'*h_est(:, l_pr, kl_pr))^2;
                ckl(l_pr, kl_pr, l, kl) = abs(V_MMSE_r(:, l, kl)'*Rcorr_error(:, :, l_pr, kl_pr)*V_MMSE_r(:, l, kl));
            end
        end
    end
end