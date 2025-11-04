clc; clear; close all;

%% System parameters
BW = 1e6; % System bandwidth
Noise_power = 10^((-173 - 30)/10)*BW; % Noise power
Pathloss = @(x, y)( 10.^-(3.53 + 3.76.*log10(x) + y./10) ); % large-scale pathloss model
% Pathloss = @(x)( 10.^-(3.53 + 3.76.*log10(x) ) ); % large-scale pathloss model
K = 10; % Number of UEs
M_H = 10; % Number of antenna per horizontal row
M_V = 10; % Number of rows
Mr = M_V * M_H; % Number of total antennas
d_H = 1/2; % Horizontal antenna spacing
d_V = 1/2; % Vertical antenna spacing
rs = 50; % Scatter radii
L_ = [1; 2; 3; 4]; % Pilot reuse factor/number of groups
Nth = 200; % Channel uses
Package_Meta = 12*8; % Metadata size in bits
Package_Payload = 12*8; % Payload size in bits
BLER_TH = 1e-4;
PtdBm = 30; % Transmit power in dBm
Pt = 10^((PtdBm - 30)/10); % Transmit power in W
Pp = Pt; % Pilot power in W
r_inner = 30; % inner radii of UE range
r_outer = 300; % outer radii of UE range

MC = 100; % The number of  repeat trials
num = length(L_);
NA_SE = zeros(num, 1);
NA_JE = zeros(num, 1);
NA_RS = zeros(num, 1);


for mc = 1:MC
    %% Generate UE locations
    Shadowing = 10*randn(K, 1);
    height_BS = 5 + 5*rand; % Height of BS
    vertical_UE = 1 + 2.*rand(K, 1); % Height of UEs
    horizontal_UE = r_inner + (r_outer - r_inner) * rand(K, 1); % Horizontal distance of UEs
    height = height_BS - vertical_UE; % Elevation of UEs
    phi = sort(2 * pi * rand(K, 1)); % Azimuth angle
    UEposition = [horizontal_UE .* cos(phi), horizontal_UE .* sin(phi)]; % UE positions
    dphi = atan(rs./horizontal_UE); % Horizontal angular spread
    theta_max = atan(height./(max(horizontal_UE-rs,0)));
    theta_min = atan(height./(horizontal_UE + rs));
    theta = (theta_max + theta_min)/2; % Elevation angles
    dtheta = (theta_max - theta_min)/2; % Vertical angular spread
    dUE = sqrt(vertical_UE.^2 + horizontal_UE.^2); % Distance between UEs and antenna arrays
    Beta = Pathloss(dUE, Shadowing); % large-scale pathloss


    %% Generate corelation matrices
    Rcorr = zeros(M_V*M_H, M_H*M_V, K);
    for k = 1:K % Include pathloss
        Rcorr(:, :, k) = Beta(k).*functionRlocalscattering3D(M_H, M_V, d_H, d_V, phi(k), dphi(k), theta(k), dtheta(k), 'Uniform');
    end

    for num_ = 1:num
        %% Pilot Asignment
        L = L_(num_);
        opts = statset('MaxIter', 500, 'Display', 'final'); % Kmeans cluster
        [labels, centers] = kmeans(UEposition, L, ...
            'Replicates', 10, ...
            'Options', opts, ...
            'Distance', 'sqeuclidean');

        [KL, ~] = histcounts(labels); % Number of users in each group

        Rcorr_r = zeros(M_V*M_H, M_H*M_V, L, max(KL)); % correlation matrices after pilot reuse
        Beta_r = zeros(L, max(KL)); % Large-scale pathloss after pilot reuse

        for l = 1:L
            group_users = find(labels == l);
            if ~isempty(group_users)
                Beta_r(l, 1:KL(l)) = Beta(group_users);
                Rcorr_r(:, :, l, 1:KL(l)) = Rcorr(:, :, group_users);
            end
        end


        %% Channel Estimation
        tau_p = max(KL); % Length of the pilot sequence
        h_true = sqrt(0.5).*randn(Mr, L, tau_p) + 1i*sqrt(0.5).*randn(Mr, L, tau_p); % True channel
        h_est = zeros(Mr, L, tau_p); % Estimated channel

        for l = 1:L
            for kl = 1:tau_p
                if kl <= KL(l)
                    Rsqrt = sqrtm(Rcorr_r(:, :, l, kl));
                    h_true(:, l, kl) = Rsqrt*h_true(:, l, kl); % Generate True channel
                else
                    h_true(:, l, kl) = zeros(Mr, 1);
                end
            end
        end

        Np = sqrt(Noise_power/2)*randn(Mr, 1) + 1i*sqrt(Noise_power/2)*randn(Mr, 1); % Generate noise vector

        Rcorr_est = zeros(Mr, Mr, L, tau_p); % Estimated channel correlation matrices
        y = zeros(Mr, tau_p); % Post-processed pilot signal
        for kl = 1:tau_p
            for l = 1:L
                y(:, kl) = y(:, kl) + sqrt(Pp*tau_p)*h_true(:, l, kl);
            end
            y(:, kl) = y(:, kl) + Np;
        end
        U = zeros(Mr, Mr, L, tau_p); % Auxiliary matrix
        U_inv = zeros(Mr, Mr, L, tau_p); % Inv of the auxiliary matrix

        for l = 1:L
            for kl = 1:KL(l)
                for l_pr = 1:L
                    for kl_pr = 1:KL(l)
                        if kl_pr == kl
                            U(:, :, l, kl) = U(:, :, l, kl) + Pp.*tau_p.*Rcorr_r(:, :, l_pr, kl_pr);
                        end
                    end
                end
                U(:, :, l, kl) = U(:, :, l, kl) + Noise_power.*eye(Mr);
                U_inv(:, :, l, kl) = inv(U(:, :, l, kl));
                h_est(:, l, kl) = sqrt(Pp*tau_p)*Rcorr_r(:, :, l, kl)*U_inv(:, :, l, kl)*y(:, kl);
                Rcorr_est(:, :, l, kl) = Pp.*tau_p.* Rcorr_r(:, :, l, kl) * U_inv(:, :, l, kl) * Rcorr_r(:, :, l, kl);
            end
        end

        Rcorr_error = Rcorr_r - Rcorr_est; % Estimation error

        %% Power Control
        Nd = Nth - tau_p;
        R_Meta = Package_Meta/Nd; % Coding rate for Metadata in bpcu
        R_Payload = Package_Payload/Nd; % Coding rate for Payload in bpcu
        R_JE = (Package_Payload + Package_Meta)/Nd; % Coding rate for whole package in bpcu

        NA_SE(num_) = NA_SE(num_) + Power_Allocation_SE...
            (BLER_TH, Nd, K, L, KL, Mr, tau_p, Package_Meta, Package_Payload, Pt, Noise_power, labels, h_est, Rcorr_error);
        NA_JE(num_) = NA_JE(num_) + Power_Allocation_JE(BLER_TH, Nd, R_JE, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);
        NA_RS(num_) = NA_RS(num_) + Power_Allocation_RS(Nd, BLER_TH, R_Meta, R_Payload, K, L, KL, Mr, tau_p, Pt, Noise_power, labels, h_est, Rcorr_error);

    end
    clc
    fprintf('this is the %d iteration \n', mc)
end

save("Data\NA_vs_Group.mat", "NA_JE", "NA_SE", "NA_RS");

