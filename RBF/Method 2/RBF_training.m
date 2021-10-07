function [ W, C ] = RBF_training( data, label, n_center_vec, sigmaFix )
%RBF_TRAINING Summary of this function goes here
%   Detailed explanation goes here

    % Using kmeans to find cinter vector
    [idx, C] = kmeans(data, n_center_vec);
    
    % Calulate sigma 
    n_data = size(data,1);
    
    % calculate K
    K = zeros(n_center_vec, 1);
    for i=1:n_center_vec
        K(i) = numel(find(idx == i));
    end
    
    sigma = sigmaFix * ones(n_center_vec, 1);

    % Calutate weights
    % kernel matrix
    k_mat = zeros(n_data, n_center_vec);
    
    for i=1:n_center_vec
        r = bsxfun(@minus, data, C(i,:)).^2;
        r = sum(r,2);
        k_mat(:,i) = exp((-r.^2)/(2*sigma(i)^2));
    end
    k_mat(isnan(k_mat)) = 0;
    
    W = pinv(k_mat'*k_mat)*k_mat'*label;
end

