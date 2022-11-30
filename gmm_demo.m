%% Multivariate Modeling (Gaussian Mixture Models)

%% create a GMM
mus = [1 2;-3 -5;0 0]; % Means
sigmas = cat(3,[2 0;0 .5],[1 0;0 1],[3 1;1 2]); % Covariances
p = [0.33 0.33 0.34]; % Mixing proportions

obj = gmdistribution( mus, sigmas, p );

ezsurf(@(x,y)pdf(obj,[x y]),[-10 10],[-10 10]);


%% Sample a GMM
figure;
Y = random(obj, 2000);
% Y = [Y; (rand(200, 2) - 0.5 )*12];
ezcontour(@(x,y)pdf(obj,[x y]),[-10 10],[-10 10])
hold on
scatter(Y(:,1),Y(:,2),10,'.');

%% Fit a GMM
figure;
scatter(Y(:,1),Y(:,2),10,'.'); hold on;
options = statset('Display','final');
obj_fit = gmdistribution.fit(Y,7,'Options',options);

% visualize results
ezcontour(@(x,y)pdf(obj_fit,[x y]), [-8 6], [-8 6] );
hold on;
plot( obj_fit.mu(:,1), obj_fit.mu(:,2), 'd', 'MarkerFaceColor','g' );
for k = 1 : size( obj_fit.mu, 1 )
    [V, D] = eig( obj_fit.Sigma(:,:,k));
    quiver( obj_fit.mu(k,1), obj_fit.mu(k,2), V(1,1) * D(1,1), V(2,1) * D(1,1) );
    quiver( obj_fit.mu(k,1), obj_fit.mu(k,2), V(1,2) * D(2,2), V(2,2) * D(2,2) );
    %pdeellip( obj_fit.mu(k,1), obj_fit.mu(k,2), D(1,1), D(2,2), 0);
end
hold off;

%% AIC and BIC for deciding the number of mixtures
% Both AIC and BIC are negative log-likelihoods for the data 
% with penalty terms for the number of estimated parameters.
num = 7;
AIC = zeros(1,num);
BIC = zeros(1,num);
logL = zeros(1, num);
gmm_try = cell(1,num);
for k = 1:num
    gmm_try{k} = gmdistribution.fit(Y,k);
    AIC(k)= gmm_try{k}.AIC;
    BIC(k) = gmm_try{k}.BIC;
    logL(k) = -gmm_try{k}.NlogL;
end
plot( 1:num, AIC, 'r*', 1:num, BIC, 'bo' );
title( 'AIC *  vs BIC o' );
[minAIC, nc_AIC] = min(AIC);
[minBIC, nc_BIC] = min(BIC);
figure; plot( logL );
title( 'log likelihood' );