

%% regularized logistic regression

% load data
load ionosphere
Ybool = strcmp(Y,'g');
X = X(:,3:end);

% size( X )
% size( Ybool )
% plot( X(:,1), Ybool );

%% Fit the model
% [B0,FitInfo0] = lassoglm(X,Ybool,'binomial','Lambda',0,'CV',10);
[B,FitInfo] = lassoglm(X,Ybool,'binomial','NumLambda',25,'CV',10); % 10 fold cross-validation

% check FitInfo0.SE vs FitInfo.SE
% check FitInfo0.Deviance vs FitInfo.Deviance

%% examine the trace
lassoPlot(B,FitInfo,'PlotType','CV');
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log');